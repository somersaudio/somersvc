"""End-to-end training orchestration: upload, train on RunPod, download model."""

import os
import time
from pathlib import Path
from typing import Callable

from services.dataset_manager import DatasetManager
from services.job_store import update_job, get_job
from services.local_preprocessor import LocalPreprocessor
from services.r2_client import R2Client
from services.runpod_client import RunPodClient
from services.ssh_client import SSHClient


class TrainingOrchestrator:
    def __init__(
        self,
        api_key: str,
        ssh_key_path: str,
        dataset_manager: DatasetManager,
        models_dir: str,
        on_log: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
        resume_from: str = "",  # path to existing model checkpoint to resume from
        target_epochs: int = 0,  # auto-stop target — pod stops here even if app closes
    ):
        self.runpod = RunPodClient(api_key)
        self.ssh_key_path = ssh_key_path
        self.dataset_manager = dataset_manager
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.on_log = on_log or (lambda _: None)
        self.on_status = on_status or (lambda _: None)
        self.on_progress = on_progress or (lambda _: None)
        self.resume_from = resume_from
        self.target_epochs = int(target_epochs or 0)
        self._stop_requested = False
        self._pod_ip = None
        self._pod_port = None
        self._pod_id_for_stop = None  # set as soon as a pod is created/resumed
        self._current_status = "idle"
        self._app_closing = False  # set when user quits the app mid-training

    def request_stop(self):
        """Request stop. Behavior depends on phase:

        - Training running → drop the bash-loop sentinel + SIGTERM svc train,
          so Lightning saves a checkpoint cleanly. The main thread then falls
          through to PHASE 5 which downloads it.
        - Pre-training (creating pod, installing, uploading, preprocessing) →
          terminate the pod outright. There is nothing to "save" yet, and the
          user shouldn't keep paying. The main thread's blocking exec_command
          will error out as the SSH connection drops, then `finally` runs.

        Idempotent: a second call after the first is silently ignored. This
        matters because the auto-stop trigger sees both the G_<epoch>.pth and
        D_<epoch>.pth save lines, and we must not pod-terminate while PHASE 5
        is downloading.
        """
        if self._stop_requested:
            return
        self._stop_requested = True
        if self._current_status == "training" and self._pod_ip and self._pod_port:
            try:
                self.on_log("Saving checkpoint before stopping...")
                stop_ssh = SSHClient()
                stop_ssh.connect(self._pod_ip, self._pod_port, self.ssh_key_path)
                stop_ssh.exec_command(
                    "touch /tmp/somersvc_stop; "
                    "pkill -SIGTERM -f 'svc train' 2>/dev/null; "
                    "echo 'Graceful shutdown signal sent — waiting for checkpoint save...'; "
                    "sleep 8; "
                    "echo 'Training stopped'"
                )
                stop_ssh.close()
                self.on_log("Stop signal sent — will download latest checkpoint.")
                return
            except Exception as e:
                self.on_log(f"Stop signal error: {e} — falling back to pod terminate")

        # Pre-training (or graceful path failed). Kill the pod.
        pod_id = getattr(self, "_pod_id_for_stop", None)
        if pod_id:
            try:
                self.on_log("Stopping cloud GPU (training had not yet started — nothing to download)...")
                self.runpod.terminate_pod(pod_id)
                self.on_log("Cloud GPU stopped. No charges accruing.")
            except Exception as e:
                self.on_log(f"Pod terminate error: {e}")

    def request_app_close(self):
        """User is quitting the app while training is in progress. Don't
        terminate the pod, don't try to download anything — just let this
        thread die. The detached training script keeps running on the pod;
        ResumeWorker will reattach next time the app launches.
        """
        self._app_closing = True

    def _log(self, msg: str):
        self.on_log(msg)

    def _status(self, status: str):
        self._current_status = status
        self.on_status(status)

    def _check_stopped(self):
        """Raise RuntimeError("stopped") if the user has hit Stop. Called at
        every phase boundary so a pre-training stop bails out promptly
        instead of plowing through pod creation + SSH wait.
        """
        if self._stop_requested:
            raise RuntimeError("stopped")

    def run(self, job_id: str, speaker_name: str, f0_method: str = "dio") -> dict:
        """Run the full training pipeline. Returns updated job dict."""
        pod_id = None
        ssh = SSHClient()
        preprocessed_tar = None
        dataset_tar = None

        try:
            # ==========================================
            # PHASE 1: LOCAL WORK (free, no pod needed)
            # ==========================================

            # Step 1: Package dataset
            self._status("packaging")
            self.on_progress(5)
            self._log("Packaging dataset...")
            dataset_tar = self.dataset_manager.package(speaker_name)
            self._log(f"Dataset packaged: {os.path.getsize(dataset_tar) / 1024 / 1024:.1f} MB")
            self._check_stopped()

            # Step 2: Local preprocessing (runs on Mac while pod boots)
            use_local_preprocess = LocalPreprocessor.is_available()
            if use_local_preprocess:
                self._status("preprocessing")
                self.on_progress(10)
                self._log("Running preprocessing locally (saves pod time)...")
                preprocessor = LocalPreprocessor(on_log=self._log)
                preprocessed_tar = preprocessor.preprocess(dataset_tar, f0_method)
                self._log("Local preprocessing complete!")
            else:
                self._log("Local svc not available — preprocessing will run on pod")
            self._check_stopped()

            # ==========================================
            # PHASE 2: POD SETUP
            # ==========================================

            # Step 3: Create RunPod instance
            self._status("creating_pod")
            self.on_progress(20 if use_local_preprocess else 10)

            from services.job_store import load_config as _load_config
            _config = _load_config()
            cached_pod_id = _config.get("cached_pod_id")

            if cached_pod_id:
                self._log(f"Resuming cached pod {cached_pod_id}...")
                try:
                    import runpod as _runpod
                    _runpod.api_key = self.runpod.api_key
                    _runpod.resume_pod(cached_pod_id)
                    pod_id = cached_pod_id
                    self._pod_id_for_stop = pod_id
                    self._log(f"Pod resumed: {pod_id}")
                    update_job(job_id, pod_id=pod_id, status="creating_pod")
                except Exception as e:
                    self._log(f"Could not resume cached pod: {e}")
                    cached_pod_id = None
                    _config.pop("cached_pod_id", None)
                    from services.job_store import save_config as _save_config
                    _save_config(_config)

            if not cached_pod_id:
                # Last chance to bail before we actually spin up a pod and
                # start the billing meter.
                self._check_stopped()
                self._log("Creating RunPod GPU instance...")
                ssh_pub_key = ""
                pub_key_path = self.ssh_key_path + ".pub"
                if os.path.exists(pub_key_path):
                    with open(pub_key_path) as f:
                        ssh_pub_key = f.read().strip()

                # Read user's GPU tier preference from app config.
                try:
                    from services.job_store import load_config as _lc
                    _tier = (_lc() or {}).get("preferred_gpu_tier", "cheapest")
                except Exception:
                    _tier = "cheapest"
                pod = self.runpod.create_training_pod(
                    ssh_pub_key, on_log=self._log, preferred_tier=_tier,
                )
                pod_id = pod["id"]
                self._pod_id_for_stop = pod_id
                self._log(f"Pod created: {pod_id}")
                update_job(job_id, pod_id=pod_id, status="creating_pod")

            # If user clicked Stop while the API call was in flight, the pod
            # is now running and being billed — bail and let `finally`
            # terminate it.
            self._check_stopped()

            # Step 4: Wait for pod
            self._status("waiting_for_pod")
            self.on_progress(25 if use_local_preprocess else 15)
            self._log("Waiting for pod to start...")
            ip, port = self._wait_for_pod(pod_id)
            self._log(f"Pod ready at {ip}:{port}")
            self._pod_ip = ip
            self._pod_port = port
            update_job(job_id, pod_ip=ip, pod_ssh_port=port)

            # Step 5: Connect via SSH
            self._status("connecting")
            self._log("Connecting via SSH...")
            ssh.connect(ip, port, self.ssh_key_path)
            self._log("SSH connection established")

            if self._stop_requested:
                raise RuntimeError("stopped")

            # Step 6: Install dependencies (check if already installed)
            self._status("installing")
            self.on_progress(30 if use_local_preprocess else 20)
            update_job(job_id, status="installing")

            # Check if svc is already available (custom Docker image or cached pod)
            self._log("Checking for existing installation...")
            self._svc_available = False
            ssh.exec_command(
                # Require a 4.1.x install — 4.2.x bumped its dep mins past
                # what the pod's torch image provides, and a broken cached
                # pod (svc importable but unusable) would otherwise skip
                # the downgrade reinstall.
                "which svc > /dev/null 2>&1 && python3 -c 'import so_vits_svc_fork as s; v = getattr(s, \"__version__\", \"0\"); exit(0 if v.startswith(\"4.1.\") else 1)' 2>/dev/null && echo SVC_OK || echo SVC_MISSING",
                on_stdout=lambda line: setattr(self, '_svc_available', 'SVC_OK' in line),
            )

            if self._svc_available:
                self._log("so-vits-svc-fork already installed — skipping!")
            else:
                # Pin to the last 4.1.x line: 4.2.x bumped its minimum
                # deps to torch>=2.8 / lightning>=2.5 / numpy>=2 which
                # don't match the pod's PyTorch 2.1.0+cu118 image, so
                # installing the unpinned latest leaves training in a
                # silently broken state. 4.1.x works with our pinned
                # numpy<2 + lightning<2.5 stack.
                self._log("Installing so-vits-svc-fork (keeping existing PyTorch)...")
                exit_code = ssh.exec_command(
                    "pip install 'so-vits-svc-fork<4.2' --no-deps && "
                    "pip install cm-time click fastapi librosa 'lightning<2.5' 'matplotlib<3.9' "
                    "pebble praat-parselmouth psutil pysimplegui-4-foss pyworld "
                    "requests rich scipy sounddevice soundfile tensorboard "
                    "tensorboardx torchcrepe tqdm tqdm-joblib 'transformers<4.46' "
                    "'numpy<2' 'huggingface-hub<1' 'rich==13.9.4' 2>&1",
                    on_stdout=self._log,
                    on_stderr=self._log,
                )
                if exit_code != 0:
                    raise RuntimeError(f"Failed to install so-vits-svc-fork (exit code {exit_code})")

            # Heal svc-fork's plot_spectrogram_to_numpy on pods where
            # matplotlib >= 3.10 (tostring_rgb removed) or numpy >= 2
            # (np.fromstring binary mode removed) leaked through. The
            # replacement uses buffer_rgba() + alpha-strip and produces
            # an identically-shaped flat RGB byte array, so the next
            # data.reshape(W, H, 3) line in utils.py still works.
            # Idempotent: runs whether we just installed or skipped.
            self._log("Patching svc-fork utils.py for matplotlib/numpy compat...")
            patch_script = (
                "import so_vits_svc_fork, pathlib; "
                "p = pathlib.Path(so_vits_svc_fork.__file__).parent / 'utils.py'; "
                "s = p.read_text(); "
                "old = 'np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep=\\\"\\\")'; "
                "new = 'np.asarray(fig.canvas.buffer_rgba())[..., :3].reshape(-1)'; "
                "s2 = s.replace(old, new); "
                "p.write_text(s2); "
                "print('patched' if s != s2 else 'already-patched')"
            )
            ssh.exec_command(
                f'python3 -c "{patch_script}"',
                on_stdout=self._log,
                on_stderr=self._log,
            )

            # Verify CUDA
            ssh.exec_command(
                "python3 -c \"import torch; print('PyTorch:', torch.__version__); "
                "print('CUDA available:', torch.cuda.is_available()); "
                "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\"",
                on_stdout=self._log,
                on_stderr=self._log,
            )
            self._log("Installation complete")

            if self._stop_requested:
                raise RuntimeError("stopped")

            # ==========================================
            # PHASE 3: UPLOAD & PREPROCESS
            # ==========================================

            self._status("uploading")
            self.on_progress(40 if use_local_preprocess else 30)
            self._log("Uploading data...")
            update_job(job_id, status="uploading")
            ssh.exec_command("mkdir -p /workspace")

            if use_local_preprocess and preprocessed_tar:
                # Upload preprocessed data — skip preprocessing on pod entirely
                self._log("Uploading preprocessed data (skipping pod preprocessing)...")
                ssh.upload_file(preprocessed_tar, "/workspace/preprocessed.tar.gz")
                ssh.exec_command("cd /workspace && tar xzf preprocessed.tar.gz && rm preprocessed.tar.gz")
                self._log("Preprocessed data uploaded!")
                # Clean up local temp files
                os.unlink(preprocessed_tar)
                os.unlink(dataset_tar)
            else:
                # Upload raw dataset and preprocess on pod
                ssh.upload_file(dataset_tar, "/workspace/dataset.tar.gz")
                ssh.exec_command("cd /workspace && tar xzf dataset.tar.gz && rm dataset.tar.gz")
                self._log("Dataset uploaded and extracted")
                os.unlink(dataset_tar)

                # Preprocess on pod
                self._status("preprocessing")
                self.on_progress(45)
                self._log("Running preprocessing on pod...")
                update_job(job_id, status="preprocessing")

                for cmd_label, cmd in [
                    ("Resampling audio", "cd /workspace && svc pre-resample"),
                    ("Generating config", "cd /workspace && svc pre-config"),
                    ("Extracting features", f"cd /workspace && svc pre-hubert -fm {f0_method}"),
                ]:
                    self._log(f"  {cmd_label}...")
                    exit_code = ssh.exec_command(cmd, on_stdout=self._log, on_stderr=self._log)
                    if exit_code != 0:
                        self._log(f"  Command was: {cmd}")
                        raise RuntimeError(f"{cmd_label} failed (exit code {exit_code})")

                self._log("Preprocessing complete")

            # ==========================================
            # PHASE 4: TRAINING (with optional resume)
            # ==========================================

            self._status("training")
            self.on_progress(55)
            self._log("Starting training...")
            update_job(job_id, status="training")

            # Upload existing checkpoint if resuming. We track the resume
            # epoch (resume_offset) so the post-training rename pass can
            # produce filenames that reflect TOTAL epoch count instead of
            # the trainer's session-local 0..delta range.
            resume_offset = 0
            if self.resume_from and os.path.exists(self.resume_from):
                resume_dir = os.path.dirname(self.resume_from)
                resume_file = os.path.basename(self.resume_from)
                self._log(f"Uploading checkpoint for resume: {resume_file}")
                ssh.exec_command("mkdir -p /workspace/logs/44k")
                ssh.upload_file(self.resume_from, f"/workspace/logs/44k/{resume_file}")
                # Also upload D checkpoint if it exists
                d_file = resume_file.replace("G_", "D_")
                d_path = os.path.join(resume_dir, d_file)
                if os.path.exists(d_path):
                    ssh.upload_file(d_path, f"/workspace/logs/44k/{d_file}")
                    self._log(f"Uploaded discriminator checkpoint: {d_file}")
                # Upload config if it exists in the model dir
                config_path = os.path.join(resume_dir, "config.json")
                if os.path.exists(config_path):
                    ssh.exec_command("mkdir -p /workspace/configs/44k")
                    ssh.upload_file(config_path, "/workspace/configs/44k/config.json")

                # Capture the resume epoch from the filename. The trainer
                # itself starts at 0 — we no longer try to patch its epoch
                # counter (Lightning often overrides the patch); instead we
                # cap max_epochs at the delta and rename output files post
                # facto so the on-disk sequence reads as a continuation.
                _re_str = resume_file.replace("G_", "").replace(".pth", "")
                if _re_str.isdigit():
                    resume_offset = int(_re_str)
                self._log(
                    f"Resume checkpoint uploaded (offset {resume_offset}) — "
                    f"trainer will run the delta and final files will be "
                    f"renamed back to total epoch count."
                )

            # --- OPTIMIZATION 1: Increase batch size ---
            self._log("Optimizing training config for max speed...")
            target_epochs_for_pod = int(self.target_epochs or 0)
            # Trainer starts at 0 each run (Lightning ignores G_<N>.pth's
            # internal epoch on a weights-only load), so pass it the DELTA
            # for resume runs. The post-training rename pass adds the
            # offset back so on-disk filenames reflect total epochs.
            if resume_offset > 0 and target_epochs_for_pod > 0:
                effective_max_epochs = max(1, target_epochs_for_pod - resume_offset)
            else:
                effective_max_epochs = target_epochs_for_pod
            ssh.exec_command(
                "cat > /tmp/optimize.py << 'OPTEOF'\n"
                "import json, glob, subprocess\n"
                f"TARGET_EPOCHS = {effective_max_epochs}\n"
                "try:\n"
                "    out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits']).decode().strip()\n"
                "    vram_mb = int(out.split(chr(10))[0])\n"
                "except:\n"
                "    vram_mb = 24000\n"
                "if vram_mb >= 80000:\n"
                "    target_batch = 192\n"
                "elif vram_mb >= 45000:\n"
                "    target_batch = 96\n"
                "elif vram_mb >= 20000:\n"
                "    target_batch = 48\n"
                "else:\n"
                "    target_batch = 32\n"
                "configs = glob.glob('/workspace/configs/*/config.json')\n"
                "for cfg_path in configs:\n"
                "    with open(cfg_path) as f:\n"
                "        cfg = json.load(f)\n"
                "    original_batch = cfg.get('train', {}).get('batch_size', 16)\n"
                "    if 'train' not in cfg:\n"
                "        cfg['train'] = {}\n"
                "    cfg['train']['batch_size'] = target_batch\n"
                "    original_lr = cfg.get('train', {}).get('learning_rate', 0.0001)\n"
                "    scale = target_batch / original_batch\n"
                "    new_lr = original_lr * (scale ** 0.5)\n"
                "    cfg['train']['learning_rate'] = new_lr\n"
                "    # Log every step so users see real-time progress in the app\n"
                "    cfg['train']['log_interval'] = 1\n"
                "    # eval_interval (epochs) controls how often a checkpoint is\n"
                "    # written, which is also when the parser hears the epoch\n"
                "    # number. Scale with batch size: small batches → epoch every\n"
                "    # checkpoint, big batches → spaced out so we don't spam disk.\n"
                "    eval_int = max(1, min(25, target_batch // 16))\n"
                "    cfg['train']['eval_interval'] = eval_int\n"
                "    # Bake the auto-stop target into the trainer's max-epoch so\n"
                "    # the pod stops at target even if the app is closed and the\n"
                "    # UI's auto-stop never fires.\n"
                "    if TARGET_EPOCHS > 0:\n"
                "        cfg['train']['epochs'] = TARGET_EPOCHS\n"
                "    with open(cfg_path, 'w') as f:\n"
                "        json.dump(cfg, f, indent=2)\n"
                "    print(f'VRAM: {vram_mb}MB | Batch: {original_batch} -> {target_batch} | LR: {original_lr} -> {new_lr:.6f} | eval_interval: {eval_int} | epochs: {cfg[\"train\"].get(\"epochs\", \"default\")}')\n"
                "OPTEOF\n"
                "python3 /tmp/optimize.py",
                on_stdout=self._log,
            )

            # --- OPTIMIZATION 2: Cache dataset in RAM ---
            self._log("Caching dataset in RAM...")
            ssh.exec_command(
                "if [ -d /workspace/dataset ]; then "
                "cp -r /workspace/dataset /dev/shm/dataset 2>/dev/null && "
                "rm -rf /workspace/dataset && "
                "ln -sf /dev/shm/dataset /workspace/dataset && "
                "echo 'Dataset cached in RAM'; "
                "else echo 'No dataset dir to cache'; fi",
                on_stdout=self._log,
            )

            # Drop a helper script for halving batch_size after an OOM crash
            ssh.exec_command(
                "cat > /tmp/halve_batch.py << 'BPYEOF'\n"
                "import json, glob, sys\n"
                "min_batch = 4\n"
                "for cfg_path in glob.glob('/workspace/configs/*/config.json'):\n"
                "    with open(cfg_path) as f: cfg = json.load(f)\n"
                "    cur = int(cfg.get('train', {}).get('batch_size', 16))\n"
                "    new = max(min_batch, cur // 2)\n"
                "    if new == cur:\n"
                "        print(f'Already at minimum batch_size={cur}; cannot halve further')\n"
                "        sys.exit(2)\n"
                "    cfg.setdefault('train', {})['batch_size'] = new\n"
                "    cur_lr = cfg['train'].get('learning_rate', 0.0001)\n"
                "    cfg['train']['learning_rate'] = cur_lr / (2 ** 0.5)\n"
                "    new_eval = max(1, min(25, new // 16))\n"
                "    cfg['train']['eval_interval'] = new_eval\n"
                "    with open(cfg_path, 'w') as f: json.dump(cfg, f, indent=2)\n"
                "    print(f'Halved batch_size: {cur} -> {new}, LR scaled, eval_interval -> {new_eval}')\n"
                "BPYEOF\n"
                "echo Halve helper installed",
            )
            # Clear any leftover stop sentinel from prior runs on this pod
            ssh.exec_command("rm -f /tmp/somersvc_stop")

            # Stage R2 credentials on the pod so the detached runner can
            # auto-upload when training finishes — protects the model even
            # if the app is closed or the pod gets evicted afterwards.
            r2_for_runner = R2Client()
            r2_runner_ready = False
            r2_prefix_for_runner = f"models/{speaker_name}/{job_id}"
            if r2_for_runner.is_configured():
                # Record the R2 prefix + recovery metadata on the job NOW so
                # check_pending_downloads can find the model on next launch
                # even if the app is closed before PHASE 5 ever runs.
                try:
                    _dataset_files = self.dataset_manager.list_files(speaker_name)
                    _total_dur = sum(f.get("duration", 0) or 0 for f in _dataset_files)
                    _total_clips = len(_dataset_files)
                except Exception:
                    _total_dur, _total_clips = 0.0, 0
                _prev_epochs = 0
                _meta_path = self.models_dir / speaker_name / "metadata.json"
                if _meta_path.exists():
                    try:
                        import json as _json
                        with open(_meta_path) as _f:
                            _prev_epochs = _json.load(_f).get("epochs", 0)
                    except Exception:
                        pass
                update_job(
                    job_id,
                    r2_prefix=r2_prefix_for_runner,
                    speaker_name=speaker_name,
                    previous_epochs=_prev_epochs,
                    resume=bool(self.resume_from),
                    dataset_duration=round(_total_dur, 1),
                    dataset_clips=_total_clips,
                )
                self._log("Staging R2 upload script on pod (auto-save on training end)...")
                ssh.exec_command(
                    "pip install boto3 -q 2>/dev/null", on_stdout=self._log,
                )
                upload_script = r2_for_runner.get_upload_script()
                ssh.exec_command(
                    "cat > /tmp/upload_r2.py << 'R2EOF'\n"
                    f"{upload_script}\n"
                    "R2EOF\n"
                    "echo R2 script staged"
                )
                r2_runner_ready = True

            # Build the post-training upload block injected into the runner.
            # If R2 isn't configured, fall back to "no upload" — PHASE 5 will
            # download directly from the pod over SSH instead.
            import shlex
            job_id_q = shlex.quote(job_id)
            speaker_q = shlex.quote(speaker_name)
            if r2_runner_ready:
                upload_block = (
                    "if ls /workspace/logs/44k/G_*.pth >/dev/null 2>&1; then\n"
                    "  echo 'Uploading model to R2 (so the model is safe even if the pod is evicted)...'\n"
                    f"  SVC_JOB_ID={job_id_q} SVC_SPEAKER={speaker_q} "
                    "python3 /tmp/upload_r2.py 2>&1 || echo 'R2 upload failed — falling back to SSH download'\n"
                    "else\n"
                    "  echo 'No checkpoint produced — nothing to upload.'\n"
                    "fi\n"
                )
            else:
                upload_block = (
                    "echo 'R2 not configured — model stays on pod for direct SSH download.'\n"
                )

            # Auto-stop watcher block: independent of Lightning's max_epochs,
            # this polls G_*.pth filenames and SIGTERMs svc train when the
            # latest checkpoint epoch hits target. Belt + suspenders to the
            # cfg.train.epochs cap. The watcher uses the SESSION-LOCAL target
            # (delta) because the rename to total epochs happens after the
            # training loop exits, not during it.
            watcher_target = effective_max_epochs
            if watcher_target > 0:
                watcher_block = (
                    f"TRAIN_TARGET={watcher_target}\n"
                    "(\n"
                    "  sleep 30\n"
                    "  while [ ! -f /tmp/svc_train_runner.done ]; do\n"
                    "    if [ -d /workspace/logs/44k ]; then\n"
                    "      latest=$(find /workspace/logs/44k -name 'G_*.pth' "
                    "-newer /tmp/svc_session_start 2>/dev/null "
                    "| sed -E 's@.*G_([0-9]+)\\.pth@\\1@' | sort -n | tail -1)\n"
                    "      if [ -n \"$latest\" ] && [ \"$latest\" -ge \"$TRAIN_TARGET\" ]; then\n"
                    "        echo \"Auto-stop watcher: epoch $latest reached target $TRAIN_TARGET — stopping.\"\n"
                    "        touch /tmp/somersvc_stop\n"
                    "        pkill -SIGTERM -f 'svc train' 2>/dev/null\n"
                    "        break\n"
                    "      fi\n"
                    "    fi\n"
                    "    sleep 15\n"
                    "  done\n"
                    ") &\n"
                    "WATCHER_PID=$!\n"
                )
                watcher_kill = "kill $WATCHER_PID 2>/dev/null\n"
            else:
                watcher_block = ""
                watcher_kill = ""

            # Post-training rename pass: when a resume run finishes, rename
            # every G_<N>.pth / D_<N>.pth produced during this session by
            # adding the resume offset, so the on-disk filenames reflect
            # the TOTAL epoch count instead of the trainer's session-local
            # 0..delta range. Drops the original resume checkpoints since
            # the renamed final ones supersede them.
            if resume_offset > 0:
                rename_block = (
                    f"RESUME_OFFSET={resume_offset}\n"
                    "if [ -d /workspace/logs/44k ]; then\n"
                    "  echo \"Renaming session checkpoints (+$RESUME_OFFSET epochs)...\"\n"
                    "  find /workspace/logs/44k \\( -name 'G_*.pth' -o -name 'D_*.pth' \\) "
                    "-newer /tmp/svc_session_start | while read f; do\n"
                    "    base=$(basename \"$f\")\n"
                    "    num=$(echo \"$base\" | sed -E 's/[GD]_([0-9]+)\\.pth/\\1/')\n"
                    "    prefix=$(echo \"$base\" | sed -E 's/([GD])_[0-9]+\\.pth/\\1/')\n"
                    "    new_num=$((num + RESUME_OFFSET))\n"
                    "    mv \"$f\" \"/workspace/logs/44k/${prefix}_${new_num}.pth\" "
                    "&& echo \"  $base -> ${prefix}_${new_num}.pth\"\n"
                    "  done\n"
                    f"  rm -f /workspace/logs/44k/G_{resume_offset}.pth\n"
                    f"  rm -f /workspace/logs/44k/D_{resume_offset}.pth\n"
                    "fi\n"
                )
            else:
                rename_block = ""

            # Write the retry/OOM loop to a script and launch it DETACHED via
            # setsid. This way closing the app (which closes our SSH channel)
            # cannot SIGHUP the training — it lives in its own session.
            self._log("Starting training (detached — safe to close the app)...")
            ssh.exec_command(
                "cat > /tmp/svc_train_runner.sh << 'TRAINEOF'\n"
                "#!/bin/bash\n"
                "cd /workspace\n"
                "pip install 'rich==13.9.4' -q 2>/dev/null\n"
                # Mark session start AFTER any pre-existing checkpoint files
                # so the post-training rename pass can identify which files
                # were written by this run via `find -newer`.
                "touch /tmp/svc_session_start\n"
                "sleep 1\n"
                f"{watcher_block}"
                "FINAL_EXIT=0\n"
                "for attempt in 1 2 3 4 5; do\n"
                "  if [ -f /tmp/somersvc_stop ]; then\n"
                "    echo \"Stop requested before attempt $attempt — exiting retry loop\"\n"
                "    break\n"
                "  fi\n"
                "  echo \"Training attempt $attempt...\"\n"
                "  svc train --model-path /workspace/logs/44k 2>&1 | tee /tmp/svc_train.log\n"
                "  EXIT=${PIPESTATUS[0]}\n"
                "  FINAL_EXIT=$EXIT\n"
                "  if [ $EXIT -eq 0 ]; then break; fi\n"
                "  if [ -f /tmp/somersvc_stop ]; then\n"
                "    echo \"Stop requested — not restarting\"\n"
                "    break\n"
                "  fi\n"
                "  if grep -q 'CUDA out of memory\\|OutOfMemoryError' /tmp/svc_train.log; then\n"
                "    echo \"OOM detected — halving batch size before retry...\"\n"
                "    python3 /tmp/halve_batch.py || break\n"
                "  fi\n"
                "  echo \"Training crashed (exit $EXIT), restarting in 5s...\"\n"
                "  sleep 5\n"
                "done\n"
                f"{watcher_kill}"
                # Rename session checkpoints to reflect total epochs (no-op
                # for non-resume runs — rename_block is empty in that case).
                f"{rename_block}"
                # Best-effort R2 upload BEFORE writing the .done sentinel so\n"
                # the watching app knows when it's safe to skip the redundant\n"
                # PHASE-5 upload and proceed straight to download.\n"
                f"{upload_block}"
                "echo $FINAL_EXIT > /tmp/svc_train_runner.done\n"
                "TRAINEOF\n"
                "chmod +x /tmp/svc_train_runner.sh\n"
                "rm -f /tmp/svc_train_runner.log /tmp/svc_train_runner.done /tmp/svc_train_runner.pid\n"
                # `setsid -f` forks first; without -f, util-linux setsid run\n"
                # under `&` fails with EPERM (pgrp-leader) and bash is never\n"
                # exec'd, leaving an empty log and a dead PID file.\n"
                "setsid -f bash /tmp/svc_train_runner.sh < /dev/null > /tmp/svc_train_runner.log 2>&1\n"
                "sleep 1\n"
                "RUNNER_PID=$(pgrep -f 'bash /tmp/svc_train_runner.sh' | head -n1)\n"
                "echo \"${RUNNER_PID:-unknown}\" > /tmp/svc_train_runner.pid\n"
                "echo \"Training detached (PID ${RUNNER_PID:-unknown}) — log at /tmp/svc_train_runner.log\"",
                on_stdout=self._log,
            )

            # Tail the detached log + wait for the .done sentinel. If this
            # exec_command's channel dies (e.g. user closes the app), the
            # detached training keeps going untouched.
            exit_code = ssh.exec_command(
                "while [ ! -f /tmp/svc_train_runner.log ]; do sleep 1; done; "
                "tail -F /tmp/svc_train_runner.log 2>/dev/null & TAIL_PID=$!; "
                "while [ ! -f /tmp/svc_train_runner.done ]; do sleep 2; done; "
                "sleep 2; "
                "kill $TAIL_PID 2>/dev/null; "
                "exit $(cat /tmp/svc_train_runner.done 2>/dev/null || echo 1)",
                on_stdout=self._log,
                on_stderr=self._log,
            )
            if self._app_closing:
                # User closed the app. We must not touch the pod. Bail
                # immediately — the rest of run() (PHASE 5, finally pod
                # cleanup) is gated on _app_closing further down.
                return
            if exit_code != 0 and not self._stop_requested:
                raise RuntimeError(f"Training failed after retries (exit code {exit_code})")
            if self._stop_requested:
                self._log("Training stopped by user — downloading model...")
            else:
                self._log("Training complete!")

            # ==========================================
            # PHASE 5: UPLOAD TO R2 + DOWNLOAD
            # ==========================================

            self._status("uploading_model")
            self.on_progress(85)

            # Gather metadata info before we lose SSH
            import json
            dataset_files = self.dataset_manager.list_files(speaker_name)
            total_duration = sum(f.get("duration", 0) or 0 for f in dataset_files)
            total_clips = len(dataset_files)

            # Save job metadata so we can recover if app closes
            model_dir = self.models_dir / speaker_name
            meta_path = model_dir / "metadata.json"
            previous_epochs = 0
            previous_batch = 16
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        old_meta = json.load(f)
                    previous_epochs = old_meta.get("epochs", 0)
                    previous_batch = old_meta.get("batch_size", 16)
                except Exception:
                    pass

            r2 = R2Client()
            r2_prefix = f"models/{speaker_name}/{job_id}"

            if r2.is_configured():
                # Always record where the model lives in R2 so
                # check_pending_downloads / ResumeWorker can find it later.
                update_job(job_id, status="uploading_model",
                           r2_prefix=r2_prefix,
                           speaker_name=speaker_name,
                           previous_epochs=previous_epochs,
                           previous_batch=previous_batch,
                           resume=bool(self.resume_from),
                           dataset_duration=round(total_duration, 1),
                           dataset_clips=total_clips)

                # If the detached runner already uploaded (it does this when
                # R2 was staged in PHASE 4), skip the redundant upload and go
                # straight to download.
                if r2.file_exists(f"{r2_prefix}/_complete.json"):
                    self._log("Model already uploaded by pod runner — skipping re-upload.")
                    self._status("downloading")
                    self.on_progress(90)
                    self._log("Downloading model from cloud...")
                    self._download_from_r2(job_id, speaker_name, r2, r2_prefix,
                                           previous_epochs, total_duration, total_clips)
                    self._status("completed")
                    self.on_progress(100)
                    update_job(job_id, status="completed")
                    self._log("Training job completed!")
                    return get_job(job_id)

                # Otherwise (runner upload failed or wasn't staged), upload now.
                self._log("Uploading model to cloud storage...")
                ssh.exec_command("pip install boto3 -q 2>/dev/null", on_stdout=self._log)
                upload_script = r2.get_upload_script()
                ssh.exec_command(f"cat > /tmp/upload_r2.py << 'R2EOF'\n{upload_script}\nR2EOF")
                exit_code = ssh.exec_command(
                    f'SVC_JOB_ID="{job_id}" SVC_SPEAKER="{speaker_name}" python3 /tmp/upload_r2.py',
                    on_stdout=self._log,
                    on_stderr=self._log,
                )
                if exit_code != 0:
                    raise RuntimeError("Failed to upload model to R2")
                self._log("Model uploaded to cloud storage!")

                # Now download from R2 to local (faster than from pod, and survives pod death)
                self._status("downloading")
                self.on_progress(90)
                self._log("Downloading model from cloud...")
                self._download_from_r2(job_id, speaker_name, r2, r2_prefix,
                                       previous_epochs, total_duration, total_clips)
            else:
                # Fallback: direct download from pod (old behavior)
                self._log("R2 not configured — downloading directly from pod...")
                self._status("downloading")
                self.on_progress(90)
                update_job(job_id, status="downloading")

                model_dir.mkdir(parents=True, exist_ok=True)
                files = ssh.list_remote_files("/workspace/logs/44k")
                g_files = sorted([f for f in files if f.startswith("G_") and f.endswith(".pth")])
                if not g_files:
                    raise RuntimeError("No model checkpoint found after training")

                latest_g = g_files[-1]
                self._log(f"Downloading checkpoint: {latest_g}")
                ssh.download_file(f"/workspace/logs/44k/{latest_g}", str(model_dir / latest_g))
                ssh.download_file("/workspace/configs/44k/config.json", str(model_dir / "config.json"))
                self._save_metadata(model_dir, latest_g, previous_epochs, total_duration, total_clips)
                self._log("Model downloaded successfully")

            # Done
            self._status("completed")
            self.on_progress(100)
            update_job(job_id, status="completed")
            self._log("Training job completed!")
            return get_job(job_id)

        except Exception as e:
            if self._stop_requested:
                self._log("Training cancelled by user.")
                update_job(job_id, status="failed", error="cancelled")
            else:
                self._log(f"ERROR: {e}")
                update_job(job_id, status="failed", error=str(e))
            raise
        finally:
            # If the user is closing the app, do NOT touch the pod or the
            # SSH session — let training continue undisturbed. ResumeWorker
            # will reattach on next launch.
            if self._app_closing:
                self._log("App closing — leaving training running on pod.")
                return

            # Best-effort salvage: if the user cancelled and PHASE 5 didn't
            # already finish the download, check the pod for any G_*.pth that
            # we can pull down before tearing it down. If nothing exists yet
            # (still in install/upload/preprocess), just terminate.
            if (self._stop_requested and pod_id
                    and self._pod_ip and self._pod_port):
                _job = get_job(job_id) if job_id else None
                _already_done = bool(_job and _job.get("status") == "completed")
                if not _already_done:
                    self._salvage_partial_checkpoint(speaker_name, job_id)

            ssh.close()
            # Clean up temp files if they still exist
            if dataset_tar and os.path.exists(dataset_tar):
                os.unlink(dataset_tar)
            if preprocessed_tar and os.path.exists(preprocessed_tar):
                os.unlink(preprocessed_tar)
            if pod_id:
                job = get_job(job_id) if job_id else None
                job_status = job.get("status", "") if job else ""
                r2_uploaded = job.get("r2_prefix") if job else None
                if self._stop_requested:
                    # User cancelled — pod terminate (salvage already ran above).
                    self._log("Terminating pod after cancellation...")
                    try:
                        self.runpod.terminate_pod(pod_id)
                        self._log("Pod terminated (billing stopped).")
                    except Exception as e:
                        self._log(f"Could not terminate pod: {e}. Terminate manually: {pod_id}")
                elif job_status == "completed" or r2_uploaded:
                    # Model is safe (downloaded or in R2) — terminate pod
                    self._log("Terminating RunPod instance...")
                    self.runpod.terminate_pod(pod_id)
                    self._log("Pod terminated (billing stopped)")
                else:
                    # Something failed before R2 upload — stop but preserve data
                    self._log("Stopping pod (preserving data — upload may have failed)...")
                    try:
                        import runpod
                        runpod.api_key = self.runpod.api_key
                        runpod.stop_pod(pod_id)
                        self._log(f"Pod stopped — data preserved. Terminate manually on runpod.io when ready.")
                        self._log(f"Pod ID: {pod_id}")
                    except Exception:
                        self._log(f"Could not stop pod. Terminate manually: {pod_id}")

    def _salvage_partial_checkpoint(self, speaker_name: str, job_id: str):
        """When user cancels mid-run, try to download any G_*.pth that exists.

        If no checkpoint exists yet (still in install/upload/preprocess), this
        is a no-op besides logging. Uses a fresh SSH session so a half-dead
        primary connection doesn't block the salvage attempt.
        """
        try:
            self._log("Checking pod for any partial checkpoint...")
            salvage_ssh = SSHClient()
            salvage_ssh.connect(self._pod_ip, self._pod_port, self.ssh_key_path)
            try:
                files = salvage_ssh.list_remote_files("/workspace/logs/44k")
            except Exception:
                files = []
            g_files = sorted([f for f in files
                              if f.startswith("G_") and f.endswith(".pth")])
            if not g_files:
                self._log("No checkpoint to download — nothing was trained yet.")
                salvage_ssh.close()
                return

            latest_g = g_files[-1]
            self._log(f"Found partial checkpoint {latest_g} — downloading...")
            model_dir = self.models_dir / speaker_name
            model_dir.mkdir(parents=True, exist_ok=True)
            salvage_ssh.download_file(
                f"/workspace/logs/44k/{latest_g}", str(model_dir / latest_g)
            )
            try:
                salvage_ssh.download_file(
                    "/workspace/configs/44k/config.json",
                    str(model_dir / "config.json"),
                )
            except Exception:
                pass

            try:
                dataset_files = self.dataset_manager.list_files(speaker_name)
                total_duration = sum(
                    f.get("duration", 0) or 0 for f in dataset_files
                )
                total_clips = len(dataset_files)
            except Exception:
                total_duration, total_clips = 0.0, 0

            previous_epochs = 0
            meta_path = model_dir / "metadata.json"
            if meta_path.exists():
                try:
                    import json as _json
                    with open(meta_path) as _f:
                        previous_epochs = _json.load(_f).get("epochs", 0)
                except Exception:
                    pass

            self._save_metadata(model_dir, latest_g, previous_epochs,
                                total_duration, total_clips)
            update_job(job_id, status="completed",
                       model_path=str(model_dir / latest_g))
            self._log(f"Saved partial model: {latest_g}")
            salvage_ssh.close()
        except Exception as e:
            self._log(f"Salvage failed: {e}")

    def _save_metadata(self, model_dir: Path, checkpoint: str,
                       previous_epochs: int, total_duration: float, total_clips: int):
        """Save model metadata.json."""
        import json
        epoch_str = checkpoint.replace("G_", "").replace(".pth", "")
        epoch_num = int(epoch_str) if epoch_str.isdigit() else 0
        # Filenames are post-renamed to reflect TOTAL epochs (the runner
        # adds the resume offset on its way out), so the checkpoint name
        # is already the total — no need to add previous_epochs again.
        total_epochs = epoch_num

        batch_size = 16
        try:
            config_data = json.loads(open(str(model_dir / "config.json")).read())
            batch_size = config_data.get("train", {}).get("batch_size", 16)
        except Exception:
            pass

        metadata = {
            "epochs": total_epochs,
            "batch_size": batch_size,
            "dataset_duration_s": round(total_duration, 1),
            "dataset_clips": total_clips,
            "checkpoint": checkpoint,
            "trained_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        }
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _download_from_r2(self, job_id: str, speaker_name: str, r2: R2Client,
                          r2_prefix: str, previous_epochs: int,
                          total_duration: float, total_clips: int):
        """Download a trained model from R2 to local disk."""
        import json

        model_dir = self.models_dir / speaker_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Find the checkpoint file in R2
        files = r2.list_files(r2_prefix + "/")
        g_files = sorted([f for f in files if "G_" in f and f.endswith(".pth")])
        if not g_files:
            raise RuntimeError(f"No checkpoint found in R2 at {r2_prefix}")

        latest_key = g_files[-1]
        checkpoint_name = latest_key.split("/")[-1]

        self._log(f"Downloading {checkpoint_name} from cloud...")
        r2.download_file(latest_key, str(model_dir / checkpoint_name))

        config_key = f"{r2_prefix}/config.json"
        if r2.file_exists(config_key):
            r2.download_file(config_key, str(model_dir / "config.json"))

        self._save_metadata(model_dir, checkpoint_name,
                           previous_epochs, total_duration, total_clips)
        self._log("Model downloaded from cloud!")

        # Clean up R2 files
        r2.delete_files(files)
        self._log("Cloud storage cleaned up")

        update_job(job_id, status="completed",
                   model_path=str(model_dir / checkpoint_name),
                   config_path=str(model_dir / "config.json"))

    @staticmethod
    def find_pending_cloud_models(on_log: Callable[[str], None] | None = None):
        """Read-only: list models that finished training in R2 but haven't
        been downloaded yet. Used by the UI to prompt before downloading.
        """
        from services.job_store import list_jobs
        log = on_log or (lambda _: None)
        r2 = R2Client()
        if not r2.is_configured():
            return []

        candidates = []
        for job in list_jobs():
            status = job.get("status", "")
            r2_prefix = job.get("r2_prefix")
            speaker = job.get("speaker_name")
            if not r2_prefix or not speaker:
                continue
            if status not in ("training", "uploading_model", "downloading"):
                continue
            if not r2.file_exists(f"{r2_prefix}/_complete.json"):
                continue
            files = r2.list_files(r2_prefix + "/")
            g_files = sorted([f for f in files if "G_" in f and f.endswith(".pth")])
            if not g_files:
                continue
            checkpoint_name = g_files[-1].split("/")[-1]
            candidates.append({
                "job_id": job["job_id"],
                "speaker": speaker,
                "r2_prefix": r2_prefix,
                "checkpoint_name": checkpoint_name,
                "files": files,
                "previous_epochs": job.get("previous_epochs", 0),
                "is_resume": job.get("resume", False),
                "dataset_duration": job.get("dataset_duration", 0),
                "dataset_clips": job.get("dataset_clips", 0),
            })
            log(f"Found completed model for '{speaker}' in cloud storage.")
        return candidates

    @staticmethod
    def download_pending_models(
        models_dir: str,
        candidates: list,
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[str, int, int, int], None] | None = None,
    ):
        """Download the given candidates and ALWAYS delete them from R2 once
        the local checkpoint is verified on disk.

        `on_progress(speaker, idx_1based, total, percent)` fires repeatedly
        during each artist's checkpoint download so the UI can show progress.
        """
        log = on_log or (lambda _: None)
        r2 = R2Client()
        if not r2.is_configured():
            return []

        recovered = []
        total_artists = len(candidates)
        for i, c in enumerate(candidates, start=1):
            speaker = c["speaker"]
            r2_prefix = c["r2_prefix"]
            checkpoint_name = c["checkpoint_name"]
            files = c["files"]
            try:
                model_dir = Path(models_dir) / speaker
                model_dir.mkdir(parents=True, exist_ok=True)

                log(f"Downloading {checkpoint_name} for '{speaker}'...")
                local_ckpt = model_dir / checkpoint_name

                ckpt_key = f"{r2_prefix}/{checkpoint_name}"
                total_bytes = r2.head_size(ckpt_key) or 1
                transferred = {"n": 0}
                if on_progress:
                    on_progress(speaker, i, total_artists, 0)

                def _cb(chunk: int, _t=total_bytes, _i=i, _s=speaker):
                    transferred["n"] += chunk
                    pct = min(int(transferred["n"] / _t * 100), 100)
                    if on_progress:
                        on_progress(_s, _i, total_artists, pct)

                r2.download_file(ckpt_key, str(local_ckpt), callback=_cb)
                if on_progress:
                    on_progress(speaker, i, total_artists, 100)

                config_key = f"{r2_prefix}/config.json"
                if r2.file_exists(config_key):
                    r2.download_file(config_key, str(model_dir / "config.json"))

                # Verify the checkpoint is really on disk before deleting from R2.
                if not local_ckpt.exists() or local_ckpt.stat().st_size == 0:
                    raise RuntimeError(
                        f"Downloaded file missing or empty: {local_ckpt}"
                    )

                # Build local metadata.json. Filename already reflects total
                # epochs (runner renamed it on the pod), so don't double-add.
                import json
                epoch_str = checkpoint_name.replace("G_", "").replace(".pth", "")
                epoch_num = int(epoch_str) if epoch_str.isdigit() else 0
                total_epochs = epoch_num
                batch_size = 16
                try:
                    cfg = json.loads(open(str(model_dir / "config.json")).read())
                    batch_size = cfg.get("train", {}).get("batch_size", 16)
                except Exception:
                    pass
                metadata = {
                    "epochs": total_epochs,
                    "batch_size": batch_size,
                    "dataset_duration_s": c.get("dataset_duration", 0),
                    "dataset_clips": c.get("dataset_clips", 0),
                    "checkpoint": checkpoint_name,
                    "trained_at": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                }
                try:
                    with open(model_dir / "metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)
                except Exception as e:
                    log(f"Wrote model but metadata save failed (non-fatal): {e}")

                # Mark job complete locally so we never re-download.
                update_job(c["job_id"], status="completed",
                           model_path=str(local_ckpt))

                # Local copy is verified — always clear cloud storage now.
                try:
                    r2.delete_files(files)
                    log(f"Cleared '{speaker}' from cloud storage.")
                except Exception as e:
                    log(f"Cloud cleanup failed for '{speaker}' (model is safe locally): {e}")

                log(f"Model '{speaker}' recovered successfully!")
                recovered.append(speaker)
            except Exception as e:
                log(f"Failed to recover '{speaker}': {e}")
        return recovered

    @staticmethod
    def check_pending_downloads(models_dir: str, on_log: Callable[[str], None] | None = None):
        """Back-compat: find + download together. Newer callers should use
        find_pending_cloud_models() then download_pending_models() so the
        user can be prompted between the two.
        """
        candidates = TrainingOrchestrator.find_pending_cloud_models(on_log)
        if not candidates:
            return []
        return TrainingOrchestrator.download_pending_models(
            models_dir, candidates, on_log
        )

    def _wait_for_pod(self, pod_id: str, timeout: int = 300) -> tuple[str, int]:
        """Wait for pod to become RUNNING and return (ip, ssh_port).

        Bails out early via _check_stopped() so a Stop click during pod
        boot doesn't sit waiting up to 5 minutes for SSH to come up.
        """
        start = time.time()
        while time.time() - start < timeout:
            self._check_stopped()
            pod = self.runpod.get_pod(pod_id)
            if not pod:
                self._log("  Pod not found, waiting...")
                self._sleep_interruptible(10)
                continue

            status = pod.get("desiredStatus", "UNKNOWN")
            runtime = pod.get("runtime", {})

            if status == "RUNNING" and runtime:
                ip, port = self.runpod.get_pod_ssh_info(pod_id)
                if ip and port:
                    self._log(f"  SSH available at {ip}:{port}")
                    self._sleep_interruptible(5)
                    return ip, port
                else:
                    self._log("  Pod running, waiting for SSH endpoint...")
            else:
                self._log(f"  Pod status: {status}, runtime: {'ready' if runtime else 'initializing'}...")

            self._sleep_interruptible(10)
        raise TimeoutError(f"Pod {pod_id} did not start within {timeout}s")

    def _sleep_interruptible(self, seconds: float):
        """Sleep but wake up immediately if Stop is requested."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._stop_requested:
                return
            time.sleep(min(0.5, deadline - time.time()))
