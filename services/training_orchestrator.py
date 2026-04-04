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
        self._stop_requested = False
        self._pod_ip = None
        self._pod_port = None

    def request_stop(self):
        """Request graceful stop — force-saves checkpoint, then kills training."""
        self._stop_requested = True
        if self._pod_ip and self._pod_port:
            try:
                self.on_log("Saving checkpoint before stopping...")
                stop_ssh = SSHClient()
                stop_ssh.connect(self._pod_ip, self._pod_port, self.ssh_key_path)
                # SIGTERM triggers graceful shutdown — Lightning saves checkpoint before exiting
                stop_ssh.exec_command(
                    "pkill -SIGTERM -f 'svc train' 2>/dev/null; "
                    "echo 'Graceful shutdown signal sent — waiting for checkpoint save...'; "
                    "sleep 8; "
                    "echo 'Training stopped'"
                )
                stop_ssh.close()
            except Exception as e:
                self.on_log(f"Stop signal error: {e}")

    def _log(self, msg: str):
        self.on_log(msg)

    def _status(self, status: str):
        self.on_status(status)

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
                    self._log(f"Pod resumed: {pod_id}")
                    update_job(job_id, pod_id=pod_id, status="creating_pod")
                except Exception as e:
                    self._log(f"Could not resume cached pod: {e}")
                    cached_pod_id = None
                    _config.pop("cached_pod_id", None)
                    from services.job_store import save_config as _save_config
                    _save_config(_config)

            if not cached_pod_id:
                self._log("Creating RunPod GPU instance...")
                ssh_pub_key = ""
                pub_key_path = self.ssh_key_path + ".pub"
                if os.path.exists(pub_key_path):
                    with open(pub_key_path) as f:
                        ssh_pub_key = f.read().strip()

                pod = self.runpod.create_training_pod(ssh_pub_key, on_log=self._log)
                pod_id = pod["id"]
                self._log(f"Pod created: {pod_id}")
                update_job(job_id, pod_id=pod_id, status="creating_pod")

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

            # Step 6: Install dependencies (check if already installed)
            self._status("installing")
            self.on_progress(30 if use_local_preprocess else 20)
            update_job(job_id, status="installing")

            # Check if svc is already available (custom Docker image or cached pod)
            self._log("Checking for existing installation...")
            self._svc_available = False
            ssh.exec_command(
                "which svc > /dev/null 2>&1 && python3 -c 'import so_vits_svc_fork' 2>/dev/null && echo SVC_OK || echo SVC_MISSING",
                on_stdout=lambda line: setattr(self, '_svc_available', 'SVC_OK' in line),
            )

            if self._svc_available:
                self._log("so-vits-svc-fork already installed — skipping!")
            else:
                self._log("Installing so-vits-svc-fork (keeping existing PyTorch)...")
                exit_code = ssh.exec_command(
                    "pip install so-vits-svc-fork --no-deps && "
                    "pip install cm-time click fastapi librosa 'lightning<2.5' matplotlib "
                    "pebble praat-parselmouth psutil pysimplegui-4-foss pyworld "
                    "requests rich scipy sounddevice soundfile tensorboard "
                    "tensorboardx torchcrepe tqdm tqdm-joblib 'transformers<4.46' "
                    "'numpy<2' 'huggingface-hub<1' 'rich==13.9.4' 2>&1",
                    on_stdout=self._log,
                    on_stderr=self._log,
                )
                if exit_code != 0:
                    raise RuntimeError(f"Failed to install so-vits-svc-fork (exit code {exit_code})")

            # Verify CUDA
            ssh.exec_command(
                "python3 -c \"import torch; print('PyTorch:', torch.__version__); "
                "print('CUDA available:', torch.cuda.is_available()); "
                "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\"",
                on_stdout=self._log,
                on_stderr=self._log,
            )
            self._log("Installation complete")

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

            # Upload existing checkpoint if resuming
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
                self._log("Resume checkpoint uploaded — training will continue from last epoch")

                # Patch so-vits-svc-fork to start epoch counter from checkpoint epoch
                resume_epoch = resume_file.replace("G_", "").replace(".pth", "")
                if resume_epoch.isdigit():
                    ssh.exec_command(
                        "cat > /tmp/patch_epoch.py << 'PATCHEOF'\n"
                        "import pathlib, re\n"
                        "for p in pathlib.Path('/').glob('**/so_vits_svc_fork/train.py'):\n"
                        "    code = p.read_text()\n"
                        "    if 'Setting current epoch to 0' in code:\n"
                        f"        code = code.replace('Setting current epoch to 0', 'Setting current epoch to {resume_epoch}')\n"
                        f"        code = code.replace('trainer.current_epoch = 0', 'trainer.current_epoch = {resume_epoch}')\n"
                        f"        code = code.replace('self.current_epoch = 0', 'self.current_epoch = {resume_epoch}')\n"
                        "        p.write_text(code)\n"
                        f"        print('Epoch counter will start from {resume_epoch}')\n"
                        "    break\n"
                        "PATCHEOF\n"
                        "python3 /tmp/patch_epoch.py",
                        on_stdout=self._log,
                    )

            # --- OPTIMIZATION 1: Increase batch size ---
            self._log("Optimizing training config for max speed...")
            ssh.exec_command(
                "cat > /tmp/optimize.py << 'OPTEOF'\n"
                "import json, glob, subprocess\n"
                "try:\n"
                "    out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits']).decode().strip()\n"
                "    vram_mb = int(out.split(chr(10))[0])\n"
                "except:\n"
                "    vram_mb = 24000\n"
                "if vram_mb >= 80000:\n"
                "    target_batch = 256\n"
                "elif vram_mb >= 45000:\n"
                "    target_batch = 128\n"
                "elif vram_mb >= 20000:\n"
                "    target_batch = 64\n"
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
                "    with open(cfg_path, 'w') as f:\n"
                "        json.dump(cfg, f, indent=2)\n"
                "    print(f'VRAM: {vram_mb}MB | Batch: {original_batch} -> {target_batch} | LR: {original_lr} -> {new_lr:.6f}')\n"
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

            exit_code = ssh.exec_command(
                "cd /workspace && pip install 'rich==13.9.4' -q 2>/dev/null && "
                "svc train --model-path /workspace/logs/44k",
                on_stdout=self._log,
                on_stderr=self._log,
            )
            if exit_code != 0 and not self._stop_requested:
                raise RuntimeError(f"Training failed (exit code {exit_code})")
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
                # Upload model to R2 from the pod
                self._log("Uploading model to cloud storage...")
                update_job(job_id, status="uploading_model",
                           r2_prefix=r2_prefix,
                           speaker_name=speaker_name,
                           previous_epochs=previous_epochs,
                           previous_batch=previous_batch,
                           resume=bool(self.resume_from),
                           dataset_duration=round(total_duration, 1),
                           dataset_clips=total_clips)

                # Install boto3 and run upload script on pod
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
            self._log(f"ERROR: {e}")
            update_job(job_id, status="failed", error=str(e))
            raise
        finally:
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
                if job_status == "completed" or r2_uploaded:
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

    def _save_metadata(self, model_dir: Path, checkpoint: str,
                       previous_epochs: int, total_duration: float, total_clips: int):
        """Save model metadata.json."""
        import json
        epoch_str = checkpoint.replace("G_", "").replace(".pth", "")
        epoch_num = int(epoch_str) if epoch_str.isdigit() else 0

        if self.resume_from:
            total_epochs = previous_epochs + epoch_num
        else:
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
    def check_pending_downloads(models_dir: str, on_log: Callable[[str], None] | None = None):
        """Check for models uploaded to R2 while app was closed. Call on startup."""
        from services.job_store import list_jobs
        log = on_log or (lambda _: None)

        r2 = R2Client()
        if not r2.is_configured():
            return []

        jobs = list_jobs()
        recovered = []

        for job in jobs:
            status = job.get("status", "")
            r2_prefix = job.get("r2_prefix")
            speaker = job.get("speaker_name")

            if not r2_prefix or not speaker:
                continue

            # Check for jobs that were training/uploading when app closed
            if status in ("training", "uploading_model", "downloading"):
                # Check if model made it to R2
                complete_key = f"{r2_prefix}/_complete.json"
                if r2.file_exists(complete_key):
                    log(f"Found completed model for '{speaker}' in cloud storage!")
                    try:
                        model_dir = Path(models_dir) / speaker
                        model_dir.mkdir(parents=True, exist_ok=True)

                        files = r2.list_files(r2_prefix + "/")
                        g_files = sorted([f for f in files if "G_" in f and f.endswith(".pth")])
                        if not g_files:
                            continue

                        latest_key = g_files[-1]
                        checkpoint_name = latest_key.split("/")[-1]

                        log(f"Downloading {checkpoint_name}...")
                        r2.download_file(latest_key, str(model_dir / checkpoint_name))

                        config_key = f"{r2_prefix}/config.json"
                        if r2.file_exists(config_key):
                            r2.download_file(config_key, str(model_dir / "config.json"))

                        # Build metadata
                        import json
                        prev_epochs = job.get("previous_epochs", 0)
                        is_resume = job.get("resume", False)
                        epoch_str = checkpoint_name.replace("G_", "").replace(".pth", "")
                        epoch_num = int(epoch_str) if epoch_str.isdigit() else 0
                        total_epochs = (prev_epochs + epoch_num) if is_resume else epoch_num

                        batch_size = 16
                        try:
                            cfg = json.loads(open(str(model_dir / "config.json")).read())
                            batch_size = cfg.get("train", {}).get("batch_size", 16)
                        except Exception:
                            pass

                        metadata = {
                            "epochs": total_epochs,
                            "batch_size": batch_size,
                            "dataset_duration_s": job.get("dataset_duration", 0),
                            "dataset_clips": job.get("dataset_clips", 0),
                            "checkpoint": checkpoint_name,
                            "trained_at": __import__("datetime").datetime.now(
                                __import__("datetime").timezone.utc
                            ).isoformat(),
                        }
                        with open(model_dir / "metadata.json", "w") as f:
                            json.dump(metadata, f, indent=2)

                        # Clean up R2
                        r2.delete_files(files)

                        # Mark job complete
                        update_job(job["job_id"], status="completed",
                                   model_path=str(model_dir / checkpoint_name))
                        log(f"Model '{speaker}' recovered successfully!")
                        recovered.append(speaker)
                    except Exception as e:
                        log(f"Failed to recover '{speaker}': {e}")

        return recovered

    def _wait_for_pod(self, pod_id: str, timeout: int = 300) -> tuple[str, int]:
        """Wait for pod to become RUNNING and return (ip, ssh_port)."""
        start = time.time()
        while time.time() - start < timeout:
            pod = self.runpod.get_pod(pod_id)
            if not pod:
                self._log("  Pod not found, waiting...")
                time.sleep(10)
                continue

            status = pod.get("desiredStatus", "UNKNOWN")
            runtime = pod.get("runtime", {})

            if status == "RUNNING" and runtime:
                ip, port = self.runpod.get_pod_ssh_info(pod_id)
                if ip and port:
                    self._log(f"  SSH available at {ip}:{port}")
                    time.sleep(5)
                    return ip, port
                else:
                    self._log("  Pod running, waiting for SSH endpoint...")
            else:
                self._log(f"  Pod status: {status}, runtime: {'ready' if runtime else 'initializing'}...")

            time.sleep(10)
        raise TimeoutError(f"Pod {pod_id} did not start within {timeout}s")
