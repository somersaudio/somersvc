"""End-to-end training orchestration: upload, train on RunPod, download model."""

import os
import time
from pathlib import Path
from typing import Callable

from services.dataset_manager import DatasetManager
from services.job_store import update_job
from services.local_preprocessor import LocalPreprocessor
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
        """Request graceful stop — kills training process on pod, then downloads model."""
        self._stop_requested = True
        if self._pod_ip and self._pod_port:
            try:
                self.on_log("Stopping training gracefully...")
                stop_ssh = SSHClient()
                stop_ssh.connect(self._pod_ip, self._pod_port, self.ssh_key_path)
                stop_ssh.exec_command("pkill -f 'svc train' 2>/dev/null; echo 'Training stopped'")
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

            # Step 2: Local preprocessing (if svc is installed locally)
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
            # PHASE 5: DOWNLOAD MODEL
            # ==========================================

            self._status("downloading")
            self.on_progress(85)
            self._log("Downloading trained model...")
            update_job(job_id, status="downloading")

            model_dir = self.models_dir / speaker_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Find the latest checkpoint
            files = ssh.list_remote_files("/workspace/logs/44k")
            g_files = sorted([f for f in files if f.startswith("G_") and f.endswith(".pth")])
            if not g_files:
                raise RuntimeError("No model checkpoint found after training")

            latest_g = g_files[-1]
            self._log(f"Downloading checkpoint: {latest_g}")
            ssh.download_file(
                f"/workspace/logs/44k/{latest_g}",
                str(model_dir / latest_g),
            )

            # Download config
            ssh.download_file(
                "/workspace/configs/44k/config.json",
                str(model_dir / "config.json"),
            )
            self._log("Model downloaded successfully")

            # Done
            self._status("completed")
            self.on_progress(100)
            update_job(
                job_id,
                status="completed",
                model_path=str(model_dir / latest_g),
                config_path=str(model_dir / "config.json"),
            )
            self._log("Training job completed!")
            return update_job(job_id, status="completed")

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
                self._log("Pod is still running — terminate it manually on runpod.io when ready.")
                self._log(f"Pod ID: {pod_id}")

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
