"""QThread worker for resuming/monitoring active jobs on app reopen."""

import os
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from services.job_store import get_active_jobs, update_job, load_config
from services.runpod_client import RunPodClient
from services.ssh_client import SSHClient


class ResumeWorker(QThread):
    log_line = pyqtSignal(str)
    status_changed = pyqtSignal(str, str)  # job_id, new_status
    progress = pyqtSignal(int)
    job_finished = pyqtSignal(str)  # job_id
    job_failed = pyqtSignal(str, str)  # job_id, error
    elapsed_seconds = pyqtSignal(int)  # seconds since job started

    def __init__(self, api_key: str, ssh_key_path: str, models_dir: str):
        super().__init__()
        self.api_key = api_key
        self.ssh_key_path = ssh_key_path
        self.models_dir = Path(models_dir)
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        if not self.api_key:
            return

        active_jobs = get_active_jobs()
        if not active_jobs:
            return

        client = RunPodClient(self.api_key)

        for job in active_jobs:
            if not self._running:
                break

            pod_id = job.get("pod_id")
            job_id = job["job_id"]
            speaker = job["speaker_name"]

            if not pod_id:
                update_job(job_id, status="failed", error="No pod ID recorded")
                self.job_failed.emit(job_id, "No pod ID recorded")
                continue

            # Calculate elapsed time from job creation
            self._emit_elapsed(job)

            # Check pod status
            self.log_line.emit(f"Checking job {job_id[:8]}...")
            pod_status = client.get_pod_status(pod_id)

            if pod_status == "TERMINATED":
                # Pod is gone
                if job["status"] == "completed":
                    self.log_line.emit(f"Job {job_id[:8]}: already completed")
                    continue

                # Check if model was already downloaded
                model_dir = self.models_dir / speaker
                if model_dir.exists() and any(model_dir.glob("G_*.pth")):
                    update_job(job_id, status="completed")
                    self.log_line.emit(f"Job {job_id[:8]}: model already downloaded, marking complete")
                    self.job_finished.emit(job_id)
                else:
                    update_job(job_id, status="failed", error="Pod terminated before model was downloaded")
                    self.log_line.emit(f"Job {job_id[:8]}: pod terminated, no model found")
                    self.job_failed.emit(job_id, "Pod terminated before model was downloaded")
                continue

            if pod_status != "RUNNING":
                self.log_line.emit(f"Job {job_id[:8]}: pod status is {pod_status}, waiting...")
                self.status_changed.emit(job_id, "waiting_for_pod")
                # Wait for it to start
                for _ in range(60):
                    if not self._running:
                        return
                    time.sleep(5)
                    pod_status = client.get_pod_status(pod_id)
                    if pod_status == "RUNNING":
                        break
                    if pod_status == "TERMINATED":
                        update_job(job_id, status="failed", error="Pod terminated")
                        self.job_failed.emit(job_id, "Pod terminated")
                        break
                if pod_status != "RUNNING":
                    continue

            # Pod is running — get SSH info and connect
            ip, port = client.get_pod_ssh_info(pod_id)
            if not ip or not port:
                self.log_line.emit(f"Job {job_id[:8]}: can't get SSH info")
                continue

            self.log_line.emit(f"Reconnecting to pod {pod_id} at {ip}:{port}...")
            self.status_changed.emit(job_id, job["status"])

            ssh = SSHClient()
            try:
                ssh.connect(ip, port, self.ssh_key_path)
                self.log_line.emit("SSH reconnected")

                # Determine what stage the job is in
                job_status = job["status"]

                if job_status in ("creating_pod", "waiting_for_pod", "connecting",
                                   "installing", "uploading", "preprocessing"):
                    # Job was in early stages — hard to resume mid-pipeline
                    # Check if training is already running
                    self.log_line.emit("Checking if training is running on pod...")
                    exit_code = ssh.exec_command(
                        "pgrep -f 'svc train' > /dev/null 2>&1 && echo TRAINING_RUNNING || echo NOT_RUNNING",
                        on_stdout=lambda line: self._check_training_status(line, job_id),
                    )

                if job_status == "training" or getattr(self, '_training_detected', False):
                    # Training is running — tail the log
                    self.log_line.emit("Training in progress, monitoring...")
                    self.status_changed.emit(job_id, "training")
                    self.progress.emit(55)
                    self._monitor_training(ssh, job_id)

                # Check if training completed (checkpoint exists)
                files = ssh.list_remote_files("/workspace/logs/44k")
                g_files = sorted([f for f in files if f.startswith("G_") and f.endswith(".pth")])

                if g_files:
                    # Training finished — download model
                    self.log_line.emit("Training complete! Downloading model...")
                    self.status_changed.emit(job_id, "downloading")
                    self.progress.emit(85)

                    model_dir = self.models_dir / speaker
                    model_dir.mkdir(parents=True, exist_ok=True)

                    latest_g = g_files[-1]
                    self.log_line.emit(f"Downloading checkpoint: {latest_g}")
                    ssh.download_file(
                        f"/workspace/logs/44k/{latest_g}",
                        str(model_dir / latest_g),
                    )

                    # Download config
                    try:
                        ssh.download_file(
                            "/workspace/configs/44k/config.json",
                            str(model_dir / "config.json"),
                        )
                    except Exception:
                        self.log_line.emit("Warning: config.json not found, using default")

                    self.log_line.emit("Model downloaded successfully!")
                    update_job(
                        job_id,
                        status="completed",
                        model_path=str(model_dir / latest_g),
                        config_path=str(model_dir / "config.json"),
                    )
                    self.progress.emit(100)
                    self.job_finished.emit(job_id)

                    # Terminate pod
                    self.log_line.emit("Terminating RunPod instance...")
                    client.terminate_pod(pod_id)
                    self.log_line.emit("Pod terminated (billing stopped)")
                else:
                    self.log_line.emit("No checkpoints found yet — training may still be running")
                    # Keep polling
                    self._poll_until_done(ssh, client, pod_id, job_id, speaker)

            except Exception as e:
                self.log_line.emit(f"Resume error: {e}")
                self.job_failed.emit(job_id, str(e))
            finally:
                ssh.close()

    def _check_training_status(self, line: str, job_id: str):
        if "TRAINING_RUNNING" in line:
            self._training_detected = True
            self.log_line.emit("Training process detected on pod")
        else:
            self._training_detected = False

    def _monitor_training(self, ssh: SSHClient, job_id: str):
        """Tail the training output until the svc train process exits."""
        # Check if training process is still running, and tail its output
        ssh.exec_command(
            "tail -f /workspace/logs/44k/train.log 2>/dev/null &"
            " TAIL_PID=$!;"
            " while pgrep -f 'svc train' > /dev/null 2>&1; do sleep 5; done;"
            " kill $TAIL_PID 2>/dev/null",
            on_stdout=self.log_line.emit,
        )

    def _poll_until_done(self, ssh: SSHClient, client: RunPodClient,
                         pod_id: str, job_id: str, speaker: str):
        """Poll every 30s checking for checkpoints or process completion."""
        for _ in range(120):  # Up to 60 minutes
            if not self._running:
                return
            time.sleep(30)

            # Check if training process is still running
            exit_code = ssh.exec_command(
                "pgrep -f 'svc train' > /dev/null 2>&1",
            )

            files = ssh.list_remote_files("/workspace/logs/44k")
            g_files = sorted([f for f in files if f.startswith("G_") and f.endswith(".pth")])

            if exit_code != 0 and g_files:
                # Training process finished and checkpoints exist
                self.log_line.emit("Training finished! Downloading model...")
                self.status_changed.emit(job_id, "downloading")
                self.progress.emit(85)

                model_dir = self.models_dir / speaker
                model_dir.mkdir(parents=True, exist_ok=True)

                latest_g = g_files[-1]
                self.log_line.emit(f"Downloading: {latest_g}")
                ssh.download_file(
                    f"/workspace/logs/44k/{latest_g}",
                    str(model_dir / latest_g),
                )
                try:
                    ssh.download_file(
                        "/workspace/configs/44k/config.json",
                        str(model_dir / "config.json"),
                    )
                except Exception:
                    pass

                update_job(
                    job_id,
                    status="completed",
                    model_path=str(model_dir / latest_g),
                    config_path=str(model_dir / "config.json"),
                )
                self.progress.emit(100)
                self.job_finished.emit(job_id)

                self.log_line.emit("Terminating RunPod instance...")
                client.terminate_pod(pod_id)
                self.log_line.emit("Pod terminated (billing stopped)")
                return

            if g_files:
                self.log_line.emit(f"Training in progress... latest checkpoint: {g_files[-1]}")
            else:
                self.log_line.emit("Waiting for training to produce checkpoints...")

    def _emit_elapsed(self, job: dict):
        """Calculate and emit elapsed seconds since job creation."""
        from datetime import datetime, timezone
        try:
            created = datetime.fromisoformat(job["created_at"])
            now = datetime.now(timezone.utc)
            elapsed = int((now - created).total_seconds())
            self.elapsed_seconds.emit(elapsed)
        except Exception:
            pass
