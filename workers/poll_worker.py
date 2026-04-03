"""QThread worker for polling RunPod job status on app startup."""

import time

from PyQt6.QtCore import QThread, pyqtSignal

from services.job_store import get_active_jobs, update_job
from services.runpod_client import RunPodClient


class PollWorker(QThread):
    job_updated = pyqtSignal(str, str)  # job_id, new_status
    job_completed = pyqtSignal(str)  # job_id
    log_line = pyqtSignal(str)

    def __init__(self, api_key: str, poll_interval: int = 30):
        super().__init__()
        self.api_key = api_key
        self.poll_interval = poll_interval
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        if not self.api_key:
            return

        client = RunPodClient(self.api_key)

        while self._running:
            active_jobs = get_active_jobs()
            if not active_jobs:
                break

            for job in active_jobs:
                if not self._running:
                    break

                pod_id = job.get("pod_id")
                if not pod_id:
                    continue

                status = client.get_pod_status(pod_id)

                if status == "TERMINATED":
                    # Pod is gone — if job wasn't completed, mark as failed
                    if job["status"] not in ("completed", "downloading"):
                        update_job(job["job_id"], status="failed", error="Pod terminated unexpectedly")
                        self.job_updated.emit(job["job_id"], "failed")
                        self.log_line.emit(f"Job {job['job_id'][:8]}: pod terminated unexpectedly")
                elif status == "RUNNING":
                    self.job_updated.emit(job["job_id"], job["status"])
                    self.log_line.emit(f"Job {job['job_id'][:8]}: still running ({job['status']})")

            # Sleep in small increments so we can stop quickly
            for _ in range(self.poll_interval):
                if not self._running:
                    break
                time.sleep(1)
