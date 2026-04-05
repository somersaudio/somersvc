"""JSON-based job persistence for tracking RunPod training jobs."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from services.paths import USER_DIR

APP_DIR = USER_DIR
JOBS_FILE = APP_DIR / "jobs.json"
CONFIG_FILE = APP_DIR / "config.json"


def _ensure_app_dir():
    APP_DIR.mkdir(parents=True, exist_ok=True)


def _read_jobs() -> list[dict]:
    _ensure_app_dir()
    if not JOBS_FILE.exists():
        return []
    with open(JOBS_FILE, "r") as f:
        return json.load(f)


def _write_jobs(jobs: list[dict]):
    _ensure_app_dir()
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)


def create_job(speaker_name: str) -> dict:
    job = {
        "job_id": str(uuid.uuid4()),
        "speaker_name": speaker_name,
        "pod_id": None,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pod_ip": None,
        "pod_ssh_port": None,
        "model_path": None,
        "config_path": None,
        "error": None,
    }
    jobs = _read_jobs()
    jobs.append(job)
    _write_jobs(jobs)
    return job


def update_job(job_id: str, **kwargs) -> dict | None:
    jobs = _read_jobs()
    for job in jobs:
        if job["job_id"] == job_id:
            job.update(kwargs)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_jobs(jobs)
            return job
    return None


def get_job(job_id: str) -> dict | None:
    for job in _read_jobs():
        if job["job_id"] == job_id:
            return job
    return None


def list_jobs() -> list[dict]:
    return _read_jobs()


def get_active_jobs() -> list[dict]:
    return [
        j for j in _read_jobs()
        if j["status"] not in ("completed", "failed")
    ]


def save_config(config: dict):
    _ensure_app_dir()
    existing = load_config()
    existing.update(config)
    with open(CONFIG_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def load_config() -> dict:
    _ensure_app_dir()
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)
