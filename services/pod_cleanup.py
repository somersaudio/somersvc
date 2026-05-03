"""Detect and clean up orphaned RunPod instances on app launch."""

import os

import runpod

from services.job_store import list_jobs, update_job


POD_NAME_PREFIX = "somersvc-training"


def cleanup_orphaned_pods(api_key: str, on_log=None):
    """Terminate any somersvc-training pods that have no active job tracking them.

    Returns a list of (pod_id, action, reason) tuples for logging.
    """
    log = on_log or (lambda _: None)
    if not api_key:
        return []

    runpod.api_key = api_key
    try:
        pods = runpod.get_pods()
    except Exception as e:
        log(f"Pod cleanup: could not list pods ({e})")
        return []

    if not pods:
        return []

    # Build pod_id -> job map from jobs.json
    jobs = list_jobs()
    jobs_by_pod = {j.get("pod_id"): j for j in jobs if j.get("pod_id")}

    actions = []
    for pod in pods:
        pod_id = pod.get("id", "")
        pod_name = pod.get("name", "")
        pod_status = pod.get("desiredStatus", "")

        # Only consider pods this app created
        if not pod_name.startswith(POD_NAME_PREFIX):
            continue

        # Skip pods that aren't running or stopped (already terminated)
        if pod_status not in ("RUNNING", "EXITED", "STOPPED"):
            continue

        job = jobs_by_pod.get(pod_id)

        if job is None:
            # Pod has no record in jobs.json - orphan from a crashed app session
            log(f"Found untracked pod {pod_id} ({pod_status}) — terminating")
            try:
                runpod.terminate_pod(pod_id)
                actions.append((pod_id, "terminated", "no job record"))
                log(f"Terminated orphaned pod {pod_id}")
            except Exception as e:
                actions.append((pod_id, "failed", f"terminate error: {e}"))
                log(f"Could not terminate {pod_id}: {e}")
            continue

        job_status = job.get("status", "")

        # Pod is running but job is finished - leak
        if job_status in ("completed", "failed") and pod_status == "RUNNING":
            log(f"Found leaking pod {pod_id} (job {job_status}) — terminating")
            try:
                runpod.terminate_pod(pod_id)
                actions.append((pod_id, "terminated", f"job already {job_status}"))
                log(f"Terminated leaking pod {pod_id}")
            except Exception as e:
                actions.append((pod_id, "failed", f"terminate error: {e}"))
                log(f"Could not terminate {pod_id}: {e}")
            continue

    return actions
