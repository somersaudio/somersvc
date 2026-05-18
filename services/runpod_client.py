"""RunPod API wrapper for GPU pod lifecycle management."""

import runpod

# The four GPUs the Settings "Cloud GPU" picker exposes — and the ONLY
# ones pod creation ever uses. All are CUDA 11.8 / PyTorch 2.1 capable
# (sm_50–sm_90; no Blackwell: RTX 5090/5080, PRO 4500/6000, B200/B300).
#
# Each tier tries its own GPU first, then degrades through the other
# three — nearest in speed first, cheaper on a tie — so an unavailable
# pick still lands on a GPU the user could have picked from the list,
# never an off-list one.
TIER_CHAINS = {
    "cheapest": [   # A40
        "NVIDIA A40",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA H100 80GB HBM3",
    ],
    "balanced": [   # RTX 6000 Ada
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA A40",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA H100 80GB HBM3",
    ],
    "fast": [       # A100 SXM
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA A40",
    ],
    "fastest": [    # H100 SXM
        "NVIDIA H100 80GB HBM3",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA A40",
    ],
}

# Errors from runpod.create_pod that mean "this GPU type can't host the
# pod right now" — retry with the next GPU in the chain instead of
# aborting the whole run. Anything else (auth, quota) still aborts.
_GPU_RETRIABLE = (
    "no longer any instances",
    "no instances available",
    "unavailable",
    "does not have the resources",
    "try a different machine",
)


class RunPodClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        runpod.api_key = api_key

    def test_connection(self) -> bool:
        try:
            gpus = runpod.get_gpus()
            return isinstance(gpus, list)
        except Exception:
            return False

    def get_available_gpus(self) -> list[dict]:
        try:
            return runpod.get_gpus()
        except Exception:
            return []

    def create_training_pod(
        self,
        ssh_public_key: str = "",
        on_log=None,
        preferred_tier: str = "cheapest",
    ) -> dict:
        log = on_log or (lambda _: None)
        env_vars = {}
        if ssh_public_key:
            env_vars["PUBLIC_KEY"] = ssh_public_key

        # Only the user's tier GPUs are ever tried — the tier's primary
        # first, then the other three from the picker as fallback. The
        # chain never reaches off-list GPUs.
        chain = list(TIER_CHAINS.get(preferred_tier, TIER_CHAINS["cheapest"]))
        log(f"GPU tier: {preferred_tier} → trying {chain[0]} first")

        for gpu_type in chain:
            try:
                log(f"Trying {gpu_type}...")
                pod = runpod.create_pod(
                    name="somersvc-training",
                    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                    gpu_type_id=gpu_type,
                    gpu_count=1,
                    volume_in_gb=50,
                    container_disk_in_gb=20,
                    support_public_ip=True,
                    start_ssh=True,
                    ports="22/tcp",
                    env=env_vars,
                )
                log(f"Got {gpu_type}!")
                return pod
            except Exception as e:
                # A capacity error means "try a different machine" — move
                # to the next GPU in the chain rather than failing the run.
                if any(s in str(e).lower() for s in _GPU_RETRIABLE):
                    log(f"  {gpu_type} unavailable, trying next...")
                    continue
                raise

        raise RuntimeError(
            "Every GPU for the selected tier is unavailable on RunPod "
            "right now. Try again in a few minutes, or pick a different "
            "tier in Settings."
        )

    def get_pod(self, pod_id: str) -> dict | None:
        try:
            return runpod.get_pod(pod_id)
        except Exception:
            return None

    def get_pod_status(self, pod_id: str) -> str:
        pod = self.get_pod(pod_id)
        if pod is None:
            return "TERMINATED"
        return pod.get("desiredStatus", "UNKNOWN")

    def get_pod_ssh_info(self, pod_id: str) -> tuple[str | None, int | None]:
        pod = self.get_pod(pod_id)
        if not pod:
            return None, None

        # Try runtime.ports first
        runtime = pod.get("runtime", {})
        if runtime:
            ports = runtime.get("ports", [])
            for port_info in ports or []:
                if port_info.get("privatePort") == 22:
                    return port_info.get("ip"), port_info.get("publicPort")

        # Fallback: check machine.podExternalIp and port mappings
        machine = pod.get("machine", {})
        if machine:
            ext_ip = machine.get("podExternalIp")
            if ext_ip and runtime:
                ports = runtime.get("ports", [])
                for port_info in ports or []:
                    if port_info.get("privatePort") == 22:
                        return ext_ip, port_info.get("publicPort")

        return None, None

    def terminate_pod(self, pod_id: str) -> bool:
        try:
            runpod.terminate_pod(pod_id)
            return True
        except Exception:
            return False
