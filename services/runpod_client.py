"""RunPod API wrapper for GPU pod lifecycle management."""

import runpod

# GPUs compatible with CUDA 11.8 / PyTorch 2.1 (sm_50 to sm_90 only)
# NO Blackwell (sm_120): RTX 5090, 5080, PRO 4500/6000, B200, B300
GPU_PREFERENCE = [
    "NVIDIA A40",                 # ~$0.44/hr — best value
    "NVIDIA GeForce RTX 4090",    # ~$0.69/hr
    "NVIDIA GeForce RTX 3090",    # ~$0.46/hr
    "NVIDIA RTX A6000",           # ~$0.49/hr
    "NVIDIA RTX A5000",           # ~$0.27/hr
    "NVIDIA A100 80GB PCIe",      # ~$1.39/hr
    "NVIDIA A100-SXM4-80GB",      # ~$1.49/hr
]

# Settings → "Cloud GPU" radio tiers map to ordered preference chains.
# Each chain starts with the tier's primary GPU and degrades through
# the next-best options on the same Pareto frontier so an unavailable
# pick still gets something usable.
TIER_CHAINS = {
    "cheapest": [
        "NVIDIA A40",
        "NVIDIA RTX A6000",
        "NVIDIA RTX 6000 Ada Generation",
    ],
    "balanced": [
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA RTX A6000",
        "NVIDIA A40",
    ],
    "fast": [
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA RTX 6000 Ada Generation",
    ],
    "fastest": [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA RTX 6000 Ada Generation",
    ],
}


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

        # Build the chain: tier preference first, then fall through to the
        # rest of GPU_PREFERENCE so an unavailable pick degrades silently.
        chain = list(TIER_CHAINS.get(preferred_tier, TIER_CHAINS["cheapest"]))
        for gpu in GPU_PREFERENCE:
            if gpu not in chain:
                chain.append(gpu)
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
                if "no longer any instances" in str(e).lower() or "unavailable" in str(e).lower():
                    log(f"  {gpu_type} unavailable, trying next...")
                    continue
                raise

        raise RuntimeError("No GPUs available on RunPod right now. Try again in a few minutes.")

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
