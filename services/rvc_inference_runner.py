"""RVC inference runner — runs in the venv_rvc environment via subprocess."""

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVC_VENV = os.path.join(APP_DIR, "venv_rvc")
RVC_PYTHON = os.path.join(RVC_VENV, "bin", "python")

# Script that runs inside the RVC venv
RVC_INFER_SCRIPT = '''
import sys, json, warnings, torch
warnings.filterwarnings("ignore")

# Patch torch.load to allow weights_only=False by default (needed for fairseq HuBERT)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

args = json.loads(sys.argv[1])

from rvc_python.infer import RVCInference

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

# Detect model version by checking embedding weight shape
cpt = torch.load(args["model_path"], map_location="cpu", weights_only=False)
version = "v2"
for k, v in cpt.get("weight", {}).items():
    if "emb_phone" in k:
        version = "v1" if v.shape[1] == 256 else "v2"
        break
print(f"Detected RVC {version}", flush=True)
del cpt

rvc = RVCInference(device=device)
rvc.load_model(args["model_path"], version=version)

if args.get("index_path"):
    rvc.set_params(index_path=args["index_path"])

rvc.set_params(
    f0method=args.get("f0_method", "rmvpe"),
    f0up_key=args.get("transpose", 0),
    protect=args.get("protect", 0.33),
)

print("Running inference...", flush=True)
try:
    rvc.infer_file(args["input_path"], args["output_path"])
except (AttributeError, TypeError):
    # Fallback: some versions return tuple from vc_single
    import soundfile as sf
    from rvc_python.lib.audio import load_audio
    import numpy as np
    audio = load_audio(args["input_path"], 16000)
    result = rvc.vc.vc_single(
        0, args["input_path"],
        args.get("transpose", 0),
        None, args.get("f0_method", "rmvpe"),
        "", "", args.get("protect", 0.33),
        0.75, 3, rvc.vc.tgt_sr, 0, 0.25
    )
    if isinstance(result, tuple):
        wav_data = result[1] if len(result) > 1 else result[0]
        if isinstance(wav_data, tuple):
            wav_data = wav_data[0]
    else:
        wav_data = result
    sf.write(args["output_path"], wav_data, rvc.vc.tgt_sr)
print(f"Output saved: {args['output_path']}", flush=True)
'''

SUPPRESS = [
    "UserWarning", "pkg_resources", "FutureWarning",
    "INFO | fairseq", "tensorboardX",
]


class RVCInferenceRunner:
    @staticmethod
    def is_available() -> bool:
        return os.path.exists(RVC_PYTHON)

    def run(
        self,
        source_wav: str,
        model_path: str,
        output_dir: str,
        transpose: int = 0,
        f0_method: str = "rmvpe",
        protect: float = 0.33,
        index_path: str = "",
        on_log: Callable[[str], None] | None = None,
    ) -> str:
        """Run RVC voice conversion. Returns output file path."""
        log = on_log or (lambda _: None)

        if not self.is_available():
            raise RuntimeError("RVC environment not found. Run setup.sh to install.")

        os.makedirs(output_dir, exist_ok=True)
        source_name = Path(source_wav).stem
        output_path = os.path.join(output_dir, f"{source_name}.out.wav")

        args = {
            "model_path": model_path,
            "input_path": source_wav,
            "output_path": output_path,
            "transpose": transpose,
            "f0_method": f0_method,
            "protect": protect,
            "index_path": index_path,
        }

        log("Starting RVC inference...")

        process = subprocess.Popen(
            [RVC_PYTHON, "-c", RVC_INFER_SCRIPT, json.dumps(args)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )

        for line in iter(process.stdout.readline, ""):
            stripped = line.rstrip("\n")
            if any(s in stripped for s in SUPPRESS):
                continue
            if not stripped.strip():
                continue
            log(stripped)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"RVC inference failed (exit code {process.returncode})")

        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file not created: {output_path}")

        return output_path


def detect_model_type(model_dir: str) -> str:
    """Detect whether a model directory contains an SVC or RVC model.
    Returns 'svc' or 'rvc'."""
    files = os.listdir(model_dir) if os.path.isdir(model_dir) else []

    # SVC models have G_*.pth checkpoints (starting with G_) + config.json
    has_svc = any(f.startswith("G_") and f.endswith(".pth") for f in files)
    has_config = "config.json" in files

    if has_svc and has_config:
        return "svc"

    # RVC models: .pth files that don't contain _G_ or _D_ patterns and aren't G_/D_ prefixed
    # Also check for .index files which are RVC-specific
    has_index = any(f.endswith(".index") for f in files)
    rvc_files = _get_rvc_pth_files(files)

    if rvc_files:
        return "rvc"

    # If we have G_ files but no config.json, still treat as SVC
    if has_svc:
        return "svc"

    # Fallback
    return "svc"


def _get_rvc_pth_files(files: list[str]) -> list[str]:
    """Get .pth files that are RVC models (not SVC G_/D_ checkpoints)."""
    rvc = []
    for f in files:
        if not f.endswith(".pth"):
            continue
        # Skip SVC checkpoints: G_*.pth, D_*.pth, or *_G_*.pth, *_D_*.pth
        if f.startswith(("G_", "D_")):
            continue
        if "_G_" in f or "_D_" in f:
            continue
        rvc.append(f)
    return rvc
