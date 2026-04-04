"""Inspect model checkpoints to extract metadata for grading."""

import json
import os
import re
import subprocess
from pathlib import Path

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVC_PYTHON = os.path.join(APP_DIR, "venv_rvc", "bin", "python")


def inspect_model(model_dir: str) -> dict:
    """Inspect a model directory and extract whatever metadata we can.
    Saves to metadata.json and returns the metadata dict."""
    meta_path = os.path.join(model_dir, "metadata.json")

    # If metadata already has full training data, don't overwrite
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
        if existing.get("dataset_clips", 0) > 0:
            return existing

    from services.rvc_inference_runner import detect_model_type, _get_rvc_pth_files
    model_type = detect_model_type(model_dir)
    files = os.listdir(model_dir)

    metadata = {"model_type": model_type}

    if model_type == "rvc":
        rvc_files = _get_rvc_pth_files(files)
        if rvc_files:
            pth_path = os.path.join(model_dir, rvc_files[0])
            rvc_meta = _inspect_rvc(pth_path)
            metadata.update(rvc_meta)
    else:
        g_files = sorted([f for f in files if f.startswith("G_") and f.endswith(".pth")])
        if g_files:
            pth_path = os.path.join(model_dir, g_files[-1])
            svc_meta = _inspect_svc(pth_path, g_files[-1])
            metadata.update(svc_meta)

    # Merge with existing metadata (don't overwrite known good data)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
        for k, v in existing.items():
            if v and k not in metadata:
                metadata[k] = v
            elif v and metadata.get(k) in (0, None, ""):
                metadata[k] = v

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _inspect_rvc(pth_path: str) -> dict:
    """Extract metadata from an RVC checkpoint."""
    # Use venv_rvc Python since main venv may not have compatible torch
    script = f'''
import torch, json, sys
cpt = torch.load("{pth_path}", map_location="cpu", weights_only=False)
meta = {{}}

# Epoch count from info field
info = cpt.get("info", "")
if info:
    import re
    m = re.search(r"(\\d+)\\s*epoch", str(info), re.IGNORECASE)
    if m:
        meta["epochs"] = int(m.group(1))

# Sample rate
sr = cpt.get("sr", "")
if sr:
    sr_str = str(sr)
    if "48" in sr_str:
        meta["sample_rate"] = 48000
    elif "40" in sr_str:
        meta["sample_rate"] = 40000
    elif "32" in sr_str:
        meta["sample_rate"] = 32000

# F0 support
meta["f0"] = bool(cpt.get("f0", 0))

# Version detection
version = "v2"
for k, v in cpt.get("weight", {{}}).items():
    if "emb_phone" in k:
        version = "v1" if v.shape[1] == 256 else "v2"
        break
meta["rvc_version"] = version

# Parameter count
total_params = sum(v.numel() for v in cpt.get("weight", {{}}).values())
meta["total_params"] = total_params

print(json.dumps(meta))
'''
    try:
        if os.path.exists(RVC_PYTHON):
            result = subprocess.run(
                [RVC_PYTHON, "-c", script],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "PYTHONWARNINGS": "ignore"},
            )
        else:
            result = subprocess.run(
                ["python3", "-c", script],
                capture_output=True, text=True, timeout=30,
            )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)
    except Exception:
        pass

    return {}


def _inspect_svc(pth_path: str, filename: str) -> dict:
    """Extract metadata from an SVC checkpoint."""
    meta = {}

    # Epoch from filename
    epoch_str = filename.replace("G_", "").replace(".pth", "")
    if epoch_str.isdigit():
        meta["epochs"] = int(epoch_str)

    return meta


def compute_downloaded_grade(metadata: dict) -> tuple[str, str, str]:
    """Compute a simplified grade for downloaded models.
    Returns (grade, color, tooltip)."""
    model_type = metadata.get("model_type", "svc")
    epochs = metadata.get("epochs", 0)
    sr = metadata.get("sample_rate", 0)
    f0 = metadata.get("f0", False)
    version = metadata.get("rvc_version", "")

    if epochs == 0:
        return ("?", "#888", "Unknown quality — no training info available")

    # Scoring system for downloaded models
    score = 0
    tips = []

    # Epoch score (0-3)
    if epochs >= 1000:
        score += 3
        tips.append(f"Training: Extensive ({epochs} epochs)")
    elif epochs >= 500:
        score += 2
        tips.append(f"Training: Good ({epochs} epochs)")
    elif epochs >= 200:
        score += 1
        tips.append(f"Training: Moderate ({epochs} epochs)")
    else:
        tips.append(f"Training: Light ({epochs} epochs)")

    # Sample rate score (0-1)
    if sr >= 48000:
        score += 1
        tips.append("Sample rate: 48kHz (highest)")
    elif sr >= 40000:
        score += 1
        tips.append("Sample rate: 40kHz (good)")
    elif sr > 0:
        tips.append(f"Sample rate: {sr // 1000}kHz")

    # F0 support (0-1)
    if f0:
        score += 1
        tips.append("Pitch modeling: Yes (good for singing)")
    elif model_type == "rvc":
        tips.append("Pitch modeling: Unknown")

    # Version (0-1)
    if version == "v2":
        score += 1
        tips.append("Architecture: RVC v2 (latest)")
    elif version == "v1":
        tips.append("Architecture: RVC v1")

    grades = {
        6: ("S", "#a855f7"),
        5: ("A+", "#22c55e"),
        4: ("A", "#22c55e"),
        3: ("B+", "#5599ff"),
        2: ("B", "#5599ff"),
        1: ("C", "#f59e0b"),
        0: ("D", "#ef4444"),
    }
    grade, color = grades.get(min(score, 6), ("?", "#888"))

    tip_lines = [f"Quality: {grade} (downloaded model)", ""] + tips
    return (grade, color, "\n".join(tip_lines))
