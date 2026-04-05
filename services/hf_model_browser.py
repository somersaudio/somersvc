"""Browse and download RVC voice models from HuggingFace."""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable

import requests

HF_REPO = "QuickWick/Music-AI-Voices"
HF_API_URL = f"https://huggingface.co/api/models/{HF_REPO}/tree/main"
from services.paths import USER_DIR
CACHE_DIR = USER_DIR
CACHE_FILE = CACHE_DIR / "hf_models_cache.json"
CACHE_MAX_AGE = 86400  # 24 hours


def _clean_artist_name(folder_name: str) -> str:
    """Extract clean Spotify-searchable artist name from HF folder name."""
    name = re.sub(r"\s*\(RVC\)", "", folder_name)
    name = re.sub(r"\s*\(\d{4}[-–]\d{4}\)", "", name)
    # Remove parenthetical info containing keywords
    name = re.sub(r"\s*\([^)]*(?:RVC|Epoch|Steps|VA |JP |EN |Version|Upd|Crepe|VOCALOID|Homestar|MHA|JoJo|Boku|Overwatch|Paladins|TF2|Linkin|Panic)[^)]*\)", "", name, flags=re.IGNORECASE)
    # Remove any remaining long parentheticals (20+ chars) but keep short ones like (2000-2003)
    name = re.sub(r"\s*\([^)]{20,}\)", "", name)
    name = re.sub(r"\s*\d+k?\s*(Epoch|Steps|epoch|steps).*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*\d+\.?\d*[kK]$", "", name)
    name = re.sub(
        r"\s*(Unknown|Steps|Model|Demo|V\d|v\d|Upd|hmmmm|swap.*|general.*|Might be.*|-$).*",
        "", name, flags=re.IGNORECASE,
    )
    return name.strip().rstrip(".-_ ")


def fetch_available_models(force_refresh: bool = False) -> list[dict]:
    """Fetch list of available RVC models from HuggingFace.
    Returns list of dicts: {'artist': str, 'folder': str, 'type': 'rvc'|'svc'}
    """
    # Check cache
    if not force_refresh and CACHE_FILE.exists():
        import time
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_MAX_AGE:
            with open(CACHE_FILE) as f:
                return json.load(f)

    # Fetch from HuggingFace API
    resp = requests.get(HF_API_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    folders = [item["path"] for item in data if item["type"] == "directory"]

    # Group by clean artist name, prefer RVC models
    artists = {}
    for folder in folders:
        is_rvc = "(RVC)" in folder
        clean = _clean_artist_name(folder)
        if not clean or len(clean) < 2:
            continue

        key = clean.lower()
        if key not in artists:
            artists[key] = {"artist": clean, "rvc_folders": [], "svc_folders": []}

        if is_rvc:
            artists[key]["rvc_folders"].append(folder)
        else:
            artists[key]["svc_folders"].append(folder)

    # Build final list — one entry per artist, include both types if available
    models = []
    for info in sorted(artists.values(), key=lambda x: x["artist"].lower()):
        entry = {"artist": info["artist"], "folder": "", "type": "", "alternatives": []}

        if info["svc_folders"]:
            best_svc = sorted(info["svc_folders"], key=lambda f: _extract_epochs(f), reverse=True)[0]
            entry["folder"] = best_svc
            entry["type"] = "svc"
            if info["rvc_folders"]:
                best_rvc = sorted(info["rvc_folders"], key=lambda f: _extract_epochs(f), reverse=True)[0]
                entry["alternatives"].append({"folder": best_rvc, "type": "rvc"})
        elif info["rvc_folders"]:
            best_rvc = sorted(info["rvc_folders"], key=lambda f: _extract_epochs(f), reverse=True)[0]
            entry["folder"] = best_rvc
            entry["type"] = "rvc"

        if entry["folder"]:
            models.append(entry)

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(models, f)

    return models


def _extract_epochs(folder_name: str) -> int:
    """Extract epoch/step count from folder name for sorting."""
    m = re.search(r"(\d+)\s*(Epoch|Steps|k)", folder_name, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if "k" in m.group(2).lower():
            val *= 1000
        return val
    return 0


def download_model(
    folder: str,
    dest_dir: str,
    on_log: Callable[[str], None] | None = None,
    display_name: str = "",
) -> str:
    """Download an RVC model from HuggingFace to dest_dir. Returns model dir path."""
    log = on_log or (lambda _: None)

    # List files in the folder
    api_url = f"https://huggingface.co/api/models/{HF_REPO}/tree/main/{requests.utils.quote(folder)}"
    resp = requests.get(api_url, timeout=15)
    resp.raise_for_status()
    files = resp.json()

    os.makedirs(dest_dir, exist_ok=True)

    downloadable = [".pth", ".index", ".zip", ".rar", ".7z"]

    for file_info in files:
        name = file_info["path"].split("/")[-1]
        if not any(name.lower().endswith(ext) for ext in downloadable):
            continue

        dl_url = f"https://huggingface.co/{HF_REPO}/resolve/main/{requests.utils.quote(file_info['path'])}"
        local_path = os.path.join(dest_dir, name)

        size_mb = file_info.get("size", 0) / 1024 / 1024
        display = display_name or name
        log(f"Downloading {display}...")
        r = requests.get(dl_url, stream=True, timeout=300)
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded / total * 100)
                    log(f"Downloading {display}... {pct}%")

        log(f"Downloaded {display}")

        # Extract zip files and pull out .pth / .index
        if name.lower().endswith(".zip"):
            log("Extracting model from zip...")
            import zipfile
            try:
                with zipfile.ZipFile(local_path, "r") as zf:
                    for member in zf.namelist():
                        member_name = member.split("/")[-1]
                        if member_name.endswith((".pth", ".index")) and member_name:
                            zf.extract(member, dest_dir)
                            # Move from nested folder to dest_dir root
                            extracted = os.path.join(dest_dir, member)
                            target = os.path.join(dest_dir, member_name)
                            if extracted != target:
                                import shutil
                                shutil.move(extracted, target)
                            log(f"Extracted {member_name}")
                os.unlink(local_path)
                # Clean up empty subdirectories left by extraction
                for root, dirs, _files in os.walk(dest_dir, topdown=False):
                    for d in dirs:
                        dirpath = os.path.join(root, d)
                        try:
                            os.rmdir(dirpath)
                        except OSError:
                            pass
            except zipfile.BadZipFile:
                log("Warning: zip file is corrupted")

    return dest_dir
