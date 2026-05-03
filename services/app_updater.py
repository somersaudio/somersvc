"""Self-updater for the packaged .app — checks GitHub Releases for new versions."""

import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests


RELEASES_API = "https://api.github.com/repos/somersaudio/somersvc-releases/releases/latest"
APP_NAME = "SomerSVC.app"


def get_current_version() -> str:
    """Return the version baked into Info.plist when running from a bundle, else dev marker."""
    if not getattr(sys, "frozen", False):
        return "dev"
    try:
        info_plist = Path(sys.executable).parent.parent / "Info.plist"
        if not info_plist.exists():
            return "dev"
        # Cheap parse — avoid plistlib dependency at startup
        text = info_plist.read_text()
        match = re.search(
            r"<key>CFBundleShortVersionString</key>\s*<string>([^<]+)</string>",
            text,
        )
        return match.group(1) if match else "dev"
    except Exception:
        return "dev"


def _parse_version(v: str) -> tuple[int, int, int]:
    v = v.lstrip("v")
    parts = (v.split(".") + ["0", "0", "0"])[:3]
    out = []
    for p in parts:
        digits = re.match(r"\d+", p)
        out.append(int(digits.group(0)) if digits else 0)
    return tuple(out)


def check_for_update() -> dict | None:
    """Return release info if a newer version is available, else None."""
    if not getattr(sys, "frozen", False):
        # Don't auto-update when running from source (developer mode)
        return None

    try:
        resp = requests.get(RELEASES_API, timeout=10)
        if resp.status_code != 200:
            return None
        rel = resp.json()
    except Exception:
        return None

    latest_tag = rel.get("tag_name", "")
    if not latest_tag:
        return None

    if _parse_version(latest_tag) <= _parse_version(get_current_version()):
        return None

    # Find a .dmg asset matching the current arch
    arch = platform.machine()  # 'arm64' or 'x86_64'
    asset_url = None
    asset_name = None
    for asset in rel.get("assets", []):
        name = asset.get("name", "")
        if name.endswith(".dmg") and arch in name:
            asset_url = asset.get("browser_download_url")
            asset_name = name
            break

    if not asset_url:
        return None

    return {
        "tag": latest_tag,
        "url": asset_url,
        "name": asset_name,
        "notes": rel.get("body", ""),
    }


def download_and_install(asset_url: str, on_progress=None) -> bool:
    """Download the .dmg, mount it, replace SomerSVC.app in /Applications.

    Returns True on success. Caller should QApplication.quit() afterward.
    """
    log = on_progress or (lambda _msg: None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dmg_path = Path(tmp) / "SomerSVC-update.dmg"

            # Download
            log("Downloading update...")
            with requests.get(asset_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0
                with open(dmg_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total and on_progress:
                            pct = int(downloaded / total * 100)
                            log(f"Downloading update... {pct}%")

            # Mount
            log("Mounting update image...")
            mount_proc = subprocess.run(
                ["hdiutil", "attach", str(dmg_path), "-nobrowse", "-quiet"],
                capture_output=True, text=True, check=True,
            )
            mount_point = None
            for line in mount_proc.stdout.splitlines():
                if "/Volumes/" in line:
                    mount_point = line.split("\t")[-1].strip()
                    break
            if not mount_point:
                raise RuntimeError("Could not find mount point for update DMG")

            try:
                src_app = Path(mount_point) / APP_NAME
                if not src_app.exists():
                    raise RuntimeError(f"Update DMG missing {APP_NAME}")

                dest_app = Path("/Applications") / APP_NAME

                # Remove old, copy new
                log("Replacing SomerSVC.app in /Applications...")
                if dest_app.exists():
                    subprocess.run(["rm", "-rf", str(dest_app)], check=True)
                subprocess.run(
                    ["cp", "-R", str(src_app), str(dest_app)],
                    check=True,
                )
                # Strip quarantine so it launches without Gatekeeper prompt
                subprocess.run(
                    ["xattr", "-d", "-r", "com.apple.quarantine", str(dest_app)],
                    capture_output=True,
                )
                log("Update installed!")
                return True
            finally:
                subprocess.run(
                    ["hdiutil", "detach", mount_point, "-quiet"],
                    capture_output=True,
                )
    except Exception as e:
        log(f"Update failed: {e}")
        return False
