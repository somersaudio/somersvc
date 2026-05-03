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
    """Download the .dmg, stage the new .app, schedule a deferred installer.

    The actual /Applications replacement runs in a detached shell script
    AFTER this process exits — that's the only reliable way to replace a
    running .app on macOS. Caller MUST call QApplication.quit() right after
    this returns True so the installer can take over.
    """
    log = on_progress or (lambda _msg: None)
    try:
        # Use a stable temp dir so the installer script can find what we staged.
        # We don't use TemporaryDirectory here because that would auto-delete
        # before the deferred installer runs.
        staging = Path(tempfile.gettempdir()) / "somersvc_update"
        if staging.exists():
            subprocess.run(["rm", "-rf", str(staging)], check=False)
        staging.mkdir(parents=True, exist_ok=True)

        dmg_path = staging / "SomerSVC-update.dmg"

        # --- Download ---
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
                    if total:
                        pct = int(downloaded / total * 100)
                        log(f"Downloading update... {pct}%")

        # --- Mount (without -quiet so we can parse the mount point) ---
        log("Mounting update image...")
        mount_proc = subprocess.run(
            ["hdiutil", "attach", str(dmg_path), "-nobrowse", "-noverify",
             "-noautoopen", "-plist"],
            capture_output=True, text=True, check=True,
        )
        # Parse the plist output for the mount-point
        mount_point = None
        m = re.search(r"<key>mount-point</key>\s*<string>([^<]+)</string>",
                      mount_proc.stdout)
        if m:
            mount_point = m.group(1).strip()
        if not mount_point:
            # Fallback: look for /Volumes/<anything> in the output
            m = re.search(r"(/Volumes/[^<\s]+)", mount_proc.stdout)
            if m:
                mount_point = m.group(1).strip()
        if not mount_point:
            raise RuntimeError(
                f"Could not find mount point. hdiutil output:\n"
                f"{mount_proc.stdout[:400]}"
            )

        try:
            src_app = Path(mount_point) / APP_NAME
            if not src_app.exists():
                raise RuntimeError(f"Update DMG missing {APP_NAME}")

            # --- Stage the new .app at a known temp path ---
            log("Preparing update...")
            staged_app = staging / APP_NAME
            if staged_app.exists():
                subprocess.run(["rm", "-rf", str(staged_app)], check=False)
            subprocess.run(
                ["cp", "-R", str(src_app), str(staged_app)],
                check=True,
            )
            # Strip quarantine on the staged copy so the launched .app
            # doesn't trigger Gatekeeper.
            subprocess.run(
                ["xattr", "-d", "-r", "com.apple.quarantine", str(staged_app)],
                capture_output=True,
            )
        finally:
            # Detach the DMG; ignore failures.
            subprocess.run(
                ["hdiutil", "detach", mount_point, "-force"],
                capture_output=True,
            )

        # --- Write a deferred installer that runs after we quit ---
        log("Scheduling install...")
        installer_path = staging / "install.sh"
        log_path = staging / "install.log"
        # Wait for the running SomerSVC process to exit, then do the swap.
        # Detect the running app by ppid (this script's parent) or pgrep.
        installer_path.write_text(f"""#!/bin/bash
exec >"{log_path}" 2>&1
echo "[$(date)] installer starting"
# Wait until the launching SomerSVC.app process is gone (max 30s)
for i in $(seq 1 60); do
    if ! pgrep -f "SomerSVC.app/Contents/MacOS/SomerSVC" > /dev/null; then
        break
    fi
    sleep 0.5
done
sleep 1  # extra safety so file handles are released
DEST="/Applications/{APP_NAME}"
SRC="{staged_app}"
echo "[$(date)] removing $DEST"
rm -rf "$DEST"
echo "[$(date)] copying $SRC -> $DEST"
if cp -R "$SRC" "$DEST"; then
    echo "[$(date)] copy ok"
    xattr -d -r com.apple.quarantine "$DEST" 2>/dev/null || true
    echo "[$(date)] launching"
    open -a "$DEST"
else
    echo "[$(date)] copy failed (need admin?)"
    osascript -e 'display alert "Update failed" message "SomerSVC could not write to /Applications. Please drag the new SomerSVC.app from {staging} into /Applications manually."'
fi
echo "[$(date)] done"
""")
        installer_path.chmod(0o755)

        # Spawn the installer fully detached (nohup + &). It runs in the
        # background and survives our exit.
        subprocess.Popen(
            ["/bin/bash", "-c", f"nohup {installer_path} >/dev/null 2>&1 &"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        log("Update ready! Restarting...")
        return True
    except Exception as e:
        log(f"Update failed: {e}")
        return False
