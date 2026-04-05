"""SomerSVC - GUI application for so-vits-svc-fork."""

import sys
import os

# Ensure the app directory is in the path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

# Load .env file if it exists
env_path = os.path.join(app_dir, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

from services.paths import ensure_dirs
ensure_dirs()

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtCore import Qt

from ui.main_window import MainWindow
from ui.styles import DARK_THEME


def auto_update():
    """Pull latest code from GitHub on launch, reinstall deps if needed."""
    try:
        import subprocess

        # Stash any local changes so pull doesn't fail
        subprocess.run(
            ["git", "stash", "--include-untracked"],
            cwd=app_dir, capture_output=True, text=True, timeout=10,
        )

        result = subprocess.run(
            ["git", "pull", "--ff-only", "origin", "main"],
            cwd=app_dir, capture_output=True, text=True, timeout=15,
        )

        if "Already up to date" not in result.stdout:
            print(f"Updated: {result.stdout.strip()}")

            # Re-download .env in case keys changed
            env_url = "https://gist.githubusercontent.com/somersaudio/ad9423ac7f83b3035850afcbd0a2fc9f/raw/.env"
            try:
                subprocess.run(
                    ["curl", "-sL", env_url, "-o", os.path.join(app_dir, ".env")],
                    timeout=10,
                )
            except Exception:
                pass

            # Reinstall deps if requirements.txt was updated
            req_file = os.path.join(app_dir, "requirements.txt")
            if os.path.exists(req_file):
                pip = os.path.join(os.path.dirname(sys.executable), "pip")
                try:
                    subprocess.run(
                        [pip, "install", "-q", "-r", req_file],
                        cwd=app_dir, capture_output=True, text=True, timeout=120,
                    )
                except Exception:
                    pass
    except Exception:
        pass  # No git, no internet, or not a git repo — skip silently


def main():
    auto_update()

    app = QApplication(sys.argv)
    app.setApplicationName("SomerSVC")

    # Load bundled Manrope font
    fonts_dir = os.path.join(app_dir, "assets", "fonts")
    loaded = False
    if os.path.isdir(fonts_dir):
        for f in os.listdir(fonts_dir):
            if f.startswith("Manrope") and f.endswith(".ttf"):
                QFontDatabase.addApplicationFont(os.path.join(fonts_dir, f))
                loaded = True
    app.setFont(QFont("Manrope" if loaded else "Helvetica", 13))
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_THEME)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
