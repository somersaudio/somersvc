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

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ui.main_window import MainWindow
from ui.styles import DARK_THEME


def auto_update():
    """Pull latest code from GitHub on launch."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "pull", "--ff-only", "origin", "main"],
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if "Already up to date" not in result.stdout:
            print(f"Updated: {result.stdout.strip()}")
    except Exception:
        pass  # No git, no internet, or not a git repo — skip silently


def main():
    auto_update()

    app = QApplication(sys.argv)
    app.setApplicationName("SomerSVC")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_THEME)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
