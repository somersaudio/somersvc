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
    """Pull latest code from GitHub on launch, reinstall deps if needed.

    Only runs in dev mode (running from source). The frozen .app uses
    services.app_updater to check GitHub Releases and self-replace.
    """
    if getattr(sys, "frozen", False):
        return  # Skip git pull when running from a packaged .app
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

    # Frozen app: check GitHub Releases for updates after the window is up
    if getattr(sys, "frozen", False):
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: _check_app_update(window))

    sys.exit(app.exec())


def _check_app_update(parent_window):
    """Check GitHub Releases for a newer .dmg and prompt the user."""
    try:
        from services.app_updater import check_for_update
        from PyQt6.QtWidgets import QMessageBox
        update = check_for_update()
        if not update:
            return
        msg = QMessageBox(parent_window)
        msg.setWindowTitle("Update Available")
        msg.setText(f"A new version of SomerSVC is available: {update['tag']}")
        notes = (update.get('notes') or '').strip()
        if notes:
            msg.setInformativeText(notes[:400])
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)
        msg.button(QMessageBox.StandardButton.Yes).setText("Install & Restart")
        msg.button(QMessageBox.StandardButton.No).setText("Later")
        if msg.exec() != QMessageBox.StandardButton.Yes:
            return
        _run_update_with_progress(parent_window, update["url"])
    except Exception:
        pass


def _run_update_with_progress(parent_window, asset_url: str):
    """Download + install the new .dmg in a background thread with a progress dialog."""
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QLabel, QProgressBar, QVBoxLayout, QMessageBox,
    )
    from PyQt6.QtCore import QProcess
    from services.app_updater import download_and_install
    import re

    class _Worker(QThread):
        progress = pyqtSignal(int, str)   # percent (0-100, -1=unknown), status text
        done = pyqtSignal(bool)

        def __init__(self, url):
            super().__init__()
            self.url = url

        def run(self):
            def on_msg(text: str):
                m = re.search(r"(\d+)%", text)
                if m:
                    self.progress.emit(int(m.group(1)), text)
                else:
                    self.progress.emit(-1, text)
            ok = download_and_install(self.url, on_progress=on_msg)
            self.done.emit(ok)

    dlg = QDialog(parent_window)
    dlg.setWindowTitle("Installing Update")
    dlg.setModal(True)
    dlg.setFixedSize(420, 130)
    # Block the close button so the user can't half-quit during install
    dlg.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(10)

    lbl_status = QLabel("Preparing update...")
    lbl_status.setStyleSheet("color: #ddd; font-size: 13px;")
    layout.addWidget(lbl_status)

    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    layout.addWidget(bar)

    lbl_hint = QLabel("This usually takes 1-2 minutes. The app will restart automatically.")
    lbl_hint.setStyleSheet("color: rgba(255,255,255,80); font-size: 11px;")
    lbl_hint.setWordWrap(True)
    layout.addWidget(lbl_hint)

    worker = _Worker(asset_url)

    def _on_progress(pct: int, text: str):
        if pct >= 0:
            bar.setRange(0, 100)
            bar.setValue(pct)
        else:
            # Unknown progress — switch the bar to an indeterminate animation
            bar.setRange(0, 0)
        lbl_status.setText(text)
        QApplication.processEvents()

    def _on_done(ok: bool):
        if ok:
            bar.setRange(0, 100)
            bar.setValue(100)
            lbl_status.setText("Update installed! Restarting...")
            QApplication.processEvents()
            QProcess.startDetached("/usr/bin/open", ["-a", "/Applications/SomerSVC.app"])
            QApplication.quit()
        else:
            dlg.reject()
            QMessageBox.warning(
                parent_window, "Update Failed",
                "The update could not be installed. You can try again later from the app.",
            )

    worker.progress.connect(_on_progress)
    worker.done.connect(_on_done)
    worker.start()
    dlg.exec()


if __name__ == "__main__":
    main()
