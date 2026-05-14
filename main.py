"""SomerSVC - GUI application for so-vits-svc-fork."""

import sys
import os
import traceback

# PyQt6 6.5+ aborts the process when an unhandled Python exception fires
# inside a slot / event handler (sipBadCatcherResult → pyqt6_err_print →
# qFatal). Installing a custom excepthook BEFORE importing PyQt6 keeps
# the app alive: the traceback prints to stderr but the GUI keeps running.
def _qt_safe_excepthook(exctype, value, tb):
    sys.stderr.write("=== Unhandled exception in Qt slot (app stays alive) ===\n")
    traceback.print_exception(exctype, value, tb)
    sys.stderr.write("=" * 56 + "\n")
sys.excepthook = _qt_safe_excepthook

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

# Deliberately NOT importing PyQt6 / ui.* at module scope. Doing so
# triggers Cocoa init (linker side-effects + QtCore static state) which
# in subprocess mode (--svc-mode / --demucs-mode) causes macOS
# LaunchServices to spawn a second app window. We import these inside
# main() only on the GUI branch so the subprocess stays cold.


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


def _hide_subprocess_from_dock():
    """When the bundled .app re-execs itself as a subprocess (--svc-mode /
    --demucs-mode), macOS' LaunchServices treats it as a fresh launch and
    pops a second Dock icon + can spawn a duplicate GUI window. Calling
    NSApplication.setActivationPolicy_(Prohibited) before anything else
    tells macOS we're a background tool, so no Dock icon and no second
    window. ctypes path so we don't need PyObjC bundled."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        appkit = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/AppKit.framework/AppKit"
        )  # noqa: F841  (must be loaded so NSApplication symbol resolves)
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.dylib")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        # First call: 2 args (cls, sel)
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        NSApplication = objc.objc_getClass(b"NSApplication")
        shared = objc.objc_msgSend(
            NSApplication, objc.sel_registerName(b"sharedApplication")
        )
        # Second call: 3 args (instance, sel, NSInteger policy)
        objc.objc_msgSend.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long
        ]
        # NSApplicationActivationPolicyProhibited == 2
        objc.objc_msgSend(
            shared, objc.sel_registerName(b"setActivationPolicy:"), 2
        )
    except Exception:
        pass


def main():
    # When the realtime feature re-execs us with --svc-mode, hand off to
    # the so-vits-svc-fork CLI instead of starting the GUI. This lets the
    # bundled .app act as its own `svc` binary.
    if len(sys.argv) > 1 and sys.argv[1] == "--svc-mode":
        # Force matplotlib's headless Agg backend BEFORE any svc-fork
        # import. svc-fork's utils.py does `import matplotlib.pylab` at
        # module scope; on macOS the default `macosx` backend loads the
        # _macosx C extension which spins up its own NSApplication with
        # Regular activation policy — that's the "second window / second
        # Dock icon" the user has been seeing on Convert. Agg has no
        # Cocoa code path, so the policy we set below stays in force.
        os.environ.setdefault("MPLBACKEND", "Agg")
        _hide_subprocess_from_dock()
        # NumPy 2.x removed binary-mode np.fromstring; svc-fork's
        # plot_spectrogram_to_numpy still uses it. Monkey-patch so
        # validation doesn't crash. Idempotent.
        try:
            import numpy as _np
            if not getattr(_np, "_somersvc_compat_patched", False):
                _orig_fromstring = _np.fromstring
                def _fromstring_compat(string, dtype=float, count=-1, sep=""):
                    if sep == "":
                        return _np.frombuffer(string, dtype=dtype, count=count)
                    return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)
                _np.fromstring = _fromstring_compat
                _np._somersvc_compat_patched = True
        except Exception:
            pass
        from so_vits_svc_fork.__main__ import cli
        # The Click CLI parses sys.argv[1:], so reshape to look like `svc <subcmd> ...`
        sys.argv = ["svc"] + sys.argv[2:]
        cli()
        return

    # Same trick for demucs (vocal isolation) — the bundled .app has no
    # standalone `python` to run `python -m demucs`, so we re-exec ourselves.
    if len(sys.argv) > 1 and sys.argv[1] == "--demucs-mode":
        # Same matplotlib-on-macOS guard as --svc-mode; demucs and its
        # deps drag matplotlib in transitively too.
        os.environ.setdefault("MPLBACKEND", "Agg")
        _hide_subprocess_from_dock()
        from demucs.separate import main as demucs_main
        demucs_main(sys.argv[2:])
        return

    auto_update()

    # Qt + UI imports are deliberately deferred to here so a subprocess
    # spawned with --svc-mode / --demucs-mode never loads them. Loading
    # PyQt6 at module scope was triggering Cocoa side effects in the
    # subprocess and macOS would pop a second app window.
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont, QFontDatabase
    from PyQt6.QtCore import Qt
    from ui.main_window import MainWindow
    from ui.styles import DARK_THEME

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
        done = pyqtSignal(bool, str)      # success, last status / error message

        def __init__(self, url):
            super().__init__()
            self.url = url
            self._last_msg = ""

        def run(self):
            def on_msg(text: str):
                self._last_msg = text
                m = re.search(r"(\d+)%", text)
                if m:
                    self.progress.emit(int(m.group(1)), text)
                else:
                    self.progress.emit(-1, text)
            ok = download_and_install(self.url, on_progress=on_msg)
            self.done.emit(ok, self._last_msg)

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

    def _on_done(ok: bool, last_msg: str):
        if ok:
            bar.setRange(0, 100)
            bar.setValue(100)
            lbl_status.setText("Update ready! Restarting to install...")
            QApplication.processEvents()
            # Don't relaunch from here — the deferred installer in
            # /tmp/somersvc_update/install.sh will replace the .app and
            # call `open` on the new one once we exit.
            QApplication.quit()
        else:
            dlg.reject()
            detail = last_msg or "Unknown error."
            QMessageBox.warning(
                parent_window, "Update Failed",
                f"The update could not be installed.\n\n{detail}\n\n"
                f"You can try again later from the app, or download "
                f"manually from github.com/somersaudio/somersvc-releases.",
            )

    worker.progress.connect(_on_progress)
    worker.done.connect(_on_done)
    worker.start()
    dlg.exec()


if __name__ == "__main__":
    main()
