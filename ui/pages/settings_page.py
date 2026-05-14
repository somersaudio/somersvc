"""Settings page for RunPod API key and SSH key configuration."""

import os
import webbrowser

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from services.job_store import load_config, save_config
from services.runpod_client import RunPodClient
from services.ssh_setup import (
    SSH_KEY_PATH,
    SSH_PUB_PATH,
    ensure_ssh_key,
    register_public_key_with_runpod,
)


class _SetupWorker(QThread):
    finished_ok = pyqtSignal(str, str)  # ssh_path, message
    failed = pyqtSignal(str)

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key

    def run(self):
        try:
            ssh_path, pub_key = ensure_ssh_key()
        except Exception as e:
            self.failed.emit(f"Could not generate SSH key: {e}")
            return

        ok, msg = register_public_key_with_runpod(self.api_key, pub_key)
        if ok:
            self.finished_ok.emit(ssh_path, "Auto-setup complete!")
        else:
            # SSH key exists locally but auto-upload failed
            self.finished_ok.emit(
                ssh_path,
                f"SSH key generated. Could not auto-upload to RunPod ({msg}). "
                f"Use 'Copy Public Key' below and paste it at runpod.io > Settings > SSH Public Keys.",
            )


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_worker = None
        self._init_ui()
        self._load_saved()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        title = QLabel("Settings")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Configure your RunPod account for cloud GPU training")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        # ==========  QUICK SETUP  ==========
        layout.addSpacing(8)
        setup_box = QLabel("First time? Follow these 2 steps:")
        setup_box.setStyleSheet("font-weight: bold; color: #ffd76b; font-size: 13px;")
        layout.addWidget(setup_box)

        step1 = QLabel(
            "1.  Click \"Get API Key\" below. Sign in, then click \"+ Create API Key\" "
            "(give it any name, e.g. \"SomerSVC\"). Copy the generated key and paste it here."
        )
        step1.setStyleSheet("color: #aaa; font-size: 12px;")
        step1.setWordWrap(True)
        layout.addWidget(step1)

        step2 = QLabel("2.  Click \"Auto-Setup SSH Key\". This generates an SSH key and uploads it to RunPod for you.")
        step2.setStyleSheet("color: #aaa; font-size: 12px;")
        step2.setWordWrap(True)
        layout.addWidget(step2)

        layout.addSpacing(8)

        # RunPod API Key
        lbl_api = QLabel("RunPod API Key")
        lbl_api.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_api)

        api_row = QHBoxLayout()
        self.txt_api_key = QLineEdit()
        self.txt_api_key.setPlaceholderText("Paste your RunPod API key here...")
        self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        api_row.addWidget(self.txt_api_key, 1)

        self.btn_show_key = QPushButton("Show")
        self.btn_show_key.setFixedWidth(60)
        self.btn_show_key.clicked.connect(self._toggle_key_visibility)
        api_row.addWidget(self.btn_show_key)

        self.btn_get_api = QPushButton("Get API Key")
        self.btn_get_api.setObjectName("primary")
        self.btn_get_api.clicked.connect(self._open_runpod_settings)
        api_row.addWidget(self.btn_get_api)

        self.btn_test = QPushButton("Test")
        self.btn_test.clicked.connect(self._test_connection)
        api_row.addWidget(self.btn_test)
        layout.addLayout(api_row)

        # SSH Key Path
        layout.addSpacing(16)
        lbl_ssh = QLabel("SSH Private Key Path")
        lbl_ssh.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_ssh)

        ssh_row = QHBoxLayout()
        self.txt_ssh_key = QLineEdit()
        self.txt_ssh_key.setPlaceholderText(str(SSH_KEY_PATH))
        ssh_row.addWidget(self.txt_ssh_key, 1)

        self.btn_browse_ssh = QPushButton("Browse...")
        self.btn_browse_ssh.clicked.connect(self._browse_ssh_key)
        ssh_row.addWidget(self.btn_browse_ssh)

        self.btn_auto_setup = QPushButton("Auto-Setup SSH Key")
        self.btn_auto_setup.setObjectName("primary")
        self.btn_auto_setup.clicked.connect(self._auto_setup_ssh)
        ssh_row.addWidget(self.btn_auto_setup)

        self.btn_copy_pub = QPushButton("Copy Public Key")
        self.btn_copy_pub.clicked.connect(self._copy_public_key)
        ssh_row.addWidget(self.btn_copy_pub)
        layout.addLayout(ssh_row)

        # Spotify API (optional)
        layout.addSpacing(16)
        lbl_spotify = QLabel("Spotify API (optional — for artist images)")
        lbl_spotify.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_spotify)

        spotify_row = QHBoxLayout()
        self.txt_spotify_id = QLineEdit()
        self.txt_spotify_id.setPlaceholderText("Client ID")
        self.txt_spotify_id.setEchoMode(QLineEdit.EchoMode.Password)
        spotify_row.addWidget(self.txt_spotify_id, 1)

        self.txt_spotify_secret = QLineEdit()
        self.txt_spotify_secret.setPlaceholderText("Client Secret")
        self.txt_spotify_secret.setEchoMode(QLineEdit.EchoMode.Password)
        spotify_row.addWidget(self.txt_spotify_secret, 1)
        layout.addLayout(spotify_row)

        # ── Cloud GPU tier ──────────────────────────────────────────────
        layout.addSpacing(20)
        gpu_header = QLabel("Cloud GPU")
        gpu_header.setStyleSheet("color: #ddd; font-size: 13px; font-weight: 600;")
        layout.addWidget(gpu_header)
        gpu_hint = QLabel(
            "Pick the GPU tier for training on RunPod. All four are "
            "compatible with the trainer; cheaper tiers take longer "
            "but cost less per run."
        )
        gpu_hint.setStyleSheet("color: rgba(255,255,255,90); font-size: 10px;")
        gpu_hint.setWordWrap(True)
        layout.addWidget(gpu_hint)
        layout.addSpacing(4)

        self._gpu_tier_group = QButtonGroup(self)
        self.rb_gpu_cheapest = QRadioButton(
            "Cheapest      A40            ~45 min  •  ~$0.38"
        )
        self.rb_gpu_balanced = QRadioButton(
            "Balanced      RTX 6000 Ada   ~30 min  •  ~$0.48"
        )
        self.rb_gpu_fast = QRadioButton(
            "Fast          A100 SXM       ~28 min  •  ~$0.87"
        )
        self.rb_gpu_fastest = QRadioButton(
            "Fastest       H100 SXM       ~18 min  •  ~$1.25"
        )
        for i, rb in enumerate((
            self.rb_gpu_cheapest, self.rb_gpu_balanced,
            self.rb_gpu_fast, self.rb_gpu_fastest,
        )):
            rb.setStyleSheet(
                "QRadioButton { color: #ddd; font-size: 12px; "
                "font-family: Menlo, monospace; padding: 2px 0; }"
            )
            self._gpu_tier_group.addButton(rb, i)
            layout.addWidget(rb)
        self._gpu_tier_keys = {
            0: "cheapest", 1: "balanced", 2: "fast", 3: "fastest",
        }

        # Train Locally — novelty option for users with very fast Macs (M5+)
        # or NVIDIA GPUs. Off by default; pod training is the recommended
        # path for everyone else.
        layout.addSpacing(20)
        self.chk_train_local = QCheckBox(
            "Train locally (skip cloud GPU)"
        )
        self.chk_train_local.setToolTip(
            "Run the entire training pipeline on this computer instead of "
            "renting a cloud GPU. Only practical on very fast Macs (M5+) "
            "or machines with an NVIDIA GPU. Most users should leave this off."
        )
        self.chk_train_local.setStyleSheet(
            "QCheckBox { color: #ddd; font-size: 12px; }"
        )
        layout.addWidget(self.chk_train_local)
        local_hint = QLabel(
            "On a typical Mac, training that takes ~30 min on an A40 can "
            "take many hours or days. Use only if you know what you're doing."
        )
        local_hint.setStyleSheet("color: rgba(255,255,255,90); font-size: 10px;")
        local_hint.setWordWrap(True)
        layout.addWidget(local_hint)

        # Save button
        layout.addSpacing(24)
        self.btn_save = QPushButton("Save Settings")
        self.btn_save.setObjectName("primary")
        self.btn_save.setFixedWidth(160)
        self.btn_save.clicked.connect(self._save)
        layout.addWidget(self.btn_save)

        # Status label
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addStretch()

        # GetSongBPM attribution (bottom-right)
        attribution = QLabel('<a href="https://getsongbpm.com" style="color: #555; text-decoration: none;">Song key data powered by GetSongBPM.com</a>')
        attribution.setOpenExternalLinks(True)
        attribution.setAlignment(Qt.AlignmentFlag.AlignRight)
        attribution.setCursor(Qt.CursorShape.PointingHandCursor)
        attribution.setStyleSheet("font-size: 11px; padding-right: 12px; padding-bottom: 8px;")
        layout.addWidget(attribution)

    def _load_saved(self):
        config = load_config()
        api_key = config.get("runpod_api_key", os.environ.get("SOMERSVC_RUNPOD_KEY", ""))
        if api_key:
            self.txt_api_key.setText(api_key)
        ssh_path = config.get("ssh_key_path", "")
        if not ssh_path:
            # Prefer somersvc-managed key, then fall back to default id_rsa
            if SSH_KEY_PATH.exists():
                ssh_path = str(SSH_KEY_PATH)
            else:
                default = os.path.expanduser("~/.ssh/id_rsa")
                if os.path.exists(default):
                    ssh_path = default
        if ssh_path:
            self.txt_ssh_key.setText(ssh_path)
        if config.get("spotify_client_id"):
            self.txt_spotify_id.setText(config["spotify_client_id"])
        if config.get("spotify_client_secret"):
            self.txt_spotify_secret.setText(config["spotify_client_secret"])
        # Train-locally toggle defaults to off; honour saved preference.
        self.chk_train_local.setChecked(bool(config.get("train_locally", False)))
        # GPU tier — default 'cheapest' (A40), matching pre-picker behaviour.
        tier = config.get("preferred_gpu_tier", "cheapest")
        matched = False
        for idx, key in self._gpu_tier_keys.items():
            if key == tier:
                self._gpu_tier_group.button(idx).setChecked(True)
                matched = True
                break
        if not matched:
            self.rb_gpu_cheapest.setChecked(True)

    def _save(self):
        api_key = self.txt_api_key.text().strip()
        ssh_key = self.txt_ssh_key.text().strip()
        train_locally = self.chk_train_local.isChecked()

        # When training locally, we don't strictly need RunPod creds — but
        # keep the validation so the user can still flip it off later.
        if not api_key and not train_locally:
            self.lbl_status.setText("Please enter your RunPod API key")
            self.lbl_status.setStyleSheet("color: #ef4444;")
            return

        if api_key:
            ssh_path = os.path.expanduser(ssh_key) if ssh_key else ""
            if ssh_key and not os.path.exists(ssh_path):
                self.lbl_status.setText(
                    f"SSH key not found: {ssh_path}. Click 'Auto-Setup SSH Key' to generate one."
                )
                self.lbl_status.setStyleSheet("color: #ef4444;")
                return

        checked_id = self._gpu_tier_group.checkedId()
        preferred_gpu_tier = self._gpu_tier_keys.get(checked_id, "cheapest")

        config_data = {
            "runpod_api_key": api_key,
            "ssh_key_path": ssh_key,
            "train_locally": train_locally,
            "preferred_gpu_tier": preferred_gpu_tier,
        }

        spotify_id = self.txt_spotify_id.text().strip()
        spotify_secret = self.txt_spotify_secret.text().strip()
        if spotify_id:
            config_data["spotify_client_id"] = spotify_id
        if spotify_secret:
            config_data["spotify_client_secret"] = spotify_secret

        save_config(config_data)
        self.lbl_status.setText("Settings saved!")
        self.lbl_status.setStyleSheet("color: #22c55e;")

    def _test_connection(self):
        api_key = self.txt_api_key.text().strip()
        if not api_key:
            self.lbl_status.setText("Enter an API key first")
            self.lbl_status.setStyleSheet("color: #ef4444;")
            return

        self.lbl_status.setText("Testing connection...")
        self.lbl_status.setStyleSheet("color: #f59e0b;")
        self.btn_test.setEnabled(False)

        client = RunPodClient(api_key)
        if client.test_connection():
            self.lbl_status.setText("Connection successful!")
            self.lbl_status.setStyleSheet("color: #22c55e;")
        else:
            self.lbl_status.setText("Connection failed. Check your API key.")
            self.lbl_status.setStyleSheet("color: #ef4444;")

        self.btn_test.setEnabled(True)

    def _toggle_key_visibility(self):
        if self.txt_api_key.echoMode() == QLineEdit.EchoMode.Password:
            self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self.btn_show_key.setText("Hide")
        else:
            self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.btn_show_key.setText("Show")

    def _browse_ssh_key(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SSH Private Key",
            os.path.expanduser("~/.ssh"),
            "All Files (*)",
        )
        if path:
            self.txt_ssh_key.setText(path)

    def _open_runpod_settings(self):
        webbrowser.open("https://www.runpod.io/console/user/settings")

    def _auto_setup_ssh(self):
        """Generate an SSH key and try to upload it to RunPod automatically."""
        api_key = self.txt_api_key.text().strip()
        if not api_key:
            self.lbl_status.setText("Paste your RunPod API key first, then click Auto-Setup.")
            self.lbl_status.setStyleSheet("color: #ef4444;")
            return

        self.lbl_status.setText("Generating SSH key and uploading to RunPod...")
        self.lbl_status.setStyleSheet("color: #f59e0b;")
        self.btn_auto_setup.setEnabled(False)

        self._setup_worker = _SetupWorker(api_key)
        self._setup_worker.finished_ok.connect(self._on_setup_done)
        self._setup_worker.failed.connect(self._on_setup_failed)
        self._setup_worker.start()

    def _on_setup_done(self, ssh_path: str, message: str):
        self.txt_ssh_key.setText(ssh_path)
        self.lbl_status.setText(message)
        # Green if fully auto, amber if user needs to paste pubkey manually
        if "Auto-setup complete" in message:
            self.lbl_status.setStyleSheet("color: #22c55e;")
        else:
            self.lbl_status.setStyleSheet("color: #f59e0b;")
        self.btn_auto_setup.setEnabled(True)
        # Save automatically so they don't have to click Save Settings
        self._save()

    def _on_setup_failed(self, error: str):
        self.lbl_status.setText(error)
        self.lbl_status.setStyleSheet("color: #ef4444;")
        self.btn_auto_setup.setEnabled(True)

    def _copy_public_key(self):
        """Copy the public key text to clipboard so user can paste at runpod.io."""
        try:
            ensure_ssh_key()
            pub = SSH_PUB_PATH.read_text().strip()
            QGuiApplication.clipboard().setText(pub)
            self.lbl_status.setText(
                "Public key copied! Paste it at runpod.io > Settings > SSH Public Keys."
            )
            self.lbl_status.setStyleSheet("color: #22c55e;")
        except Exception as e:
            self.lbl_status.setText(f"Could not copy public key: {e}")
            self.lbl_status.setStyleSheet("color: #ef4444;")

    def get_api_key(self) -> str:
        return self.txt_api_key.text().strip() or os.environ.get("SOMERSVC_RUNPOD_KEY", "")

    def get_ssh_key_path(self) -> str:
        return os.path.expanduser(self.txt_ssh_key.text().strip())
