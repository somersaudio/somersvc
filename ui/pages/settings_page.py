"""Settings page for RunPod API key and SSH key configuration."""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from services.job_store import load_config, save_config
from services.runpod_client import RunPodClient


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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

        # RunPod API Key
        layout.addSpacing(8)
        lbl_api = QLabel("RunPod API Key")
        lbl_api.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_api)

        api_row = QHBoxLayout()
        self.txt_api_key = QLineEdit()
        self.txt_api_key.setPlaceholderText("Enter your RunPod API key...")
        self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        api_row.addWidget(self.txt_api_key, 1)

        self.btn_show_key = QPushButton("Show")
        self.btn_show_key.setFixedWidth(60)
        self.btn_show_key.clicked.connect(self._toggle_key_visibility)
        api_row.addWidget(self.btn_show_key)

        self.btn_test = QPushButton("Test Connection")
        self.btn_test.clicked.connect(self._test_connection)
        api_row.addWidget(self.btn_test)
        layout.addLayout(api_row)

        hint_api = QLabel("Get your API key from runpod.io > Settings > API Keys")
        hint_api.setObjectName("subtitle")
        layout.addWidget(hint_api)

        # SSH Key Path
        layout.addSpacing(16)
        lbl_ssh = QLabel("SSH Private Key Path")
        lbl_ssh.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_ssh)

        ssh_row = QHBoxLayout()
        self.txt_ssh_key = QLineEdit()
        self.txt_ssh_key.setPlaceholderText("~/.ssh/id_rsa")
        self.txt_ssh_key.setText(os.path.expanduser("~/.ssh/id_rsa"))
        ssh_row.addWidget(self.txt_ssh_key, 1)

        self.btn_browse_ssh = QPushButton("Browse...")
        self.btn_browse_ssh.clicked.connect(self._browse_ssh_key)
        ssh_row.addWidget(self.btn_browse_ssh)
        layout.addLayout(ssh_row)

        hint_ssh = QLabel("Your SSH public key must be added to your RunPod account")
        hint_ssh.setObjectName("subtitle")
        layout.addWidget(hint_ssh)

        # Spotify API (for artist images)
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

        hint_spotify = QLabel("Get credentials at developer.spotify.com — auto-fetches artist photos")
        hint_spotify.setObjectName("subtitle")
        layout.addWidget(hint_spotify)

        # Save button
        layout.addSpacing(24)
        self.btn_save = QPushButton("Save Settings")
        self.btn_save.setObjectName("primary")
        self.btn_save.setFixedWidth(160)
        self.btn_save.clicked.connect(self._save)
        layout.addWidget(self.btn_save)

        # Status label
        self.lbl_status = QLabel("")
        layout.addWidget(self.lbl_status)

        layout.addStretch()

    def _load_saved(self):
        config = load_config()
        api_key = config.get("runpod_api_key", os.environ.get("SOMERSVC_RUNPOD_KEY", ""))
        if api_key:
            self.txt_api_key.setText(api_key)
        if config.get("ssh_key_path"):
            self.txt_ssh_key.setText(config["ssh_key_path"])
        if config.get("spotify_client_id"):
            self.txt_spotify_id.setText(config["spotify_client_id"])
        if config.get("spotify_client_secret"):
            self.txt_spotify_secret.setText(config["spotify_client_secret"])

    def _save(self):
        api_key = self.txt_api_key.text().strip()
        ssh_key = self.txt_ssh_key.text().strip()

        if not api_key:
            self.lbl_status.setText("Please enter your RunPod API key")
            self.lbl_status.setStyleSheet("color: #ef4444;")
            return

        ssh_path = os.path.expanduser(ssh_key)
        if not os.path.exists(ssh_path):
            self.lbl_status.setText(f"SSH key not found: {ssh_path}")
            self.lbl_status.setStyleSheet("color: #ef4444;")
            return

        config_data = {"runpod_api_key": api_key, "ssh_key_path": ssh_key}

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

        # Run synchronously since it's a quick API call
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

    def get_api_key(self) -> str:
        return self.txt_api_key.text().strip() or os.environ.get("SOMERSVC_RUNPOD_KEY", "")

    def get_ssh_key_path(self) -> str:
        return os.path.expanduser(self.txt_ssh_key.text().strip())
