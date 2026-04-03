"""Models page — Spotify-style model library."""

import json
import os
import shutil

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from services.job_store import load_config
from services.spotify_client import SpotifyClient
from ui.widgets.voice_card import VoiceCard

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(APP_DIR, "data", "models")
DATASETS_DIR = os.path.join(APP_DIR, "data", "datasets")


class ModelsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_voice = ""
        self._cards: list[VoiceCard] = []
        self._init_ui()
        self._refresh_models()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        title = QLabel("Models")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Your trained voice models")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        # Model cards scroll area
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setSpacing(4)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(self.cards_container)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        layout.addWidget(scroll, 1)

        # Selected model actions (hidden until selected)
        self.actions_widget = QWidget()
        actions_layout = QHBoxLayout(self.actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_rename = QPushButton("Rename")
        self.btn_rename.clicked.connect(self._rename_model)
        actions_layout.addWidget(self.btn_rename)

        self.btn_set_image = QPushButton("Set Image")
        self.btn_set_image.clicked.connect(self._set_image)
        actions_layout.addWidget(self.btn_set_image)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(self._delete_model)
        actions_layout.addWidget(self.btn_delete)

        actions_layout.addStretch()

        self.actions_widget.setVisible(False)
        layout.addWidget(self.actions_widget)

    def _get_spotify(self) -> SpotifyClient:
        config = load_config()
        cid = config.get("spotify_client_id", "")
        secret = config.get("spotify_client_secret", "")
        return SpotifyClient(cid, secret)

    def _ensure_artist_image(self, name: str, model_dir: str):
        """Auto-fetch artist image from Spotify if not already set."""
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            if os.path.exists(os.path.join(model_dir, f"image{ext}")):
                return  # Already has image

        spotify = self._get_spotify()
        if not spotify:
            return

        # Clean up voice name for search (remove underscores, numbers)
        search_name = name.replace("_", " ").replace("-", " ")
        # Remove trailing numbers like "Julia Wolf 1" -> "Julia Wolf"
        parts = search_name.split()
        if parts and parts[-1].isdigit():
            search_name = " ".join(parts[:-1])

        save_path = os.path.join(model_dir, "image.jpg")
        spotify.download_artist_image(search_name, save_path)

    def _refresh_models(self):
        """Rebuild model cards."""
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        if not os.path.isdir(MODELS_DIR):
            self.cards_layout.addStretch()
            return

        for name in sorted(os.listdir(MODELS_DIR)):
            model_dir = os.path.join(MODELS_DIR, name)
            if not os.path.isdir(model_dir):
                continue

            has_model = any(
                f.startswith("G_") and f.endswith(".pth")
                for f in os.listdir(model_dir)
            )
            if not has_model:
                continue

            # Auto-fetch Spotify image if missing
            self._ensure_artist_image(name, model_dir)

            metadata = self._load_metadata(name)
            card = VoiceCard(name, metadata)
            card.clicked.connect(self._on_card_clicked)
            card.btn_upgrade.clicked.connect(lambda checked, v=name: self._upgrade_model(v))
            self.cards_layout.addWidget(card)
            self._cards.append(card)

            if name == self._selected_voice:
                card.set_selected(True)

        self.cards_layout.addStretch()

        if not self._cards:
            empty = QLabel("No models yet — train a voice to see it here")
            empty.setStyleSheet("color: #555; font-size: 14px; padding: 40px;")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.cards_layout.insertWidget(0, empty)

    def _load_metadata(self, voice: str) -> dict:
        meta_path = os.path.join(MODELS_DIR, voice, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    return json.load(f)
            except Exception:
                pass

        model_dir = os.path.join(MODELS_DIR, voice)
        g_files = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith("G_") and f.endswith(".pth")
        ])
        if g_files:
            epoch = g_files[-1].replace("G_", "").replace(".pth", "")
            return {"epochs": int(epoch) if epoch.isdigit() else 0}
        return {}

    def _on_card_clicked(self, voice_name: str):
        self._selected_voice = voice_name
        for card in self._cards:
            card.set_selected(card.voice_name == voice_name)
        self.actions_widget.setVisible(True)

    def _rename_model(self):
        if not self._selected_voice:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Model", "New name:", text=self._selected_voice
        )
        if not ok or not new_name.strip() or new_name.strip() == self._selected_voice:
            return

        new_name = new_name.strip()
        old_name = self._selected_voice

        old_dataset = os.path.join(DATASETS_DIR, old_name)
        new_dataset = os.path.join(DATASETS_DIR, new_name)
        if os.path.isdir(old_dataset):
            os.rename(old_dataset, new_dataset)

        old_model = os.path.join(MODELS_DIR, old_name)
        new_model = os.path.join(MODELS_DIR, new_name)
        if os.path.isdir(old_model):
            os.rename(old_model, new_model)

        self._selected_voice = new_name
        self._refresh_models()

    def _set_image(self):
        if not self._selected_voice:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Artist Image", "",
            "Images (*.png *.jpg *.jpeg *.webp);;All Files (*)",
        )
        if path:
            model_dir = os.path.join(MODELS_DIR, self._selected_voice)
            os.makedirs(model_dir, exist_ok=True)
            ext = os.path.splitext(path)[1]
            for old in ["image.png", "image.jpg", "image.jpeg", "image.webp"]:
                old_path = os.path.join(model_dir, old)
                if os.path.exists(old_path):
                    os.unlink(old_path)
            shutil.copy2(path, os.path.join(model_dir, f"image{ext}"))
            self._refresh_models()

    def _delete_model(self):
        if not self._selected_voice:
            return
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Delete '{self._selected_voice}' model?\nTraining data will be kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        model_dir = os.path.join(MODELS_DIR, self._selected_voice)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)

        self._selected_voice = ""
        self.actions_widget.setVisible(False)
        self._refresh_models()

    def _upgrade_model(self, voice_name: str = ""):
        """Switch to training page with this model's voice pre-selected for resume training."""
        voice = voice_name or self._selected_voice
        if not voice:
            return

        # Calculate suggested epochs to reach full convergence
        metadata = self._load_metadata(voice)
        epochs = metadata.get("epochs", 0)
        clips = metadata.get("dataset_clips", 1)
        old_batch = metadata.get("batch_size", 16)
        # The next run will use optimized batch size (128 on A40)
        new_batch = 128
        current_maturity = (epochs * old_batch) / clips
        target_maturity = 2000
        if current_maturity < target_maturity:
            additional_passes = (target_maturity - current_maturity) * clips
            additional_epochs = int(additional_passes / new_batch)
        else:
            additional_epochs = 500  # fallback

        main_window = self.window()
        if hasattr(main_window, 'training_page') and hasattr(main_window, 'dataset_page'):
            main_window.dataset_page.txt_voice.setText(voice)
            main_window.training_page.chk_resume.setChecked(True)
            main_window.training_page.txt_max_epochs.setText(str(additional_epochs))
            main_window.sidebar.setCurrentRow(2)
            # Auto-start training after a brief delay for UI to update
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(500, main_window.training_page._start_training)

    def get_selected_model(self) -> str:
        return self._selected_voice
