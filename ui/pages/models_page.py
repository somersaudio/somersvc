"""Models page — Spotify-style model library."""

import json
import os
import shutil
import zipfile

from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPoint, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QColor, QIcon
from PyQt6.QtWidgets import (
    QCompleter,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
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


class _SmoothScrollArea(QScrollArea):
    """Scroll area with smooth animated scrolling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target = 0
        self._anim = None

    def wheelEvent(self, event):
        sb = self.verticalScrollBar()
        if self._anim and self._anim.state() == QPropertyAnimation.State.Running:
            self._target = max(sb.minimum(), min(sb.maximum(), self._target - event.angleDelta().y()))
        else:
            self._target = max(sb.minimum(), min(sb.maximum(), sb.value() - event.angleDelta().y()))

        self._anim = QPropertyAnimation(sb, b"value")
        self._anim.setDuration(300)
        self._anim.setStartValue(sb.value())
        self._anim.setEndValue(self._target)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()
        event.accept()


class _ThumbCacheWorker(QThread):
    """Background worker to cache Spotify artist thumbnails."""
    thumb_ready = pyqtSignal(str, str)  # (artist_name, file_path)

    def __init__(self, models, cache_dir, spotify):
        super().__init__()
        self.models = models
        self.cache_dir = cache_dir
        self.spotify = spotify

    def run(self):
        import requests as _req
        for m in self.models:
            artist = m["artist"]
            path = os.path.join(self.cache_dir, f"{artist}.jpg")
            if os.path.exists(path):
                self.thumb_ready.emit(artist, path)
                continue

            if not self.spotify:
                continue

            try:
                url = self.spotify.search_artist_image(artist)
                if url:
                    r = _req.get(url, timeout=10)
                    if r.status_code == 200:
                        with open(path, "wb") as f:
                            f.write(r.content)
                        self.thumb_ready.emit(artist, path)
            except Exception:
                pass


class ModelsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_voice = ""
        self._cards: list[VoiceCard] = []
        self._init_ui()
        self._refresh_models()

    def _init_ui(self):
        # Background image label (behind everything)
        self.bg_image = QLabel(self)
        self.bg_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bg_image.setStyleSheet("background: transparent;")
        self.bg_image.lower()

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title row with text links
        title_row = QHBoxLayout()
        title = QLabel("Models")
        title.setObjectName("title")
        title_row.addWidget(title)
        title_row.addStretch()

        self.btn_import = QLabel("Import")
        self.btn_import.setStyleSheet("color: #666; font-size: 11px; background: transparent;")
        self.btn_import.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_import.mousePressEvent = lambda e: self._import_model()
        title_row.addWidget(self.btn_import)

        sep = QLabel("·")
        sep.setStyleSheet("color: #444; font-size: 11px; background: transparent;")
        title_row.addWidget(sep)

        self.btn_export_link = QLabel("Export")
        self.btn_export_link.setStyleSheet("color: #666; font-size: 11px; background: transparent;")
        self.btn_export_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_link.mousePressEvent = lambda e: self._export_model()
        title_row.addWidget(self.btn_export_link)

        layout.addLayout(title_row)

        # Browse Artists — live search bar with dropdown
        self.browse_bar = QLineEdit()
        self.browse_bar.setPlaceholderText("🔍  Search artists...")
        self.browse_bar.setStyleSheet(
            """
            QLineEdit {
                background-color: rgba(255, 255, 255, 5);
                color: #ccc;
                border: 1px solid rgba(255, 255, 255, 8);
                border-radius: 14px;
                padding: 6px 16px;
                font-size: 12px;
            }
            QLineEdit:focus {
                background-color: rgba(255, 255, 255, 10);
                border-color: rgba(255, 255, 255, 20);
                color: #eee;
            }
            """
        )
        self.browse_bar.setFixedHeight(32)
        self.browse_bar.setMaximumWidth(280)
        self.browse_bar.textChanged.connect(self._on_search_changed)
        self.browse_bar.mousePressEvent = self._on_search_clicked

        browse_row = QHBoxLayout()
        browse_row.addStretch()
        browse_row.addWidget(self.browse_bar)
        browse_row.addStretch()
        layout.addLayout(browse_row)

        # Dropdown list (floating, hidden by default)
        self._dropdown = QListWidget(self)
        self._dropdown.setStyleSheet(
            """
            QListWidget {
                background-color: rgba(20, 20, 20, 230);
                border: 1px solid rgba(255, 255, 255, 15);
                border-radius: 10px;
                padding: 4px;
                font-size: 12px;
                color: #ddd;
            }
            QListWidget::item {
                padding: 6px 10px;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background-color: rgba(255, 255, 255, 15);
            }
            QListWidget::item:hover {
                background-color: rgba(255, 255, 255, 10);
            }
            """
        )
        self._dropdown.setVisible(False)
        self._dropdown.setMinimumHeight(350)
        self._dropdown.setMaximumHeight(450)
        self._dropdown.setIconSize(QSize(32, 32))
        self._dropdown.itemClicked.connect(self._on_dropdown_item_clicked)
        self._hf_models = []
        self._hf_loaded = False
        self._image_cache_dir = os.path.join(APP_DIR, "data", "cache", "artist_thumbs")

        # Model cards scroll area
        self.cards_container = QWidget()
        self.cards_container.setStyleSheet("background: transparent;")
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setSpacing(4)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.addStretch()

        scroll = _SmoothScrollArea()
        scroll.setWidget(self.cards_container)
        scroll.viewport().setStyleSheet("background: transparent;")
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

        # Collect models and sort: user-trained first, then downloaded
        user_models = []
        downloaded_models = []

        for name in sorted(os.listdir(MODELS_DIR)):
            model_dir = os.path.join(MODELS_DIR, name)
            if not os.path.isdir(model_dir):
                continue

            files = os.listdir(model_dir)
            has_svc = any(f.startswith("G_") and f.endswith(".pth") for f in files)
            has_rvc = any(f.endswith(".pth") and not f.startswith(("G_", "D_")) for f in files)
            if not has_svc and not has_rvc:
                continue

            metadata = self._load_metadata(name)
            metadata["model_type"] = "rvc" if (has_rvc and not has_svc) else "svc"
            is_downloaded = metadata.get("source") == "downloaded"

            if is_downloaded:
                downloaded_models.append((name, model_dir, metadata))
            else:
                user_models.append((name, model_dir, metadata))

        for name, model_dir, metadata in user_models + downloaded_models:
            # Auto-fetch Spotify image if missing
            self._ensure_artist_image(name, model_dir)

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
        self._update_background(voice_name)

    def _update_background(self, voice_name: str):
        """Set the artist image as a faded background."""
        model_dir = os.path.join(MODELS_DIR, voice_name)
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            path = os.path.join(model_dir, f"image{ext}")
            if os.path.exists(path):
                img_path = path
                break

        if not img_path:
            self.bg_image.clear()
            return

        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            self.bg_image.clear()
            return

        # Scale to fill the page
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Apply Gaussian blur
        from PyQt6.QtWidgets import QGraphicsBlurEffect, QGraphicsScene, QGraphicsPixmapItem
        from PyQt6.QtCore import QRectF
        scene = QGraphicsScene()
        item = QGraphicsPixmapItem(scaled)
        blur = QGraphicsBlurEffect()
        blur.setBlurRadius(40)
        item.setGraphicsEffect(blur)
        scene.addItem(item)
        blurred = QPixmap(scaled.size())
        blurred.fill(QColor(0, 0, 0, 0))
        p = QPainter(blurred)
        scene.render(p, QRectF(blurred.rect()), QRectF(scaled.rect()))
        p.end()

        # Apply 10% opacity overlay
        faded = QPixmap(blurred.size())
        faded.fill(QColor(0, 0, 0, 0))
        painter = QPainter(faded)
        painter.setOpacity(0.10)
        painter.drawPixmap(0, 0, blurred)
        painter.end()

        self.bg_image.setPixmap(faded)

    def resizeEvent(self, event):
        """Keep background image sized to page."""
        super().resizeEvent(event)
        self.bg_image.setGeometry(0, 0, self.width(), self.height())
        if self._selected_voice:
            self._update_background(self._selected_voice)

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

    def _export_model(self):
        if not self._selected_voice:
            return
        model_dir = os.path.join(MODELS_DIR, self._selected_voice)
        if not os.path.isdir(model_dir):
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Model",
            os.path.expanduser(f"~/Desktop/{self._selected_voice}.svc"),
            "SomerSVC Model (*.svc);;All Files (*)",
        )
        if not save_path:
            return

        with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(model_dir):
                for f in files:
                    full = os.path.join(root, f)
                    arcname = os.path.join(self._selected_voice, os.path.relpath(full, model_dir))
                    zf.write(full, arcname)

        QMessageBox.information(self, "Exported", f"Model exported to:\n{save_path}")

    def _on_search_clicked(self, event):
        """Load models on first click, then show dropdown."""
        QLineEdit.mousePressEvent(self.browse_bar, event)
        if not self._hf_loaded:
            self._load_hf_models()
        else:
            self._show_dropdown()

    def _load_hf_models(self):
        """Load the HuggingFace model list."""
        from services.hf_model_browser import fetch_available_models
        self.browse_bar.setPlaceholderText("Loading artists...")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            self._hf_models = fetch_available_models()
            self._hf_loaded = True
            self.browse_bar.setPlaceholderText("🔍  Search artists...")
            self._show_dropdown()
            # Start caching artist thumbnails in background
            self._cache_artist_thumbnails()
        except Exception as e:
            self.browse_bar.setPlaceholderText("🔍  Search artists...")
            QMessageBox.warning(self, "Error", f"Could not load artists:\n{e}")

    def _cache_artist_thumbnails(self):
        """Cache Spotify artist thumbnails in background."""
        os.makedirs(self._image_cache_dir, exist_ok=True)
        # Only cache visible artists (first batch)
        self._thumb_worker = _ThumbCacheWorker(
            self._hf_models, self._image_cache_dir, self._get_spotify()
        )
        self._thumb_worker.thumb_ready.connect(self._on_thumb_ready)
        self._thumb_worker.start()

    def _on_thumb_ready(self, artist: str, path: str):
        """Update dropdown item icon when thumbnail is ready."""
        for i in range(self._dropdown.count()):
            item = self._dropdown.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole):
                if item.data(Qt.ItemDataRole.UserRole).get("artist") == artist:
                    pixmap = QPixmap(path)
                    if not pixmap.isNull():
                        from ui.widgets.voice_card import VoiceCard
                        circular = VoiceCard._make_circular(pixmap, 28)
                        item.setIcon(QIcon(circular))
                    break

    def _on_search_changed(self, text: str):
        if self._hf_loaded:
            self._show_dropdown(text)

    def _show_dropdown(self, filter_text: str = ""):
        """Show the dropdown list below the search bar."""
        ft = filter_text.lower()
        self._dropdown.clear()

        shown = 0
        for m in self._hf_models:
            if ft and ft not in m["artist"].lower():
                continue
            item = QListWidgetItem(m["artist"])
            item.setData(Qt.ItemDataRole.UserRole, m)

            # Load cached thumbnail
            thumb_path = os.path.join(self._image_cache_dir, f"{m['artist']}.jpg")
            if os.path.exists(thumb_path):
                pixmap = QPixmap(thumb_path)
                if not pixmap.isNull():
                    from ui.widgets.voice_card import VoiceCard
                    circular = VoiceCard._make_circular(pixmap, 28)
                    item.setIcon(QIcon(circular))

            self._dropdown.addItem(item)
            shown += 1
            if shown >= 50:
                break

        if shown > 0:
            # Position dropdown below the search bar, wider than the bar
            bar_pos = self.browse_bar.mapTo(self, QPoint(0, self.browse_bar.height()))
            dropdown_width = min(self.width() - 40, 500)
            bar_center = bar_pos.x() + self.browse_bar.width() // 2
            x = max(10, bar_center - dropdown_width // 2)
            self._dropdown.setFixedWidth(dropdown_width)
            self._dropdown.move(x, bar_pos.y() + 4)
            self._dropdown.setVisible(True)
            self._dropdown.raise_()
        else:
            self._dropdown.setVisible(False)

    def _on_dropdown_item_clicked(self, item):
        """Download the selected artist model."""
        model_info = item.data(Qt.ItemDataRole.UserRole)
        if not model_info:
            return

        artist = model_info["artist"]
        folder = model_info["folder"]

        self._dropdown.setVisible(False)
        self.browse_bar.clear()

        dest = os.path.join(MODELS_DIR, artist)
        if os.path.exists(dest):
            reply = QMessageBox.question(
                self, "Model Exists",
                f"'{artist}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            shutil.rmtree(dest)

        self.browse_bar.setPlaceholderText(f"Downloading {artist}...")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from services.hf_model_browser import download_model
            download_model(
                folder, dest,
                on_log=lambda msg: (
                    self.browse_bar.setPlaceholderText(msg),
                    QApplication.processEvents(),
                ),
            )

            # Inspect and mark as downloaded
            from services.model_inspector import inspect_model
            meta = inspect_model(dest)
            meta["source"] = "downloaded"
            with open(os.path.join(dest, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # Fetch artist image
            self._ensure_artist_image(artist, dest)

            self.browse_bar.setPlaceholderText("🔍  Search artists...")
            self._refresh_models()
            QMessageBox.information(self, "Downloaded", f"'{artist}' model downloaded!")
        except Exception as e:
            self.browse_bar.setPlaceholderText("🔍  Search artists...")
            QMessageBox.warning(self, "Download Failed", str(e))

    def _hide_dropdown_on_click(self, event):
        """Hide dropdown when clicking elsewhere."""
        if self._dropdown.isVisible():
            if not self._dropdown.geometry().contains(event.pos()):
                self._dropdown.setVisible(False)
        super().mousePressEvent(event)

    def mousePressEvent(self, event):
        """Hide dropdown when clicking outside."""
        if self._dropdown.isVisible():
            drop_geo = self._dropdown.geometry()
            bar_geo = self.browse_bar.geometry()
            if not drop_geo.contains(event.pos()) and not bar_geo.contains(event.pos()):
                self._dropdown.setVisible(False)
                self.browse_bar.clear()
        super().mousePressEvent(event)

    def _import_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Model", "",
            "All Models (*.svc *.pth *.zip);;SomerSVC Model (*.svc);;RVC Model (*.pth);;Zip Files (*.zip);;All Files (*)",
        )
        if not path:
            return

        # Handle loose .pth files (RVC models)
        if path.endswith(".pth"):
            name = os.path.splitext(os.path.basename(path))[0]
            # Clean up common suffixes
            for suffix in ["_v2", "_v1", "-v2", "-v1"]:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break

            name, ok = QInputDialog.getText(
                self, "Model Name", "Name for this model:", text=name
            )
            if not ok or not name.strip():
                return
            name = name.strip()

            dest = os.path.join(MODELS_DIR, name)
            os.makedirs(dest, exist_ok=True)
            shutil.copy2(path, os.path.join(dest, os.path.basename(path)))

            # Also copy .index file if one exists next to it
            pth_dir = os.path.dirname(path)
            for f in os.listdir(pth_dir):
                if f.endswith(".index"):
                    shutil.copy2(os.path.join(pth_dir, f), os.path.join(dest, f))

            self._refresh_models()
            QMessageBox.information(self, "Imported", f"RVC model '{name}' imported!")
            return

        # Handle .svc / .zip files
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                if not names:
                    QMessageBox.warning(self, "Invalid", "The file is empty.")
                    return

                # Get the top-level folder name from the zip
                top = names[0].split("/")[0]

                # Check for SVC or RVC checkpoint
                has_svc = any(
                    n.split("/")[-1].startswith("G_") and n.endswith(".pth")
                    for n in names
                )
                has_rvc = any(
                    n.endswith(".pth") and not n.split("/")[-1].startswith(("G_", "D_"))
                    for n in names
                )
                if not has_svc and not has_rvc:
                    QMessageBox.warning(self, "Invalid", "No model checkpoint found in this file.")
                    return

                dest = os.path.join(MODELS_DIR, top)
                if os.path.exists(dest):
                    reply = QMessageBox.question(
                        self, "Model Exists",
                        f"'{top}' already exists. Overwrite?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                    shutil.rmtree(dest)

                os.makedirs(MODELS_DIR, exist_ok=True)
                zf.extractall(MODELS_DIR)

            model_type = "RVC" if has_rvc and not has_svc else "SVC"
            self._refresh_models()
            QMessageBox.information(self, "Imported", f"{model_type} model '{top}' imported successfully!")
        except zipfile.BadZipFile:
            QMessageBox.warning(self, "Invalid File", "This is not a valid model file.")

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
