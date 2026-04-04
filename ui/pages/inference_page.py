"""Inference page for voice conversion using trained models."""

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from services.rvc_inference_runner import detect_model_type, _get_rvc_pth_files
from ui.widgets.audio_player import AudioPlayer
from ui.widgets.log_viewer import LogViewer
from workers.inference_worker import InferenceWorker

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(APP_DIR, "data", "models")
OUTPUT_DIR = os.path.join(APP_DIR, "data", "output")


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _hz_to_note(hz: float) -> str:
    """Convert frequency in Hz to musical note name."""
    import math
    if hz <= 0:
        return "?"
    midi = 69 + 12 * math.log2(hz / 440.0)
    midi = round(midi)
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


def _note_to_hz(note: str) -> float:
    """Convert a note name like 'C3' to Hz."""
    import re
    m = re.match(r"([A-G]#?)(\d+)", note)
    if not m:
        return 0
    name, octave = m.group(1), int(m.group(2))
    idx = NOTE_NAMES.index(name)
    midi = (octave + 1) * 12 + idx
    return 440.0 * (2 ** ((midi - 69) / 12))


class _PitchWorker(QThread):
    """Background worker to analyze vocal pitch range."""
    result = pyqtSignal(str, float)  # (display_text, median_hz)

    def __init__(self, audio_path: str):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            import numpy as np
            import librosa

            y, sr = librosa.load(self.audio_path, sr=22050, duration=120)
            f0, voiced, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"), sr=sr,
            )

            # Filter to voiced frames only
            voiced_f0 = f0[voiced & ~np.isnan(f0)]
            if len(voiced_f0) == 0:
                self.result.emit("No pitched vocals detected", 0)
                return

            low = np.percentile(voiced_f0, 5)
            high = np.percentile(voiced_f0, 95)
            median = np.median(voiced_f0)

            low_note = _hz_to_note(low)
            high_note = _hz_to_note(high)
            med_note = _hz_to_note(median)

            self.result.emit(
                f"Vocal range: {low_note} – {high_note}  ·  Center: {med_note}  ({int(median)} Hz)",
                float(median),
            )
        except Exception as e:
            self.result.emit(f"Pitch detection failed: {e}", 0)


class _KeyDetectWorker(QThread):
    """Analyze multiple clips to detect a voice's natural key."""
    result = pyqtSignal(str)  # note name like "C3"

    def __init__(self, clip_paths: list[str]):
        super().__init__()
        self.clip_paths = clip_paths

    def run(self):
        try:
            import numpy as np
            import librosa

            all_f0 = []
            for path in self.clip_paths:
                try:
                    y, sr = librosa.load(path, sr=22050)
                    f0, voiced, _ = librosa.pyin(
                        y, fmin=librosa.note_to_hz("C2"),
                        fmax=librosa.note_to_hz("C6"), sr=sr,
                    )
                    voiced_f0 = f0[voiced & ~np.isnan(f0)]
                    all_f0.extend(voiced_f0.tolist())
                except Exception:
                    continue

            if not all_f0:
                self.result.emit("")
                return

            median = np.median(all_f0)
            note = _hz_to_note(median)
            self.result.emit(note)
        except Exception:
            self.result.emit("")


class InferencePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: InferenceWorker | None = None
        self._output_path: str | None = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title row with Advanced toggle
        title_row = QHBoxLayout()
        title = QLabel("Voice Conversion")
        title.setObjectName("title")
        title_row.addWidget(title)
        title_row.addStretch()

        self.btn_advanced = QLabel("Advanced")
        self.btn_advanced.setStyleSheet("color: #666; font-size: 11px; background: transparent;")
        self.btn_advanced.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_advanced.mousePressEvent = lambda e: self._toggle_advanced()
        title_row.addWidget(self.btn_advanced)
        layout.addLayout(title_row)

        # =====================
        # MODEL SELECTOR (visual cards)
        # =====================
        self.cmb_model = QComboBox()  # hidden, used for data
        self.cmb_model.setVisible(False)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        layout.addWidget(self.cmb_model)

        from PyQt6.QtWidgets import QScrollArea
        self._model_scroll = QScrollArea()
        self._model_scroll.setFixedHeight(80)
        self._model_scroll.setWidgetResizable(True)
        self._model_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._model_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._model_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._model_scroll.viewport().setStyleSheet("background: transparent;")

        self._model_cards_widget = QWidget()
        self._model_cards_widget.setStyleSheet("background: transparent;")
        self._model_cards_layout = QHBoxLayout(self._model_cards_widget)
        self._model_cards_layout.setContentsMargins(0, 0, 0, 0)
        self._model_cards_layout.setSpacing(12)
        self._model_cards_layout.addStretch()
        self._model_scroll.setWidget(self._model_cards_widget)
        layout.addWidget(self._model_scroll)

        self._model_card_widgets = []
        self._selected_model_idx = -1

        # =====================
        # SOURCE AUDIO
        # =====================
        source_row = QHBoxLayout()
        self.txt_source = QLineEdit()
        self.txt_source.setPlaceholderText("Drop or browse for audio...")
        self.txt_source.setReadOnly(True)
        source_row.addWidget(self.txt_source, 1)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_source)
        source_row.addWidget(self.btn_browse)
        layout.addLayout(source_row)

        # Source player
        self.player_source = AudioPlayer()
        layout.addWidget(self.player_source)

        # Pitch detection display
        self.lbl_pitch = QLabel("")
        self.lbl_pitch.setStyleSheet(
            "color: #aaa; font-size: 11px; background: transparent; padding: 2px 0;"
        )
        layout.addWidget(self.lbl_pitch)

        # =====================
        # CONVERT BUTTON
        # =====================
        self.btn_convert = QPushButton("Convert")
        self.btn_convert.setObjectName("primary")
        self.btn_convert.setFixedHeight(40)
        self.btn_convert.clicked.connect(self._convert)
        layout.addWidget(self.btn_convert)

        # Log (hidden in simple view)
        self.log_viewer = LogViewer()
        self.log_viewer.setMaximumHeight(100)
        layout.addWidget(self.log_viewer)

        # =====================
        # OUTPUT PLAYER
        # =====================
        self.player_output = AudioPlayer()
        layout.addWidget(self.player_output)

        # Open output folder
        self.btn_open_folder = QPushButton("Open Output Folder")
        self.btn_open_folder.clicked.connect(self._open_output_folder)
        layout.addWidget(self.btn_open_folder)

        # =====================
        # ADVANCED SECTION (hidden by default)
        # =====================
        self._advanced_widget = QWidget()
        adv_layout = QVBoxLayout(self._advanced_widget)
        adv_layout.setContentsMargins(0, 8, 0, 0)
        adv_layout.setSpacing(8)

        # Model key row
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("Model Key:"))
        self.cmb_model_key = QComboBox()
        self.cmb_model_key.setToolTip("Set the model's natural vocal key — used to auto-calculate transpose")
        keys = ["Auto"]
        for octave in range(1, 6):
            for note in NOTE_NAMES:
                keys.append(f"{note}{octave}")
        self.cmb_model_key.addItems(keys)
        self.cmb_model_key.setFixedWidth(75)
        self.cmb_model_key.currentIndexChanged.connect(self._update_auto_transpose)
        key_row.addWidget(self.cmb_model_key)

        self.btn_detect_key = QPushButton("Detect")
        self.btn_detect_key.setToolTip("Auto-detect this model's vocal key from training data")
        self.btn_detect_key.setFixedWidth(55)
        self.btn_detect_key.clicked.connect(self._detect_model_key)
        key_row.addWidget(self.btn_detect_key)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh_models)
        key_row.addWidget(self.btn_refresh)

        key_row.addStretch()
        adv_layout.addLayout(key_row)

        # Settings group
        settings_group = QGroupBox("Inference Settings")
        settings_layout = QVBoxLayout(settings_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Transpose:"))
        self.spn_transpose = QSpinBox()
        self.spn_transpose.setRange(-24, 24)
        self.spn_transpose.setValue(0)
        self.spn_transpose.setToolTip("Pitch shift in semitones (+ = higher, - = lower)")
        row1.addWidget(self.spn_transpose)

        row1.addSpacing(15)
        row1.addWidget(QLabel("F0 Method:"))
        self.cmb_f0 = QComboBox()
        self.cmb_f0.addItems(["crepe", "crepe-tiny", "dio", "parselmouth", "harvest"])
        self.cmb_f0.setToolTip("Pitch detection method (crepe = best quality, dio = fastest)")
        row1.addWidget(self.cmb_f0)

        row1.addSpacing(15)
        self.chk_auto_f0 = QCheckBox("Auto Predict F0")
        self.chk_auto_f0.setChecked(False)
        self.chk_auto_f0.setToolTip("Auto pitch prediction — leave OFF for singing, ON for speech")
        row1.addWidget(self.chk_auto_f0)

        row1.addStretch()
        settings_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Noise Scale:"))
        self.spn_noise = QDoubleSpinBox()
        self.spn_noise.setRange(0.0, 1.0)
        self.spn_noise.setSingleStep(0.05)
        self.spn_noise.setValue(0.1)
        row2.addWidget(self.spn_noise)

        row2.addSpacing(15)
        row2.addWidget(QLabel("DB Thresh:"))
        self.spn_db = QSpinBox()
        self.spn_db.setRange(-100, 0)
        self.spn_db.setValue(-35)
        row2.addWidget(self.spn_db)

        row2.addSpacing(15)
        row2.addWidget(QLabel("Pad (s):"))
        self.spn_pad = QDoubleSpinBox()
        self.spn_pad.setRange(0.0, 5.0)
        self.spn_pad.setSingleStep(0.1)
        self.spn_pad.setValue(1.0)
        row2.addWidget(self.spn_pad)

        row2.addSpacing(15)
        row2.addWidget(QLabel("Chunk (s):"))
        self.spn_chunk = QDoubleSpinBox()
        self.spn_chunk.setRange(0.1, 60.0)
        self.spn_chunk.setSingleStep(0.5)
        self.spn_chunk.setValue(30.0)
        row2.addWidget(self.spn_chunk)

        row2.addStretch()
        settings_layout.addLayout(row2)

        adv_layout.addWidget(settings_group)

        # Vocal separation
        self.chk_separate = QCheckBox("Separate Vocals First (Demucs)")
        self.chk_separate.setChecked(False)
        self.chk_separate.setToolTip("Isolate vocals before converting, then remix with instrumentals.")
        adv_layout.addWidget(self.chk_separate)

        # Smart transpose
        self.chk_smart_transpose = QCheckBox("Smart Transpose (per-section octave matching)")
        self.chk_smart_transpose.setChecked(True)
        self.chk_smart_transpose.setToolTip(
            "Splits audio into sections at silence gaps, transposes each by ±12 semitones "
            "to best match the model's range. Keeps everything in key."
        )
        adv_layout.addWidget(self.chk_smart_transpose)

        self._advanced_widget.setVisible(False)
        layout.addWidget(self._advanced_widget)

        layout.addStretch()

        # Initial model refresh
        self._refresh_models()

    def _toggle_advanced(self):
        visible = not self._advanced_widget.isVisible()
        self._advanced_widget.setVisible(visible)
        self.btn_advanced.setText("Simple" if visible else "Advanced")
        self.btn_advanced.setStyleSheet(
            f"color: {'#aaa' if visible else '#666'}; font-size: 11px; background: transparent;"
        )

    def _refresh_models(self):
        self.cmb_model.clear()

        # Clear visual cards
        for w in self._model_card_widgets:
            w.setParent(None)
            w.deleteLater()
        self._model_card_widgets.clear()

        # Remove stretch
        while self._model_cards_layout.count():
            item = self._model_cards_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        if not os.path.exists(MODELS_DIR):
            self._model_cards_layout.addStretch()
            return

        for name in sorted(os.listdir(MODELS_DIR)):
            model_dir = os.path.join(MODELS_DIR, name)
            if not os.path.isdir(model_dir):
                continue
            model_type = detect_model_type(model_dir)
            if model_type == "svc":
                has_model = any(f.startswith("G_") and f.endswith(".pth") for f in os.listdir(model_dir))
                if not has_model:
                    continue
            elif model_type != "rvc":
                continue

            idx = self.cmb_model.count()
            self.cmb_model.addItem(f"{name} [{model_type.upper()}]", model_dir)

            # Build visual card
            card = self._build_model_card(name, model_dir, idx)
            self._model_cards_layout.addWidget(card)
            self._model_card_widgets.append(card)

        self._model_cards_layout.addStretch()

        if self.cmb_model.count() == 0:
            empty = QLabel("No models")
            empty.setStyleSheet("color: #555; font-size: 12px;")
            self._model_cards_layout.insertWidget(0, empty)

    def _build_model_card(self, name: str, model_dir: str, idx: int) -> QWidget:
        """Build a clickable model card with image and name."""
        card = QWidget()
        card.setFixedSize(64, 74)
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setStyleSheet("background: transparent;")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(4)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Image
        lbl_img = QLabel()
        lbl_img.setFixedSize(48, 48)
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_img.setStyleSheet(
            "background-color: #333; border-radius: 24px; border: 2px solid #444;"
        )

        # Load image
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            img_path = os.path.join(model_dir, f"image{ext}")
            if os.path.exists(img_path):
                from ui.widgets.voice_card import VoiceCard
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    lbl_img.setPixmap(VoiceCard._make_circular(pixmap, 44))
                break
        else:
            initials = "".join([w[0].upper() for w in name.split("_")[:2] if w])
            if not initials and name:
                initials = name[0].upper()
            lbl_img.setText(initials)
            lbl_img.setStyleSheet(
                "background-color: #2563eb; border-radius: 24px; border: 2px solid #444; "
                "color: white; font-size: 16px; font-weight: bold;"
            )

        card_layout.addWidget(lbl_img, alignment=Qt.AlignmentFlag.AlignCenter)

        # Name
        lbl_name = QLabel(name)
        lbl_name.setStyleSheet("color: #aaa; font-size: 9px; background: transparent;")
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_name.setMaximumWidth(64)
        card_layout.addWidget(lbl_name)

        # Store refs for selection highlighting
        card._img_label = lbl_img
        card._card_idx = idx

        card.mousePressEvent = lambda e, i=idx: self._select_model_card(i)
        return card

    def _select_model_card(self, idx: int):
        """Select a model card visually and sync with combo box."""
        self._selected_model_idx = idx
        self.cmb_model.setCurrentIndex(idx)

        for card in self._model_card_widgets:
            img = card._img_label
            is_sel = card._card_idx == idx
            border = "#ffffff" if is_sel else "#444"
            size = 60 if is_sel else 48

            import re as _re
            style = img.styleSheet()
            style = _re.sub(r"border: \dpx solid #[0-9a-fA-F]+", f"border: 2px solid {border}", style)
            style = _re.sub(r"border-radius: \d+px", f"border-radius: {size // 2}px", style)
            img.setStyleSheet(style)
            img.setFixedSize(size, size)

            # Re-render image at new size if it has a pixmap
            if not img.pixmap() or img.pixmap().isNull():
                continue
            model_dir = self.cmb_model.itemData(card._card_idx)
            if model_dir:
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    img_path = os.path.join(model_dir, f"image{ext}")
                    if os.path.exists(img_path):
                        from ui.widgets.voice_card import VoiceCard
                        pixmap = QPixmap(img_path)
                        img.setPixmap(VoiceCard._make_circular(pixmap, size - 4))
                        break

    def _browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source Audio",
            "",
            "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if path:
            self.txt_source.setText(path)
            self.player_source.load(path)
            self._analyze_pitch(path)

    def _analyze_pitch(self, path: str):
        """Detect vocal pitch range of the source audio."""
        self.lbl_pitch.setText("Analyzing pitch...")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        self._pitch_worker = _PitchWorker(path)
        self._pitch_worker.result.connect(self._on_pitch_result)
        self._pitch_worker.start()

    def _on_pitch_result(self, text: str, median_hz: float = 0):
        self.lbl_pitch.setText(text)
        if median_hz > 0:
            self._source_median_hz = median_hz
            self._update_auto_transpose()

    def _convert(self):
        if self.cmb_model.count() == 0 or not self.cmb_model.currentData():
            QMessageBox.warning(self, "No Model", "No trained model available. Train one first.")
            return

        source = self.txt_source.text().strip()
        if not source or not os.path.exists(source):
            QMessageBox.warning(self, "No Source", "Select a source audio file first.")
            return

        model_dir = self.cmb_model.currentData()
        model_type = detect_model_type(model_dir)

        if model_type == "svc":
            # Find latest SVC checkpoint
            model_files = sorted(
                [f for f in os.listdir(model_dir) if f.startswith("G_") and f.endswith(".pth")],
                key=lambda f: int(f.replace("G_", "").replace(".pth", "")) if f.replace("G_", "").replace(".pth", "").isdigit() else 0,
            )
            if not model_files:
                QMessageBox.warning(self, "No Model", "No checkpoint found in model directory.")
                return
            model_path = os.path.join(model_dir, model_files[-1])
            config_path = os.path.join(model_dir, "config.json")
        else:
            # Find RVC .pth file (exclude SVC G_/D_ checkpoints)
            rvc_files = _get_rvc_pth_files(os.listdir(model_dir))
            if not rvc_files:
                QMessageBox.warning(self, "No Model", "No RVC model found.")
                return
            model_path = os.path.join(model_dir, rvc_files[0])
            config_path = ""

        self.log_viewer.clear_log()
        self.btn_convert.setEnabled(False)

        self._worker = InferenceWorker(
            source_wav=source,
            model_path=model_path,
            config_path=config_path,
            output_dir=OUTPUT_DIR,
            speaker=self.cmb_model.currentText(),
            transpose=self.spn_transpose.value(),
            f0_method=self.cmb_f0.currentText(),
            auto_predict_f0=self.chk_auto_f0.isChecked(),
            noise_scale=self.spn_noise.value(),
            db_thresh=self.spn_db.value(),
            pad_seconds=self.spn_pad.value(),
            chunk_seconds=self.spn_chunk.value(),
            separate_vocals=self.chk_separate.isChecked(),
            model_type=model_type,
            model_dir=model_dir,
            smart_transpose=self.chk_smart_transpose.isChecked(),
            model_center_hz=_note_to_hz(self.cmb_model_key.currentText()) if self.cmb_model_key.currentText() != "Auto" else 0,
        )
        self._worker.log_line.connect(self.log_viewer.append_line)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, output_path: str):
        self.btn_convert.setEnabled(True)
        self._output_path = output_path
        self.player_output.load(output_path)
        self.log_viewer.append_line(f"Done! Output: {output_path}")

    def _on_error(self, error: str):
        self.btn_convert.setEnabled(True)
        self.log_viewer.append_line(f"ERROR: {error}")
        QMessageBox.critical(self, "Error", f"Inference failed:\n{error}")

    def _detect_model_key(self):
        """Auto-detect model's vocal key from training dataset clips."""
        model_name = self.cmb_model.currentText().split(" [")[0]
        dataset_dir = os.path.join(APP_DIR, "data", "datasets", model_name)

        if not os.path.isdir(dataset_dir):
            QMessageBox.information(
                self, "No Training Data",
                f"No training dataset found for '{model_name}'.\n\n"
                "Set the key manually using the dropdown.",
            )
            return

        clips = [
            os.path.join(dataset_dir, f)
            for f in sorted(os.listdir(dataset_dir))
            if f.endswith((".wav", ".flac", ".mp3"))
        ]
        if not clips:
            QMessageBox.information(self, "No Clips", "No audio clips found in dataset.")
            return

        self.btn_detect_key.setEnabled(False)
        self.btn_detect_key.setText("...")

        self._key_worker = _KeyDetectWorker(clips[:20])  # analyze up to 20 clips
        self._key_worker.result.connect(self._on_key_detected)
        self._key_worker.start()

    def _on_key_detected(self, note: str):
        self.btn_detect_key.setEnabled(True)
        self.btn_detect_key.setText("Detect")
        if note:
            idx = self.cmb_model_key.findText(note)
            if idx >= 0:
                self.cmb_model_key.setCurrentIndex(idx)

    def _on_model_changed(self, index):
        """Load saved vocal key or auto-detect from training data."""
        model_dir = self.cmb_model.currentData()
        if not model_dir:
            return

        # Clear previous pitch info
        self.lbl_pitch.setText("")

        # Check for saved key in metadata
        saved_key = ""
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            import json
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                saved_key = meta.get("vocal_key", "")
            except Exception:
                pass

        if saved_key and saved_key != "Auto":
            idx = self.cmb_model_key.findText(saved_key)
            if idx >= 0:
                self.cmb_model_key.setCurrentIndex(idx)
                self._update_auto_transpose()
                return

        # No saved key — try to auto-detect from training data
        model_name = self.cmb_model.currentText().split(" [")[0]
        dataset_dir = os.path.join(APP_DIR, "data", "datasets", model_name)

        if os.path.isdir(dataset_dir):
            clips = [
                os.path.join(dataset_dir, f)
                for f in sorted(os.listdir(dataset_dir))
                if f.endswith((".wav", ".flac", ".mp3"))
            ][:20]
            if clips:
                self._key_worker = _KeyDetectWorker(clips)
                self._key_worker.result.connect(self._on_key_detected)
                self._key_worker.start()
                return

        self.cmb_model_key.setCurrentIndex(0)
        self._update_auto_transpose()

    def _update_auto_transpose(self):
        """Auto-calculate transpose from source pitch and model key."""
        model_key = self.cmb_model_key.currentText()

        # Save the key to metadata
        model_dir = self.cmb_model.currentData()
        if model_dir and model_key != "Auto":
            meta_path = os.path.join(model_dir, "metadata.json")
            import json
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                except Exception:
                    pass
            meta["vocal_key"] = model_key
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        if model_key == "Auto" or not hasattr(self, '_source_median_hz'):
            return

        # Calculate semitone difference
        import math
        source_hz = self._source_median_hz
        target_hz = _note_to_hz(model_key)
        if source_hz > 0 and target_hz > 0:
            semitones = round(12 * math.log2(target_hz / source_hz))
            self.spn_transpose.setValue(semitones)
            self.lbl_pitch.setText(
                self.lbl_pitch.text() + f"  →  Transpose: {semitones:+d} (to match {model_key})"
            )

    def _open_output_folder(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.system(f'open "{OUTPUT_DIR}"')
