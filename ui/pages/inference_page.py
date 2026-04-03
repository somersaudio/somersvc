"""Inference page for voice conversion using trained models."""

import os
from pathlib import Path

from PyQt6.QtCore import Qt
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

from ui.widgets.audio_player import AudioPlayer
from ui.widgets.log_viewer import LogViewer
from workers.inference_worker import InferenceWorker

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(APP_DIR, "data", "models")
OUTPUT_DIR = os.path.join(APP_DIR, "data", "output")


class InferencePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: InferenceWorker | None = None
        self._output_path: str | None = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Voice Conversion")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Convert audio using a trained voice model")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.cmb_model = QComboBox()
        self.cmb_model.setMinimumWidth(250)
        model_row.addWidget(self.cmb_model, 1)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh_models)
        model_row.addWidget(self.btn_refresh)
        layout.addLayout(model_row)

        # Source audio
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source Audio:"))
        self.txt_source = QLineEdit()
        self.txt_source.setPlaceholderText("Select a WAV file to convert...")
        self.txt_source.setReadOnly(True)
        source_row.addWidget(self.txt_source, 1)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_source)
        source_row.addWidget(self.btn_browse)
        layout.addLayout(source_row)

        # Source audio player
        layout.addWidget(QLabel("Source Audio Preview:"))
        self.player_source = AudioPlayer()
        layout.addWidget(self.player_source)

        # --- Settings ---
        settings_group = QGroupBox("Inference Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Row 1: Transpose + F0 Method
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

        # Row 2: Noise Scale + DB Thresh + Cluster Ratio
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Noise Scale:"))
        self.spn_noise = QDoubleSpinBox()
        self.spn_noise.setRange(0.0, 1.0)
        self.spn_noise.setSingleStep(0.05)
        self.spn_noise.setValue(0.3)
        self.spn_noise.setToolTip("Controls breathiness/noise (0 = clean, 1 = noisy)")
        row2.addWidget(self.spn_noise)

        row2.addSpacing(15)
        row2.addWidget(QLabel("DB Thresh:"))
        self.spn_db = QSpinBox()
        self.spn_db.setRange(-100, 0)
        self.spn_db.setValue(-35)
        self.spn_db.setToolTip("Silence threshold in dB (lower = more sensitive)")
        row2.addWidget(self.spn_db)

        row2.addSpacing(15)
        row2.addWidget(QLabel("Pad (s):"))
        self.spn_pad = QDoubleSpinBox()
        self.spn_pad.setRange(0.0, 5.0)
        self.spn_pad.setSingleStep(0.1)
        self.spn_pad.setValue(1.0)
        self.spn_pad.setToolTip("Padding seconds added to each chunk")
        row2.addWidget(self.spn_pad)

        row2.addSpacing(15)
        row2.addWidget(QLabel("Chunk (s):"))
        self.spn_chunk = QDoubleSpinBox()
        self.spn_chunk.setRange(0.1, 30.0)
        self.spn_chunk.setSingleStep(0.5)
        self.spn_chunk.setValue(10.0)
        self.spn_chunk.setToolTip("Chunk size for processing — larger = better quality, more RAM")
        row2.addWidget(self.spn_chunk)

        row2.addStretch()
        settings_layout.addLayout(row2)

        layout.addWidget(settings_group)

        # Convert button
        btn_row = QHBoxLayout()
        self.btn_convert = QPushButton("Convert Voice")
        self.btn_convert.setObjectName("primary")
        self.btn_convert.clicked.connect(self._convert)
        btn_row.addWidget(self.btn_convert)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Log
        self.log_viewer = LogViewer()
        self.log_viewer.setMaximumHeight(120)
        layout.addWidget(self.log_viewer)

        # Output audio player
        layout.addWidget(QLabel("Converted Output:"))
        self.player_output = AudioPlayer()
        layout.addWidget(self.player_output)

        # Open output folder
        self.btn_open_folder = QPushButton("Open Output Folder")
        self.btn_open_folder.clicked.connect(self._open_output_folder)
        layout.addWidget(self.btn_open_folder)

        layout.addStretch()

        # Initial model refresh
        self._refresh_models()

    def _refresh_models(self):
        self.cmb_model.clear()
        if not os.path.exists(MODELS_DIR):
            return

        for name in sorted(os.listdir(MODELS_DIR)):
            model_dir = os.path.join(MODELS_DIR, name)
            if not os.path.isdir(model_dir):
                continue
            # Check for checkpoint and config
            has_model = any(f.startswith("G_") and f.endswith(".pth") for f in os.listdir(model_dir))
            has_config = os.path.exists(os.path.join(model_dir, "config.json"))
            if has_model and has_config:
                self.cmb_model.addItem(name, model_dir)

        if self.cmb_model.count() == 0:
            self.cmb_model.addItem("No models found - train one first")

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

    def _convert(self):
        if self.cmb_model.count() == 0 or not self.cmb_model.currentData():
            QMessageBox.warning(self, "No Model", "No trained model available. Train one first.")
            return

        source = self.txt_source.text().strip()
        if not source or not os.path.exists(source):
            QMessageBox.warning(self, "No Source", "Select a source audio file first.")
            return

        model_dir = self.cmb_model.currentData()
        speaker = self.cmb_model.currentText()

        # Find latest checkpoint
        model_files = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith("G_") and f.endswith(".pth")
        ])
        if not model_files:
            QMessageBox.warning(self, "No Model", "No checkpoint found in model directory.")
            return

        model_path = os.path.join(model_dir, model_files[-1])
        config_path = os.path.join(model_dir, "config.json")

        self.log_viewer.clear_log()
        self.btn_convert.setEnabled(False)

        self._worker = InferenceWorker(
            source_wav=source,
            model_path=model_path,
            config_path=config_path,
            output_dir=OUTPUT_DIR,
            speaker=speaker,
            transpose=self.spn_transpose.value(),
            f0_method=self.cmb_f0.currentText(),
            auto_predict_f0=self.chk_auto_f0.isChecked(),
            noise_scale=self.spn_noise.value(),
            db_thresh=self.spn_db.value(),
            pad_seconds=self.spn_pad.value(),
            chunk_seconds=self.spn_chunk.value(),
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

    def _open_output_folder(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.system(f'open "{OUTPUT_DIR}"')
