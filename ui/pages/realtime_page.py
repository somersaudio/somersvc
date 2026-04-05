"""Realtime voice conversion page — live mic input to converted output."""

import os
import subprocess

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.widgets.knob import Knob
from ui.widgets.log_viewer import LogViewer

from services.paths import MODELS_DIR


class RealtimePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: subprocess.Popen | None = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Realtime Voice")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Convert your microphone input in real time")
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

        # Quality selector
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Pitch Detection:"))
        self.cmb_f0 = QComboBox()
        self.cmb_f0.addItems(["Fast (dio)", "Balanced (crepe-tiny)", "Accurate (parselmouth)"])
        self.cmb_f0.setToolTip(
            "How the pitch of your voice is tracked.\n\n"
            "Fast (dio) — Lowest latency, good for live performance. May have occasional pitch glitches.\n\n"
            "Balanced (crepe-tiny) — Better pitch tracking with slightly more latency. Good middle ground.\n\n"
            "Accurate (parselmouth) — Most precise pitch detection. Best for recording but adds some delay."
        )
        quality_row.addWidget(self.cmb_f0)
        quality_row.addStretch()
        layout.addLayout(quality_row)

        # --- Knobs ---
        knobs_frame = QFrame()
        knobs_frame.setStyleSheet(
            """
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 10px;
                padding: 16px;
            }
            """
        )
        knobs_layout = QHBoxLayout(knobs_frame)
        knobs_layout.setSpacing(12)

        self.knob_pitch = Knob(
            label="Pitch",
            min_val=-24, max_val=24, default=0, step=1,
            suffix="st", decimals=0,
            tooltip=(
                "Shifts the pitch of your converted voice up or down in semitones. "
                "Use this when the source and target voice have different ranges. "
                "+12 shifts up one octave, -12 shifts down one octave. "
                "Start at 0 and adjust by ear until it sounds natural."
            ),
        )
        knobs_layout.addWidget(self.knob_pitch)

        self.knob_tone = Knob(
            label="Tone",
            min_val=0.0, max_val=1.0, default=0.4, step=0.05,
            suffix="", decimals=2,
            tooltip=(
                "Controls the breathiness and texture of the converted voice. "
                "Lower values produce a cleaner, smoother tone — like a polished studio vocal. "
                "Higher values add more air and natural breath noise. "
                "Start around 0.3-0.4 and increase if the voice sounds too robotic."
            ),
        )
        knobs_layout.addWidget(self.knob_tone)

        self.knob_response = Knob(
            label="Response",
            min_val=0.1, max_val=1.5, default=0.5, step=0.05,
            suffix="s", decimals=2,
            tooltip=(
                "How quickly the voice conversion responds to your input. "
                "Lower values mean less delay but require more processing power — "
                "if you hear glitches or crackling, increase this. "
                "Higher values add more latency but produce smoother, more stable output. "
                "0.3s feels near-instant, 0.5s is a safe default, 1.0s+ is very stable."
            ),
        )
        knobs_layout.addWidget(self.knob_response)

        self.knob_buffer = Knob(
            label="Buffer",
            min_val=0.1, max_val=1.5, default=0.5, step=0.05,
            suffix="s", decimals=2,
            tooltip=(
                "The size of the audio buffer fed to the GPU for processing. "
                "Think of it like a DAW's buffer size — lower is faster but less stable. "
                "If you hear pops, clicks, or dropouts, increase the buffer. "
                "Match this roughly to your Response setting for best results."
            ),
        )
        knobs_layout.addWidget(self.knob_buffer)

        self.knob_smoothing = Knob(
            label="Smoothing",
            min_val=0.0, max_val=0.15, default=0.01, step=0.005,
            suffix="s", decimals=3,
            tooltip=(
                "Crossfade applied between each processed audio chunk. "
                "Prevents clicks and pops at chunk boundaries — similar to a crossfade between clips. "
                "Keep this low (0.01-0.02s) for minimal latency. "
                "Increase to 0.05-0.1s if you hear clicking artifacts."
            ),
        )
        knobs_layout.addWidget(self.knob_smoothing)

        self.knob_gate = Knob(
            label="Gate",
            min_val=-80, max_val=0, default=-30, step=1,
            suffix="dB", decimals=0,
            tooltip=(
                "Noise gate threshold — audio below this level is treated as silence. "
                "Works just like a gate plugin on a channel strip. "
                "Set it so background noise is silenced but your voice comes through cleanly. "
                "-30dB is a good starting point. Lower it if quiet parts of your voice are being cut off."
            ),
        )
        knobs_layout.addWidget(self.knob_gate)

        layout.addWidget(knobs_frame)

        # Start/Stop buttons
        btn_row = QHBoxLayout()

        self.btn_start = QPushButton("Start Realtime")
        self.btn_start.setFixedHeight(50)
        self.btn_start.setStyleSheet(
            """
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 32px;
            }
            QPushButton:hover { background-color: #3b82f6; }
            QPushButton:disabled { background-color: #1e3a5f; color: #666; }
            """
        )
        self.btn_start.clicked.connect(self._start)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedHeight(50)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            """
            QPushButton {
                background-color: #dc2626;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 32px;
            }
            QPushButton:hover { background-color: #ef4444; }
            QPushButton:disabled { background-color: #3a1a1a; color: #666; }
            """
        )
        self.btn_stop.clicked.connect(self._stop)
        btn_row.addWidget(self.btn_stop)

        btn_row.addStretch()

        self.chk_passthrough = QCheckBox("Monitor Original")
        self.chk_passthrough.setToolTip(
            "Outputs your original unprocessed voice alongside the converted output. "
            "Useful for checking how much latency the conversion adds."
        )
        btn_row.addWidget(self.chk_passthrough)

        self.chk_auto_f0 = QCheckBox("Auto Pitch")
        self.chk_auto_f0.setChecked(False)
        self.chk_auto_f0.setToolTip(
            "Automatically re-predicts the pitch of your voice. "
            "Leave this OFF for singing — it preserves your original melody. "
            "Turn it ON for speech if you want the output to match the target voice's natural pitch."
        )
        btn_row.addWidget(self.chk_auto_f0)

        layout.addLayout(btn_row)

        # Status
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: #555;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)

        # Log
        layout.addWidget(QLabel("Output:"))
        self.log_viewer = LogViewer()
        self.log_viewer.setMaximumHeight(120)
        layout.addWidget(self.log_viewer, 1)

        # Initial refresh
        self._refresh_models()

    def _get_f0_method(self) -> str:
        text = self.cmb_f0.currentText()
        if "dio" in text:
            return "dio"
        elif "crepe" in text:
            return "crepe-tiny"
        elif "parselmouth" in text:
            return "parselmouth"
        return "dio"

    def _refresh_models(self):
        self.cmb_model.clear()
        if not os.path.exists(MODELS_DIR):
            return

        for name in sorted(os.listdir(MODELS_DIR)):
            model_dir = os.path.join(MODELS_DIR, name)
            if not os.path.isdir(model_dir):
                continue
            has_model = any(f.startswith("G_") and f.endswith(".pth") for f in os.listdir(model_dir))
            has_config = os.path.exists(os.path.join(model_dir, "config.json"))
            if has_model and has_config:
                self.cmb_model.addItem(name, model_dir)

        if self.cmb_model.count() == 0:
            self.cmb_model.addItem("No models found — train one first")

    def _start(self):
        if self.cmb_model.count() == 0 or not self.cmb_model.currentData():
            QMessageBox.warning(self, "No Model", "No trained model available.")
            return

        model_dir = self.cmb_model.currentData()
        speaker = self.cmb_model.currentText()

        model_files = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith("G_") and f.endswith(".pth")
        ])
        if not model_files:
            QMessageBox.warning(self, "No Model", "No checkpoint found.")
            return

        model_path = os.path.join(model_dir, model_files[-1])
        config_path = os.path.join(model_dir, "config.json")

        import sys
        svc_bin = os.path.join(os.path.dirname(sys.executable), "svc")
        cmd = [
            svc_bin, "vc",
            "-m", model_path,
            "-c", config_path,
            "-s", speaker,
            "-t", str(int(self.knob_pitch.value)),
            "-fm", self._get_f0_method(),
            "-n", str(self.knob_tone.value),
            "-db", str(int(self.knob_gate.value)),
            "-ch", str(self.knob_response.value),
            "-b", str(self.knob_buffer.value),
            "-cr", str(self.knob_smoothing.value),
            "-d", "mps",
        ]

        if self.chk_auto_f0.isChecked():
            cmd.append("-a")
        else:
            cmd.append("-na")

        if self.chk_passthrough.isChecked():
            cmd.append("-po")

        self.log_viewer.clear_log()
        self.log_viewer.append_line(f"Starting realtime conversion...")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.lbl_status.setText("LIVE")
            self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: #22c55e;")

            from PyQt6.QtCore import QTimer
            self._read_timer = QTimer()
            self._read_timer.setInterval(100)
            self._read_timer.timeout.connect(self._read_output)
            self._read_timer.start()

        except Exception as e:
            self.log_viewer.append_line(f"ERROR: {e}")
            self.lbl_status.setText("Error")
            self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: #ef4444;")

    def _read_output(self):
        if not self._process:
            return

        ret = self._process.poll()
        if ret is not None:
            remaining = self._process.stdout.read()
            if remaining:
                for line in remaining.strip().split("\n"):
                    self.log_viewer.append_line(line)
            self._on_stopped()
            return

        import select
        try:
            readable, _, _ = select.select([self._process.stdout], [], [], 0)
            if readable:
                line = self._process.stdout.readline()
                if line:
                    self.log_viewer.append_line(line.rstrip("\n"))
        except (ValueError, OSError):
            pass

    def _stop(self):
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._on_stopped()

    def _on_stopped(self):
        if hasattr(self, '_read_timer'):
            self._read_timer.stop()
        self._process = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopped")
        self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: #555;")
        self.log_viewer.append_line("Realtime conversion stopped.")
