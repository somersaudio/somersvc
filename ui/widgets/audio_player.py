"""Audio playback widget with play/stop and position slider."""

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)


class AudioPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        # UI
        self.btn_play = QPushButton("Play")
        self.btn_stop = QPushButton("Stop")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.lbl_time = QLabel("0:00 / 0:00")

        self.btn_play.setFixedWidth(70)
        self.btn_stop.setFixedWidth(70)
        self.lbl_time.setFixedWidth(100)
        self.slider.setRange(0, 0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.lbl_time)

        # Connections
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop.clicked.connect(self._stop)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.player.playbackStateChanged.connect(self._on_state_changed)

        self.setEnabled(False)

    def load(self, file_path: str):
        self.player.setSource(QUrl.fromLocalFile(file_path))
        self.setEnabled(True)
        self.btn_play.setText("Play")

    def _toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _stop(self):
        self.player.stop()

    def _on_position_changed(self, position: int):
        self.slider.setValue(position)
        self.lbl_time.setText(
            f"{self._fmt(position)} / {self._fmt(self.player.duration())}"
        )

    def _on_duration_changed(self, duration: int):
        self.slider.setRange(0, duration)

    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("Pause")
        else:
            self.btn_play.setText("Play")

    @staticmethod
    def _fmt(ms: int) -> str:
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"
