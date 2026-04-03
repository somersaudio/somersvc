"""Drag-and-drop zone widget for audio files."""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QLabel

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


class AudioDropZone(QLabel):
    files_dropped = pyqtSignal(list)  # list of file path strings

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drag & Drop Audio Files Here\n\n(.wav, .flac, .mp3, .ogg)")
        self.setMinimumHeight(150)
        self._set_default_style()

    def _set_default_style(self):
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #555;
                border-radius: 12px;
                background-color: #2a2a2a;
                color: #888;
                font-size: 14px;
                padding: 20px;
            }
            """
        )

    def _set_hover_style(self):
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #5599ff;
                border-radius: 12px;
                background-color: #1a2a3a;
                color: #5599ff;
                font-size: 14px;
                padding: 20px;
            }
            """
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            # Check if any files have supported extensions
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    self._set_hover_style()
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._set_default_style()

    def dropEvent(self, event: QDropEvent):
        self._set_default_style()
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(path)

        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
