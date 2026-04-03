"""Log viewer widget for streaming output."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPlainTextEdit


class LogViewer(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 8px;
            }
            """
        )

    def append_line(self, text: str):
        self.appendPlainText(text)
        # Auto-scroll to bottom
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        self.clear()
