"""SVC Voice Converter - GUI application for so-vits-svc-fork."""

import sys
import os

# Ensure the app directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ui.main_window import MainWindow
from ui.styles import DARK_THEME


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SVC Voice Converter")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_THEME)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
