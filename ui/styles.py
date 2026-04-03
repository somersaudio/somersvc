"""Dark theme stylesheet for the application."""

DARK_THEME = """
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: -apple-system, 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 13px;
}

QLabel {
    color: #e0e0e0;
    background: transparent;
}

QLabel#title {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
    padding: 8px 0;
}

QLabel#subtitle {
    font-size: 14px;
    color: #999;
    padding: 4px 0;
}

QPushButton {
    background-color: #3a3a3a;
    color: #e0e0e0;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #666;
}

QPushButton:pressed {
    background-color: #2a2a2a;
}

QPushButton:disabled {
    background-color: #2a2a2a;
    color: #555;
    border-color: #333;
}

QPushButton#primary {
    background-color: #2563eb;
    color: white;
    border: none;
    font-weight: bold;
}

QPushButton#primary:hover {
    background-color: #3b82f6;
}

QPushButton#primary:disabled {
    background-color: #1e3a5f;
    color: #666;
}

QPushButton#danger {
    background-color: #dc2626;
    color: white;
    border: none;
}

QPushButton#danger:hover {
    background-color: #ef4444;
}

QLineEdit {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}

QLineEdit:focus {
    border-color: #2563eb;
}

QComboBox {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox QAbstractItemView {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #444;
    selection-background-color: #2563eb;
}

QListWidget {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}

QListWidget::item {
    padding: 8px 12px;
    border-radius: 4px;
}

QListWidget::item:selected {
    background-color: #2563eb;
    color: white;
}

QListWidget::item:hover {
    background-color: #333;
}

QProgressBar {
    background-color: #2a2a2a;
    border: 1px solid #444;
    border-radius: 6px;
    height: 20px;
    text-align: center;
    color: #e0e0e0;
}

QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 5px;
}

QSlider::groove:horizontal {
    background: #333;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #2563eb;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::sub-page:horizontal {
    background: #2563eb;
    border-radius: 3px;
}

QSplitter::handle {
    background-color: #333;
}

QGroupBox {
    border: 1px solid #444;
    border-radius: 8px;
    margin-top: 16px;
    padding-top: 16px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

QScrollBar:vertical {
    background: #1e1e1e;
    width: 8px;
    border: none;
}

QScrollBar::handle:vertical {
    background: #555;
    min-height: 30px;
    border-radius: 4px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""
