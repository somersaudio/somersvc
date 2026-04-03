"""Create page — set up a voice for training."""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from services.dataset_manager import DatasetManager
from ui.widgets.audio_drop_zone import AudioDropZone

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS_DIR = os.path.join(APP_DIR, "data", "datasets")


class DatasetPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_manager = DatasetManager(DATASETS_DIR)
        self._voice_name = ""
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Create a Model")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Add voice samples and train a new model")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        # Voice name input
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Voice Name:"))
        self.txt_voice = QLineEdit()
        self.txt_voice.setPlaceholderText("e.g., Julia Wolf")
        self.txt_voice.textChanged.connect(self._on_name_changed)
        name_row.addWidget(self.txt_voice, 1)
        layout.addLayout(name_row)

        # Drop zone
        self.drop_zone = AudioDropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_zone)

        # Split progress
        self.split_progress = QProgressBar()
        self.split_progress.setRange(0, 100)
        self.split_progress.setVisible(False)
        self.split_progress.setFixedHeight(20)
        self.lbl_split_status = QLabel("")
        self.lbl_split_status.setStyleSheet("color: #5599ff; font-size: 12px;")
        self.lbl_split_status.setVisible(False)
        layout.addWidget(self.lbl_split_status)
        layout.addWidget(self.split_progress)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_browse = QPushButton("Browse Files...")
        self.btn_browse.clicked.connect(self._browse_files)
        btn_row.addWidget(self.btn_browse)

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.setObjectName("danger")
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.btn_remove)

        btn_row.addStretch()

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #888; font-size: 13px;")
        btn_row.addWidget(self.lbl_info)
        layout.addLayout(btn_row)

        # File list
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(150)
        layout.addWidget(self.file_list, 1)

    def _on_name_changed(self, text: str):
        self._voice_name = text.strip()
        self._refresh_file_list()

    def _on_files_dropped(self, paths: list[str]):
        if not self._voice_name:
            QMessageBox.warning(self, "No Name", "Enter a voice name first.")
            return
        self._add_files(paths)

    def _browse_files(self):
        if not self._voice_name:
            QMessageBox.warning(self, "No Name", "Enter a voice name first.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if files:
            self._add_files(files)

    def _add_files(self, paths: list[str]):
        self.dataset_manager.create_speaker(self._voice_name)
        self.split_progress.setVisible(True)
        self.lbl_split_status.setVisible(True)
        self.split_progress.setValue(0)

        total = len(paths)
        for i, p in enumerate(paths):
            name = os.path.basename(p)
            self.lbl_split_status.setText(f"Processing {name}...")
            self.split_progress.setValue(int((i / total) * 100))
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            self.dataset_manager.add_files(self._voice_name, [p])

        self.split_progress.setValue(100)
        self.lbl_split_status.setText("Done!")
        self._refresh_file_list()

        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.split_progress.setVisible(False))
        QTimer.singleShot(2000, lambda: self.lbl_split_status.setVisible(False))

    def _remove_selected(self):
        if not self._voice_name:
            return
        for item in self.file_list.selectedItems():
            filename = item.data(Qt.ItemDataRole.UserRole)
            if filename:
                self.dataset_manager.remove_file(self._voice_name, filename)
        self._refresh_file_list()

    def _refresh_file_list(self):
        self.file_list.clear()
        if not self._voice_name:
            self.lbl_info.setText("")
            return

        files = self.dataset_manager.list_files(self._voice_name)
        total_duration = 0.0

        for f in files:
            dur = f.get("duration")
            duration_str = f"{dur:.1f}s" if dur else "?"
            if dur:
                total_duration += dur
            text = f"  {f['name']}    {duration_str}    {f['size_mb']:.1f} MB"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, f["name"])
            self.file_list.addItem(item)

        mins = int(total_duration) // 60
        secs = int(total_duration) % 60
        self.lbl_info.setText(f"{len(files)} clips  |  {mins}:{secs:02d}")

    def get_speaker_name(self) -> str:
        return self._voice_name

    def get_dataset_manager(self) -> DatasetManager:
        return self.dataset_manager
