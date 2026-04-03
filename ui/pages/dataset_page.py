"""Dataset management page for adding voice samples."""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
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
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Voice Dataset")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Add audio samples of the voice you want to clone")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        # Speaker name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Speaker Name:"))
        self.txt_speaker = QLineEdit()
        self.txt_speaker.setPlaceholderText("e.g., my_voice")
        self.txt_speaker.textChanged.connect(self._on_speaker_changed)
        name_row.addWidget(self.txt_speaker, 1)
        layout.addLayout(name_row)

        # Summary stats bar
        stats_frame = QFrame()
        stats_frame.setStyleSheet(
            """
            QFrame {
                background-color: #252525;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 12px;
            }
            """
        )
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(16, 10, 16, 10)

        self.lbl_clip_count = QLabel("0")
        self.lbl_clip_count.setStyleSheet("font-size: 24px; font-weight: bold; color: #5599ff;")
        self.lbl_clip_label = QLabel("clips")
        self.lbl_clip_label.setStyleSheet("color: #888; font-size: 12px;")
        clip_col = QVBoxLayout()
        clip_col.setSpacing(0)
        clip_col.addWidget(self.lbl_clip_count, alignment=Qt.AlignmentFlag.AlignCenter)
        clip_col.addWidget(self.lbl_clip_label, alignment=Qt.AlignmentFlag.AlignCenter)
        stats_layout.addLayout(clip_col)

        self._add_stats_divider(stats_layout)

        self.lbl_duration_val = QLabel("0:00")
        self.lbl_duration_val.setStyleSheet("font-size: 24px; font-weight: bold; color: #22c55e;")
        self.lbl_duration_label = QLabel("total duration")
        self.lbl_duration_label.setStyleSheet("color: #888; font-size: 12px;")
        dur_col = QVBoxLayout()
        dur_col.setSpacing(0)
        dur_col.addWidget(self.lbl_duration_val, alignment=Qt.AlignmentFlag.AlignCenter)
        dur_col.addWidget(self.lbl_duration_label, alignment=Qt.AlignmentFlag.AlignCenter)
        stats_layout.addLayout(dur_col)

        self._add_stats_divider(stats_layout)

        self.lbl_status_icon = QLabel("--")
        self.lbl_status_icon.setStyleSheet("font-size: 24px; font-weight: bold; color: #888;")
        self.lbl_status_label = QLabel("status")
        self.lbl_status_label.setStyleSheet("color: #888; font-size: 12px;")
        status_col = QVBoxLayout()
        status_col.setSpacing(0)
        status_col.addWidget(self.lbl_status_icon, alignment=Qt.AlignmentFlag.AlignCenter)
        status_col.addWidget(self.lbl_status_label, alignment=Qt.AlignmentFlag.AlignCenter)
        stats_layout.addLayout(status_col)

        layout.addWidget(stats_frame)

        # Drop zone
        self.drop_zone = AudioDropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_zone)

        # Split progress bar (hidden by default)
        self.split_progress = QProgressBar()
        self.split_progress.setRange(0, 100)
        self.split_progress.setVisible(False)
        self.split_progress.setFixedHeight(20)
        self.lbl_split_status = QLabel("")
        self.lbl_split_status.setStyleSheet("color: #5599ff; font-size: 12px;")
        self.lbl_split_status.setVisible(False)
        layout.addWidget(self.lbl_split_status)
        layout.addWidget(self.split_progress)

        # Browse button
        btn_row = QHBoxLayout()
        self.btn_browse = QPushButton("Browse Files...")
        self.btn_browse.clicked.connect(self._browse_files)
        btn_row.addWidget(self.btn_browse)

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.setObjectName("danger")
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(self.btn_clear)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # File list
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(150)
        layout.addWidget(self.file_list, 1)

        # Validation
        self.lbl_warnings = QLabel("")
        self.lbl_warnings.setWordWrap(True)
        layout.addWidget(self.lbl_warnings)

    def _on_speaker_changed(self, text: str):
        self._refresh_file_list()

    def _on_files_dropped(self, paths: list[str]):
        speaker = self.txt_speaker.text().strip()
        if not speaker:
            QMessageBox.warning(self, "No Speaker Name", "Please enter a speaker name first.")
            return
        self._add_files(paths)

    def _browse_files(self):
        speaker = self.txt_speaker.text().strip()
        if not speaker:
            QMessageBox.warning(self, "No Speaker Name", "Please enter a speaker name first.")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if files:
            self._add_files(files)

    def _add_files(self, paths: list[str]):
        speaker = self.txt_speaker.text().strip()

        # Show splitting progress
        self.split_progress.setVisible(True)
        self.lbl_split_status.setVisible(True)
        self.split_progress.setValue(0)

        total = len(paths)
        all_warnings = []

        for i, p in enumerate(paths):
            name = os.path.basename(p)
            self.lbl_split_status.setText(f"Processing {name}...")
            self.split_progress.setValue(int((i / total) * 100))
            # Process events so the UI updates
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()

            warnings = self.dataset_manager.add_files(speaker, [p])
            all_warnings.extend(warnings)

        self.split_progress.setValue(100)
        self.lbl_split_status.setText("Done!")
        self._refresh_file_list()

        # Hide progress after a moment
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.split_progress.setVisible(False))
        QTimer.singleShot(2000, lambda: self.lbl_split_status.setVisible(False))

        if all_warnings:
            self.lbl_warnings.setText("\n".join(all_warnings))
            self.lbl_warnings.setStyleSheet("color: #5599ff;")

    def _remove_selected(self):
        speaker = self.txt_speaker.text().strip()
        if not speaker:
            return

        items = self.file_list.selectedItems()
        for item in items:
            filename = item.data(Qt.ItemDataRole.UserRole)
            if filename:
                self.dataset_manager.remove_file(speaker, filename)

        self._refresh_file_list()

    def _clear_all(self):
        speaker = self.txt_speaker.text().strip()
        if not speaker:
            return
        reply = QMessageBox.question(
            self, "Clear All", f"Remove all clips for '{speaker}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            import shutil
            speaker_dir = self.dataset_manager.get_speaker_dir(speaker)
            if speaker_dir.exists():
                shutil.rmtree(speaker_dir)
            self._refresh_file_list()

    def _refresh_file_list(self):
        self.file_list.clear()
        speaker = self.txt_speaker.text().strip()
        if not speaker:
            self._update_stats(0, 0.0)
            return

        files = self.dataset_manager.list_files(speaker)
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

        self._update_stats(len(files), total_duration)

    def _update_stats(self, clip_count: int, total_duration: float):
        self.lbl_clip_count.setText(str(clip_count))
        self.lbl_clip_label.setText("clip" if clip_count == 1 else "clips")

        mins = int(total_duration) // 60
        secs = int(total_duration) % 60
        self.lbl_duration_val.setText(f"{mins}:{secs:02d}")

        # Status indicator
        if clip_count == 0:
            self.lbl_status_icon.setText("--")
            self.lbl_status_icon.setStyleSheet("font-size: 24px; font-weight: bold; color: #888;")
            self.lbl_status_label.setText("add samples")
        elif total_duration < 60:
            self.lbl_status_icon.setText("LOW")
            self.lbl_status_icon.setStyleSheet("font-size: 24px; font-weight: bold; color: #ef4444;")
            self.lbl_status_label.setText("need more audio")
        elif total_duration < 300:
            self.lbl_status_icon.setText("OK")
            self.lbl_status_icon.setStyleSheet("font-size: 24px; font-weight: bold; color: #f59e0b;")
            self.lbl_status_label.setText("usable")
        else:
            self.lbl_status_icon.setText("GOOD")
            self.lbl_status_icon.setStyleSheet("font-size: 24px; font-weight: bold; color: #22c55e;")
            self.lbl_status_label.setText("ready to train")

    def _validate(self):
        speaker = self.txt_speaker.text().strip()
        if not speaker:
            self.lbl_warnings.setText("Enter a speaker name first")
            self.lbl_warnings.setStyleSheet("color: #ef4444;")
            return

        warnings = self.dataset_manager.validate(speaker)
        if warnings:
            self.lbl_warnings.setText("\n".join(warnings))
            self.lbl_warnings.setStyleSheet("color: #f59e0b;")
        else:
            self.lbl_warnings.setText("Dataset looks good!")
            self.lbl_warnings.setStyleSheet("color: #22c55e;")

    @staticmethod
    def _add_stats_divider(layout):
        divider = QFrame()
        divider.setFixedWidth(1)
        divider.setFixedHeight(40)
        divider.setStyleSheet("background-color: #444;")
        layout.addWidget(divider)

    def get_speaker_name(self) -> str:
        return self.txt_speaker.text().strip()

    def get_dataset_manager(self) -> DatasetManager:
        return self.dataset_manager
