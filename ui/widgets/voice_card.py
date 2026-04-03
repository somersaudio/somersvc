"""Voice card widget — artist-style card for the Voices page."""

import os

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QBrush, QPainterPath, QColor, QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class VoiceCard(QWidget):
    """A card displaying voice info with artist image, name, creator, and epoch count."""
    clicked = pyqtSignal(str)  # voice name

    def __init__(self, voice_name: str, metadata: dict = None, parent=None):
        super().__init__(parent)
        self.voice_name = voice_name
        self.metadata = metadata or {}
        self._selected = False
        self._init_ui()

    def _init_ui(self):
        self.setFixedHeight(100)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(14)

        # Artist image (circular)
        self.lbl_image = QLabel()
        self.lbl_image.setFixedSize(76, 76)
        self.lbl_image.setStyleSheet(
            "background-color: #333; border-radius: 38px; border: 2px solid #444;"
        )
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Try to load saved image
        self._load_image()

        layout.addWidget(self.lbl_image)

        # Name + creator column
        info_col = QVBoxLayout()
        info_col.setSpacing(2)

        self.lbl_name = QLabel(self.voice_name)
        self.lbl_name.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #e0e0e0; background: transparent;"
        )
        info_col.addWidget(self.lbl_name)

        creator = self.metadata.get("creator", "")
        self.lbl_creator = QLabel(creator if creator else "")
        self.lbl_creator.setStyleSheet(
            "font-size: 11px; color: #777; background: transparent;"
        )
        info_col.addWidget(self.lbl_creator)

        layout.addLayout(info_col, 1)

        # Upgrade button (only visible if model needs more training)
        self.btn_upgrade = QPushButton("Upgrade")
        self.btn_upgrade.setStyleSheet(
            """
            QPushButton {
                background-color: #a855f7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c084fc; }
            """
        )
        self.btn_upgrade.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_upgrade.setFixedHeight(26)
        self.btn_upgrade.setVisible(self._needs_upgrade())
        layout.addWidget(self.btn_upgrade)

        # Quality grade badge
        grade, grade_color, tip = self._compute_grade()
        grade_col = QVBoxLayout()
        grade_col.setSpacing(0)

        self.lbl_grade = QLabel(grade)
        self.lbl_grade.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {grade_color}; background: transparent;"
        )
        self.lbl_grade.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_grade.setToolTip(tip)
        grade_col.addWidget(self.lbl_grade)

        # Small detail underneath
        epochs = self.metadata.get("epochs", 0)
        batch = self.metadata.get("batch_size", 16)
        if epochs > 0:
            if epochs >= 1000:
                detail = f"{epochs / 1000:.1f}Ke"
            else:
                detail = f"{epochs}e"
        else:
            detail = ""
        self.lbl_detail = QLabel(detail)
        self.lbl_detail.setStyleSheet(
            "font-size: 9px; color: #666; background: transparent;"
        )
        self.lbl_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grade_col.addWidget(self.lbl_detail)

        layout.addLayout(grade_col)

    def _needs_upgrade(self) -> bool:
        epochs = self.metadata.get("epochs", 0)
        clips = self.metadata.get("dataset_clips", 0)
        batch = self.metadata.get("batch_size", 16)
        if epochs == 0 or clips == 0:
            return False
        maturity = (epochs * batch) / clips
        return maturity < 2000

    def _compute_grade(self) -> tuple[str, str, str]:
        """Compute quality grade based on dataset duration and total data processed.
        Returns (grade, color, tooltip)."""
        epochs = self.metadata.get("epochs", 0)
        duration = self.metadata.get("dataset_duration_s", 0)
        clips = self.metadata.get("dataset_clips", 0)
        batch = self.metadata.get("batch_size", 16)

        if epochs == 0 or clips == 0:
            return ("--", "#555", "No training data yet")

        data_passes = epochs * batch
        dur_min = int(duration) // 60
        dur_sec = int(duration) % 60

        # Duration score
        if duration >= 600:
            dur_score = 3
            dur_tip = "Audio: Excellent (10+ min)"
        elif duration >= 300:
            dur_score = 2
            dur_tip = "Audio: Good (5-10 min)"
        elif duration >= 120:
            dur_score = 1
            dur_tip = f"Audio: Fair ({dur_min}:{dur_sec:02d})"
        else:
            dur_score = 0
            dur_tip = f"Audio: Low ({dur_min}:{dur_sec:02d})"

        # Training score: based on data passes per clip (maturity)
        maturity = data_passes / clips
        if maturity >= 2000:
            train_score = 3
            train_tip = "Training: Fully converged"
        elif maturity >= 800:
            train_score = 2
            train_tip = "Training: Well trained"
        elif maturity >= 300:
            train_score = 1
            train_tip = "Training: Partially trained"
        else:
            train_score = 0
            train_tip = "Training: Undertrained"

        total = dur_score + train_score

        grades = {
            6: ("S", "#a855f7"),
            5: ("A+", "#22c55e"),
            4: ("A", "#22c55e"),
            3: ("B+", "#5599ff"),
            2: ("B", "#5599ff"),
            1: ("C", "#f59e0b"),
            0: ("D", "#ef4444"),
        }
        grade, color = grades.get(total, ("?", "#888"))

        # Build improvement tips
        tips = [f"Quality: {grade}", "", dur_tip, train_tip, ""]

        if dur_score < 3:
            needed = {0: "2+ minutes", 1: "5+ minutes", 2: "10+ minutes"}
            tips.append(f"Add more audio samples ({needed.get(dur_score, '')} total)")

        if train_score < 3:
            tips.append("Train for more epochs to improve quality")

        if dur_score >= 3 and train_score >= 3:
            tips.append("Maximum quality reached!")

        tip = "\n".join(tips)
        return (grade, color, tip)

    def _load_image(self):
        """Load voice image from data/models/{voice}/image.png"""
        img_paths = [
            os.path.join("data", "models", self.voice_name, "image.png"),
            os.path.join("data", "models", self.voice_name, "image.jpg"),
        ]
        for img_path in img_paths:
            if os.path.exists(img_path):
                pixmap = QPixmap(img_path)
                self.lbl_image.setPixmap(self._make_circular(pixmap, 72))
                return

        # Default: show initials
        initials = "".join([w[0].upper() for w in self.voice_name.split("_")[:2]])
        self.lbl_image.setText(initials)
        self.lbl_image.setStyleSheet(
            "background-color: #2563eb; border-radius: 38px; "
            "color: white; font-size: 22px; font-weight: bold;"
        )

    @staticmethod
    def _make_circular(pixmap: QPixmap, size: int) -> QPixmap:
        """Crop a pixmap into a circle."""
        scaled = pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                Qt.TransformationMode.SmoothTransformation)
        # Center crop
        x = (scaled.width() - size) // 2
        y = (scaled.height() - size) // 2
        cropped = scaled.copy(x, y, size, size)

        result = QPixmap(size, size)
        result.fill(Qt.GlobalColor.transparent)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, cropped)
        painter.end()
        return result

    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_style()

    def _update_style(self):
        if self._selected:
            self.setStyleSheet(
                """
                VoiceCard {
                    background-color: #1e3a5f;
                    border: 1px solid #2563eb;
                    border-radius: 10px;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                VoiceCard {
                    background-color: #1e1e1e;
                    border: 1px solid #333;
                    border-radius: 10px;
                }
                VoiceCard:hover {
                    background-color: #252525;
                    border-color: #444;
                }
                """
            )

    def mousePressEvent(self, event):
        self.clicked.emit(self.voice_name)

    def mouseDoubleClickEvent(self, event):
        """Double-click to set artist image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Artist Image", "",
            "Images (*.png *.jpg *.jpeg *.webp);;All Files (*)",
        )
        if path:
            import shutil
            model_dir = os.path.join("data", "models", self.voice_name)
            os.makedirs(model_dir, exist_ok=True)
            ext = os.path.splitext(path)[1]
            dest = os.path.join(model_dir, f"image{ext}")
            shutil.copy2(path, dest)
            self._load_image()
