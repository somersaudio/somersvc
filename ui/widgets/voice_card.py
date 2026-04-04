"""Voice card widget — artist-style card for the Voices page."""

import os

from PyQt6.QtCore import (
    Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal,
)

from PyQt6.QtGui import QPixmap, QPainter, QBrush, QPainterPath, QColor, QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QGraphicsBlurEffect,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class _ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        self.clicked.emit()


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
        self.setFixedHeight(80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # Artist image container (image + grade badge overlay)
        from PyQt6.QtWidgets import QWidget as _QW
        img_container = _QW()
        img_container.setFixedSize(80, 80)

        self._is_downloaded = self.metadata.get("source") == "downloaded"
        self._ring_color = "#c0c0c0" if self._is_downloaded else "#B0903D"

        self.lbl_image = QLabel(img_container)
        self.lbl_image.setFixedSize(60, 60)
        self.lbl_image.move(4, 4)
        self.lbl_image.setStyleSheet(
            f"background-color: #333; border-radius: 30px; border: 2px solid {self._ring_color};"
        )
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Grade badge overlaid on bottom-right of image
        self._grade, self._grade_color, tip = self._compute_grade()
        badge_style = (
            f"font-size: 10px; font-weight: bold; "
            f"border-radius: 12px; border: 2px solid {self._grade_color};"
        )

        # Grade label (normal state)
        self.lbl_grade = _ClickableLabel(self._grade, img_container)
        self.lbl_grade.setFixedSize(24, 24)
        self.lbl_grade.move(48, 48)
        self.lbl_grade.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_grade.setToolTip(tip)
        self.lbl_grade.setStyleSheet(
            f"background-color: #1e1e1e; color: {self._grade_color}; " + badge_style
        )

        # Arrow label (upgrade state) — stacked on top, starts invisible
        upgrade_tip = self._build_upgrade_tip()
        self._lbl_arrow = _ClickableLabel("⬆", img_container)
        self._lbl_arrow.setFixedSize(24, 24)
        self._lbl_arrow.move(48, 48)
        self._lbl_arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_arrow.setToolTip(upgrade_tip)
        # Wrap arrow in a container so we can have glow + opacity
        self._arrow_container = QWidget(img_container)
        self._arrow_container.setFixedSize(36, 36)
        self._arrow_container.move(42, 42)
        self._arrow_container.setStyleSheet("background: transparent;")
        self._lbl_arrow.setParent(self._arrow_container)
        self._lbl_arrow.move(6, 6)

        # Glow on the arrow label itself
        glow = QGraphicsDropShadowEffect(self._lbl_arrow)
        glow.setColor(QColor(self._grade_color))
        glow.setBlurRadius(16)
        glow.setOffset(0, 0)
        self._lbl_arrow.setGraphicsEffect(glow)

        # Opacity on the containers for crossfade
        self._grade_opacity = QGraphicsOpacityEffect(self.lbl_grade)
        self._grade_opacity.setOpacity(1.0)
        self.lbl_grade.setGraphicsEffect(self._grade_opacity)

        self._arrow_opacity = QGraphicsOpacityEffect(self._arrow_container)
        self._arrow_opacity.setOpacity(0.0)
        self._arrow_container.setGraphicsEffect(self._arrow_opacity)

        # Fade animations
        self._fade_out_grade = QPropertyAnimation(self._grade_opacity, b"opacity")
        self._fade_out_grade.setDuration(600)
        self._fade_out_grade.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._fade_in_arrow = QPropertyAnimation(self._arrow_opacity, b"opacity")

        self._fade_in_arrow.setDuration(600)
        self._fade_in_arrow.setEasingCurve(QEasingCurve.Type.InOutCubic)

        # Pulse badge between grade and ⬆ if model needs upgrade
        self._badge_showing_arrow = False
        self._pulse_timer = None
        # Hidden button to preserve the upgrade signal connection
        self.btn_upgrade = QPushButton()
        self.btn_upgrade.setVisible(False)
        if self._needs_upgrade():
            self._pulse_timer = QTimer(self)
            self._pulse_timer.timeout.connect(self._toggle_badge)
            self._pulse_timer.start(2400)
            self.lbl_grade.setCursor(Qt.CursorShape.PointingHandCursor)
            self._lbl_arrow.setCursor(Qt.CursorShape.PointingHandCursor)
            self.lbl_grade.clicked.connect(self.btn_upgrade.click)
            self._lbl_arrow.clicked.connect(self.btn_upgrade.click)
            self._arrow_container.raise_()

        # Try to load saved image
        self._load_image()

        layout.addWidget(img_container)

        # Name + creator column
        info_col = QVBoxLayout()
        info_col.setSpacing(2)

        self.lbl_name = QLabel(self.voice_name)
        self.lbl_name.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #e0e0e0; background: transparent;"
        )
        self.lbl_name.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        info_col.addWidget(self.lbl_name)

        creator = self.metadata.get("creator", "")
        self.lbl_creator = QLabel(creator if creator else "")
        self.lbl_creator.setStyleSheet(
            "font-size: 9px; color: #777; background: transparent;"
        )
        info_col.addWidget(self.lbl_creator)

        # Model type in tooltip only
        model_type = self.metadata.get("model_type", "svc").upper()
        self.setToolTip(f"{self.voice_name} — {model_type} model")

        info_wrapper = QVBoxLayout()
        info_wrapper.addStretch()
        info_wrapper.addLayout(info_col)
        info_wrapper.addStretch()
        layout.addLayout(info_wrapper, 1)

    def _build_upgrade_tip(self) -> str:
        """Build a tooltip explaining the upgrade opportunity."""
        epochs = self.metadata.get("epochs", 0)
        clips = self.metadata.get("dataset_clips", 0)
        batch = self.metadata.get("batch_size", 16)
        duration = self.metadata.get("dataset_duration_s", 0)

        lines = ["⬆ Upgrade Available", "", f"Current grade: {self._grade}"]

        if clips > 0 and epochs > 0:
            maturity = (epochs * batch) / clips
            progress = min(int(maturity / 2000 * 100), 99)
            lines.append(f"Training progress: {progress}%")

            if maturity < 300:
                lines.append("Status: Undertrained — significant improvement possible")
            elif maturity < 800:
                lines.append("Status: Partially trained — more training will help")
            else:
                lines.append("Status: Well trained — a bit more to reach full quality")

        # What grade they could reach
        dur_score = 3 if duration >= 600 else 2 if duration >= 300 else 1 if duration >= 120 else 0
        potential_total = dur_score + 3  # assume full training convergence
        grade_map = {6: "S", 5: "A+", 4: "A", 3: "B+", 2: "B", 1: "C", 0: "D"}
        potential = grade_map.get(potential_total, "?")

        if potential != self._grade:
            lines.append(f"Potential grade: {potential}")

        lines.append("")
        lines.append("Click to upgrade this model")

        return "\n".join(lines)

    def _needs_upgrade(self) -> bool:
        epochs = self.metadata.get("epochs", 0)
        clips = self.metadata.get("dataset_clips", 0)
        batch = self.metadata.get("batch_size", 16)
        if epochs == 0 or clips == 0:
            return False
        # Check if training dataset actually exists
        dataset_dir = os.path.join("data", "datasets", self.voice_name)
        if not os.path.isdir(dataset_dir) or not os.listdir(dataset_dir):
            return False
        maturity = (epochs * batch) / clips
        return maturity < 2000

    def _toggle_badge(self):
        """Crossfade between grade letter and ⬆ arrow."""
        self._badge_showing_arrow = not self._badge_showing_arrow
        if self._badge_showing_arrow:
            self._fade_out_grade.setStartValue(1.0)
            self._fade_out_grade.setEndValue(0.0)
            self._fade_in_arrow.setStartValue(0.0)
            self._fade_in_arrow.setEndValue(1.0)
        else:
            self._fade_out_grade.setStartValue(0.0)
            self._fade_out_grade.setEndValue(1.0)
            self._fade_in_arrow.setStartValue(1.0)
            self._fade_in_arrow.setEndValue(0.0)
        self._fade_out_grade.start()
        self._fade_in_arrow.start()

    def _compute_grade(self) -> tuple[str, str, str]:
        """Compute quality grade based on dataset duration and total data processed.
        Returns (grade, color, tooltip)."""
        epochs = self.metadata.get("epochs", 0)
        duration = self.metadata.get("dataset_duration_s", 0)
        clips = self.metadata.get("dataset_clips", 0)
        batch = self.metadata.get("batch_size", 16)

        # For downloaded models without full training data, use simplified grading
        if clips == 0 and (self.metadata.get("sample_rate") or self.metadata.get("rvc_version")):
            from services.model_inspector import compute_downloaded_grade
            return compute_downloaded_grade(self.metadata)

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
                self.lbl_image.setPixmap(self._make_circular(pixmap, 56))
                return

        # Default: show initials
        initials = "".join([w[0].upper() for w in self.voice_name.split("_")[:2]])
        self.lbl_image.setText(initials)
        self.lbl_image.setStyleSheet(
            f"background-color: #2563eb; border-radius: 30px; border: 2px solid {self._ring_color}; "
            "color: white; font-size: 20px; font-weight: bold;"
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
        border_color = "#ffffff" if selected else self._ring_color
        # Replace any previous border color
        style = self.lbl_image.styleSheet()
        import re as _re
        style = _re.sub(r"border: \dpx solid #[0-9a-fA-F]+", f"border: 2px solid {border_color}", style)
        self.lbl_image.setStyleSheet(style)

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
                    background-color: #141414;
                    border: 1px solid #2a2a2a;
                    border-radius: 10px;
                }
                VoiceCard:hover {
                    background-color: #1a1a1a;
                    border-color: #383838;
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
