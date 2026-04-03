"""Rotary knob (potentiometer) widget for audio-style controls."""

import math

from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QConicalGradient
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy


class Knob(QWidget):
    """A rotary knob widget that looks like a studio potentiometer."""
    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str = "",
        tooltip: str = "",
        min_val: float = 0.0,
        max_val: float = 1.0,
        default: float = 0.5,
        step: float = 0.01,
        suffix: str = "",
        decimals: int = 2,
        parent=None,
    ):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._value = default
        self._step = step
        self._suffix = suffix
        self._decimals = decimals
        self._dragging = False
        self._last_y = 0

        self.setFixedSize(80, 110)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if tooltip:
            self.setToolTip(tooltip)

        self._label = label

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, v: float):
        v = max(self._min, min(self._max, v))
        if v != self._value:
            self._value = v
            self.valueChanged.emit(v)
            self.update()

    def _normalized(self) -> float:
        """Return value as 0.0-1.0 range."""
        if self._max == self._min:
            return 0.0
        return (self._value - self._min) / (self._max - self._min)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()

        # Draw label above knob
        p.setPen(QColor("#999"))
        p.setFont(QFont("sans-serif", 9))
        p.drawText(QRectF(0, 0, w, 16), Qt.AlignmentFlag.AlignCenter, self._label)

        # Knob center and radius
        cx = w / 2
        cy = 50
        radius = 24

        # Draw track (arc background)
        start_angle = 225  # degrees
        span = 270
        track_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        pen = QPen(QColor("#333"), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(track_rect, int((start_angle) * 16), int(-span * 16))

        # Draw active arc
        norm = self._normalized()
        active_span = span * norm
        pen = QPen(QColor("#5599ff"), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        if active_span > 0:
            p.drawArc(track_rect, int(start_angle * 16), int(-active_span * 16))

        # Draw knob body
        knob_radius = 18
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor("#3a3a3a")))
        p.drawEllipse(QPointF(cx, cy), knob_radius, knob_radius)

        # Inner circle
        p.setBrush(QBrush(QColor("#2a2a2a")))
        p.drawEllipse(QPointF(cx, cy), knob_radius - 3, knob_radius - 3)

        # Draw indicator line
        angle_deg = start_angle - active_span
        angle_rad = math.radians(angle_deg)
        inner_r = 6
        outer_r = knob_radius - 4
        x1 = cx + inner_r * math.cos(angle_rad)
        y1 = cy - inner_r * math.sin(angle_rad)
        x2 = cx + outer_r * math.cos(angle_rad)
        y2 = cy - outer_r * math.sin(angle_rad)

        pen = QPen(QColor("#5599ff"), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # Draw value text below knob
        p.setPen(QColor("#ccc"))
        p.setFont(QFont("sans-serif", 10, QFont.Weight.Bold))
        if self._decimals == 0:
            val_text = f"{int(self._value)}{self._suffix}"
        else:
            val_text = f"{self._value:.{self._decimals}f}{self._suffix}"
        p.drawText(QRectF(0, 82, w, 20), Qt.AlignmentFlag.AlignCenter, val_text)

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_y = event.position().y()

    def mouseMoveEvent(self, event):
        if self._dragging:
            dy = self._last_y - event.position().y()
            self._last_y = event.position().y()

            # Sensitivity: full range over ~150 pixels of drag
            range_val = self._max - self._min
            sensitivity = range_val / 150.0
            new_val = self._value + dy * sensitivity

            # Snap to step
            new_val = round(new_val / self._step) * self._step
            self.value = new_val

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def mouseDoubleClickEvent(self, event):
        """Double click to reset to default middle value."""
        # Reset is not implemented — could store default and restore
        pass

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.value = self._value + self._step
        elif delta < 0:
            self.value = self._value - self._step
