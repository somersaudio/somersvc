"""Rotary knob (potentiometer) widget for audio-style controls."""

import math

from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QConicalGradient
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QLineEdit


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
        compact: bool = False,
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
        self._compact = compact

        if compact:
            self.setFixedSize(52, 72)
        else:
            self.setFixedSize(80, 110)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if tooltip:
            self.setToolTip(tooltip)

        self._label = label
        self._default = default
        self._val_rect = QRectF()  # clickable area for value text
        self._editing = False
        self._edit_box = None

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
        c = self._compact

        # Draw label above knob
        p.setPen(QColor("#999"))
        p.setFont(QFont("sans-serif", 7 if c else 9))
        label_h = 12 if c else 16
        p.drawText(QRectF(0, 0, w, label_h), Qt.AlignmentFlag.AlignCenter, self._label)

        # Knob center and radius
        cx = w / 2
        cy = 32 if c else 50
        radius = 15 if c else 24

        # Draw track (arc background)
        start_angle = 225  # degrees
        span = 270
        track_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        arc_w = 3 if c else 4
        pen = QPen(QColor("#333"), arc_w)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(track_rect, int((start_angle) * 16), int(-span * 16))

        # Draw active arc
        norm = self._normalized()
        active_span = span * norm
        pen = QPen(QColor("#cccccc"), arc_w)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        if active_span > 0:
            p.drawArc(track_rect, int(start_angle * 16), int(-active_span * 16))

        # Draw knob body
        knob_radius = 11 if c else 18
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor("#3a3a3a")))
        p.drawEllipse(QPointF(cx, cy), knob_radius, knob_radius)

        # Inner circle
        p.setBrush(QBrush(QColor("#2a2a2a")))
        inner_inset = 2 if c else 3
        p.drawEllipse(QPointF(cx, cy), knob_radius - inner_inset, knob_radius - inner_inset)

        # Draw indicator line
        angle_deg = start_angle - active_span
        angle_rad = math.radians(angle_deg)
        inner_r = 4 if c else 6
        outer_r = knob_radius - (3 if c else 4)
        x1 = cx + inner_r * math.cos(angle_rad)
        y1 = cy - inner_r * math.sin(angle_rad)
        x2 = cx + outer_r * math.cos(angle_rad)
        y2 = cy - outer_r * math.sin(angle_rad)

        pen = QPen(QColor("#cccccc"), 1.5 if c else 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # Draw value text below knob
        p.setPen(QColor("#ccc"))
        p.setFont(QFont("sans-serif", 8 if c else 10, QFont.Weight.Bold))
        if self._decimals == 0:
            val_text = f"{int(self._value)}{self._suffix}"
        else:
            val_text = f"{self._value:.{self._decimals}f}{self._suffix}"
        val_y = 53 if c else 82
        val_h = 16 if c else 20
        self._val_rect = QRectF(0, val_y, w, val_h)
        p.drawText(self._val_rect, Qt.AlignmentFlag.AlignCenter, val_text)

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._val_rect.contains(event.position()):
                self._start_editing()
                return
            self._dragging = True
            self._last_y = event.position().y()

    def mouseMoveEvent(self, event):
        if self._dragging:
            dy = self._last_y - event.position().y()
            self._last_y = event.position().y()

            # Sensitivity: full range over ~80px (compact) or ~150px (normal)
            range_val = self._max - self._min
            pixels = 80.0 if self._compact else 150.0
            sensitivity = range_val / pixels
            new_val = self._value + dy * sensitivity

            # Snap to step
            new_val = round(new_val / self._step) * self._step
            self.value = new_val

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def mouseDoubleClickEvent(self, event):
        """Double click knob body to reset to default."""
        if not self._val_rect.contains(event.position()):
            self.value = self._default

    def _start_editing(self):
        if self._editing:
            return
        self._editing = True
        r = self._val_rect
        self._edit_box = QLineEdit(self)
        self._edit_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font_size = 8 if self._compact else 10
        self._edit_box.setStyleSheet(f"""
            QLineEdit {{
                background: #222;
                color: #fff;
                border: 1px solid #cccccc;
                border-radius: 3px;
                font-size: {font_size}px;
                font-weight: bold;
                padding: 0;
            }}
        """)
        self._edit_box.setGeometry(int(r.x()), int(r.y()), int(r.width()), int(r.height()))
        # Show current value without suffix
        if self._decimals == 0:
            self._edit_box.setText(str(int(self._value)))
        else:
            self._edit_box.setText(f"{self._value:.{self._decimals}f}")
        self._edit_box.selectAll()
        self._edit_box.setFocus()
        self._edit_box.show()
        self._edit_box.returnPressed.connect(self._finish_editing)
        self._edit_box.editingFinished.connect(self._finish_editing)

    def _finish_editing(self):
        if not self._editing or not self._edit_box:
            return
        self._editing = False
        try:
            val = float(self._edit_box.text())
            val = round(val / self._step) * self._step
            self.value = val
        except ValueError:
            pass
        self._edit_box.deleteLater()
        self._edit_box = None
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        # Scale wheel steps: each 120 units = 1 notch, move by 3 steps per notch
        steps = (delta / 120.0) * 3 * self._step
        self.value = self._value + steps
