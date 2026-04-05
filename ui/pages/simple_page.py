"""Simple one-screen interface for SomerSVC."""

import os

from PyQt6.QtCore import Qt, QRectF, QSize, QThread, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from services.rvc_inference_runner import detect_model_type, _get_rvc_pth_files
from ui.widgets.log_viewer import LogViewer
from workers.inference_worker import InferenceWorker

from services.paths import APP_DIR, MODELS_DIR, DATASETS_DIR, OUTPUT_DIR, CACHE_DIR

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _hz_to_note(hz):
    import math
    if hz <= 0:
        return "?"
    midi = 69 + 12 * math.log2(hz / 440.0)
    midi = round(midi)
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"


_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B"}

def _note_to_hz(note):
    import re, math
    m = re.match(r"([A-G][#b]?)(\d+)", note)
    if not m:
        return 0
    name = m.group(1)
    name = _FLAT_TO_SHARP.get(name, name)
    if name not in NOTE_NAMES:
        return 0
    idx = NOTE_NAMES.index(name)
    midi = (int(m.group(2)) + 1) * 12 + idx
    return 440.0 * (2 ** ((midi - 69) / 12))


class _PitchWorker(QThread):
    result = pyqtSignal(str, float)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            import numpy as np, librosa
            y, sr = librosa.load(self.path, sr=22050, duration=120)
            f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                          fmax=librosa.note_to_hz("C6"), sr=sr)
            voiced_f0 = f0[voiced & ~np.isnan(f0)]
            if len(voiced_f0) == 0:
                self.result.emit("No vocals detected", 0)
                return
            low = _hz_to_note(float(np.percentile(voiced_f0, 5)))
            high = _hz_to_note(float(np.percentile(voiced_f0, 95)))
            median = float(np.median(voiced_f0))
            self.result.emit(f"{low} – {high}  ·  Center: {_hz_to_note(median)}", median)
        except Exception as e:
            self.result.emit(f"Pitch detection failed", 0)


class _KeyDetectWorker(QThread):
    result = pyqtSignal(str, float)

    def __init__(self, clips):
        super().__init__()
        self.clips = clips

    def run(self):
        try:
            import numpy as np, librosa
            all_f0 = []
            for p in self.clips:
                try:
                    y, sr = librosa.load(p, sr=22050)
                    f0, v, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                             fmax=librosa.note_to_hz("C6"), sr=sr)
                    all_f0.extend(f0[v & ~np.isnan(f0)].tolist())
                except Exception:
                    pass
            if all_f0:
                median = float(np.median(all_f0))
                self.result.emit(_hz_to_note(median), median)
            else:
                self.result.emit("", 0)
        except Exception:
            self.result.emit("", 0)


class _NormalizeWorker(QThread):
    """Background worker to normalize audio without blocking the UI."""
    finished = pyqtSignal(str)  # normalized path

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        import numpy as np
        import soundfile as sf
        try:
            audio, sr = sf.read(self.path)
            peak = np.max(np.abs(audio))
            if peak > 0 and abs(peak - 0.75) > 0.05:
                audio = audio * (0.75 / peak)
                norm_dir = os.path.join(OUTPUT_DIR, ".normalized")
                os.makedirs(norm_dir, exist_ok=True)
                norm_path = os.path.join(norm_dir, os.path.basename(self.path))
                if not norm_path.lower().endswith(".wav"):
                    norm_path = os.path.splitext(norm_path)[0] + ".wav"
                sf.write(norm_path, audio, sr)
                self.finished.emit(norm_path)
                return
        except Exception as e:
            print(f"Normalization error (non-fatal): {e}")
        self.finished.emit(self.path)


class _WaveformAnalyzer(QThread):
    """Background worker to load waveform data and compute section splits."""
    finished = pyqtSignal(object, list, list, list)  # samples, sections, transposes, median_hz_list

    def __init__(self, audio_path, model_center_hz=0):
        super().__init__()
        self.audio_path = audio_path
        self.model_center_hz = model_center_hz

    def run(self):
        try:
            import numpy as np
            import soundfile as sf

            audio, sr = sf.read(self.audio_path)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            # Downsample to ~2000 points for drawing
            n_points = 2000
            chunk = max(1, len(audio) // n_points)
            samples = []
            for i in range(0, len(audio), chunk):
                block = audio[i:i + chunk]
                samples.append(float(np.max(np.abs(block))))
            duration = len(audio) / sr

            from services.section_splitter import find_section_splits
            sections = find_section_splits(self.audio_path, max_sections=20)

            # Compute transposes if we have model center
            transposes = []
            median_hz_list = []
            if self.model_center_hz > 0 and len(sections) > 1:
                import tempfile
                from services.section_splitter import (
                    split_audio_file, analyze_section_pitches,
                    calculate_section_transposes,
                )
                tmp_dir = tempfile.mkdtemp(prefix="svc_wf_")
                try:
                    section_paths = split_audio_file(self.audio_path, sections, tmp_dir)
                    pitched = analyze_section_pitches(section_paths)
                    pitched = calculate_section_transposes(pitched, self.model_center_hz)
                    transposes = [s.get("transpose", 0) for s in pitched]
                    median_hz_list = [s.get("median_hz", 0) for s in pitched]
                finally:
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                transposes = [0] * len(sections)
                median_hz_list = [0] * len(sections)

            # Normalize section times to 0-1
            norm_sections = [(s / duration, e / duration) for s, e in sections]

            self.finished.emit(samples, norm_sections, transposes, median_hz_list)
        except Exception as e:
            print(f"Waveform analysis error: {e}")
            self.finished.emit([], [], [], [])


class _WaveformSamplesOnly(QThread):
    """Lightweight worker that only loads waveform samples — no section splitting."""
    finished = pyqtSignal(list)

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            import numpy as np
            import soundfile as sf
            audio, sr = sf.read(self.audio_path)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            n_points = 2000
            chunk = max(1, len(audio) // n_points)
            samples = []
            for i in range(0, len(audio), chunk):
                block = audio[i:i + chunk]
                samples.append(float(np.max(np.abs(block))))
            self.finished.emit(samples)
        except Exception as e:
            print(f"Waveform samples load error: {e}")
            self.finished.emit([])


class _WaveformWidget(QWidget):
    """Waveform with section splits, transpose coloring, and built-in audio playback."""

    sections_changed = pyqtSignal()  # emitted when user drags a split marker
    interacted = pyqtSignal()        # emitted when user clicks/scrubs this waveform
    MARGIN_X = 12
    MARGIN_Y = 6
    WAVE_H = 50
    TIME_H = 14
    PLAY_SIZE = 20
    _time_font = QFont("Manrope", 9)
    SPLIT_HIT_PX = 8  # pixels tolerance for grabbing a split line

    _label_font = QFont("Manrope", 8)

    def __init__(self, parent=None, readonly=False):
        super().__init__(parent)
        self._readonly = readonly
        self._samples = []
        self._label = ""
        self._sections = []
        self._transposes = []
        self._section_info = []     # list of dicts with pitch info per section
        self._median_hz = []        # median pitch per section (for distance calc)
        self._model_center_hz = 0
        self._converted = False     # True after conversion completes
        self._active_section = -1
        self._progress = 0.0
        self._playhead = 0.0
        self._dragging = False
        self._drag_split_idx = -1
        self._hover_split_idx = -1
        self._wave_cache = None     # pre-rendered waveform pixmap
        self.setFixedHeight(self.WAVE_H + self.TIME_H + self.MARGIN_Y * 2)
        self.setVisible(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

        # Audio player
        from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
        self._player = QMediaPlayer()
        self._audio_out = QAudioOutput()
        self._player.setAudioOutput(self._audio_out)
        self._duration_ms = 0
        self._has_source = False
        self._is_playing = False
        self._player.durationChanged.connect(self._on_duration)
        self._player.positionChanged.connect(self._on_position)
        self._player.playbackStateChanged.connect(self._on_playback_state)

    def _on_playback_state(self, state):
        from PyQt6.QtMultimedia import QMediaPlayer
        self._is_playing = state == QMediaPlayer.PlaybackState.PlayingState

    def load(self, path):
        from PyQt6.QtCore import QUrl
        self._player.setSource(QUrl.fromLocalFile(path))
        self._has_source = True
        self._playhead = 0.0
        self.update()

    def hide_and_stop(self):
        """Hide the widget and stop any playing audio."""
        if self._is_playing:
            self._player.stop()
        self._is_playing = False
        self._playhead = 0.0
        self.setVisible(False)

    def set_data(self, samples, sections, transposes, median_hz=None, model_center_hz=0):
        self._samples = samples
        self._sections = sections
        self._transposes = transposes
        self._median_hz = median_hz or [0] * len(sections)
        self._model_center_hz = model_center_hz
        self._section_info = []
        self._converted = False
        self._active_section = -1
        self._progress = 0.0
        self._wave_cache = None
        self.setVisible(len(samples) > 0)
        self.update()

    def set_section_info(self, info_list):
        """Set detailed info per section after conversion: [{transpose, from_note, to_note, distance}, ...]"""
        self._section_info = info_list
        self._converted = True
        self.update()

    def set_active_section(self, idx):
        self._active_section = idx
        self.update()

    def set_progress(self, progress):
        self._progress = progress
        self.update()

    def clear(self):
        if self._is_playing:
            self._player.stop()
        self._is_playing = False
        self._has_source = False
        self._samples = []
        self._sections = []
        self._transposes = []
        self._section_info = []
        self._converted = False
        self._active_section = -1
        self._progress = 0.0
        self._playhead = 0.0
        self._duration_ms = 0
        self.setVisible(False)
        self.update()

    def _on_duration(self, ms):
        self._duration_ms = ms

    def _on_position(self, ms):
        if self._duration_ms > 0 and not self._dragging and self.isVisible():
            new_pos = ms / self._duration_ms
            if abs(new_pos - self._playhead) > 0.005:
                self._playhead = new_pos
                self.update()

    def _x_to_frac(self, x):
        draw_w = self.width() - self.MARGIN_X * 2
        return max(0.0, min(1.0, (x - self.MARGIN_X) / draw_w))

    def _frac_to_x(self, frac):
        draw_w = self.width() - self.MARGIN_X * 2
        return self.MARGIN_X + frac * draw_w

    def _find_nearest_split(self, x):
        """Return index of split boundary near x (skip first=0 and last), or -1."""
        for i, (start, _) in enumerate(self._sections):
            if i == 0:
                continue
            sx = self._frac_to_x(start)
            if abs(x - sx) < self.SPLIT_HIT_PX:
                return i
        return -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._samples:
            self.interacted.emit()
            from PyQt6.QtMultimedia import QMediaPlayer
            ex, ey = event.position().x(), event.position().y()
            frac = self._x_to_frac(ex)

            # Play button area
            if ex < self.MARGIN_X + self.PLAY_SIZE + 4 and ey > self.WAVE_H + self.MARGIN_Y:
                from PyQt6.sip import isdeleted
                if isdeleted(self._player):
                    return
                if self._is_playing:
                    self._player.pause()
                else:
                    # Stop any other waveform's player via parent
                    parent = self.parent()
                    while parent:
                        if hasattr(parent, '_waveform') and hasattr(parent, '_waveform_output'):
                            other = parent._waveform_output if self is parent._waveform else parent._waveform
                            if other._is_playing and not isdeleted(other._player):
                                other._player.pause()
                                other.update()
                            break
                        parent = parent.parent()
                    self._player.play()
                self.update()
                return

            # Check if grabbing a split marker (not in readonly mode)
            if not self._readonly and ey <= self.WAVE_H + self.MARGIN_Y:
                split_idx = self._find_nearest_split(ex)
                if split_idx >= 0:
                    self._drag_split_idx = split_idx
                    self._dragging = True
                    return

            # Otherwise scrub playhead
            self._drag_split_idx = -1
            self._dragging = True
            self._playhead = frac
            if self._duration_ms > 0:
                self._player.setPosition(int(frac * self._duration_ms))
            self.update()

    def mouseMoveEvent(self, event):
        ex = event.position().x()
        ey = event.position().y()

        if self._dragging and self._drag_split_idx >= 0:
            # Dragging a split marker
            frac = self._x_to_frac(ex)
            idx = self._drag_split_idx

            # Clamp: can't go past neighboring splits (with min section width)
            min_gap = 0.02  # ~2% of track minimum section
            prev_end = self._sections[idx - 1][0] + min_gap if idx > 0 else min_gap
            next_start = self._sections[idx][1] - min_gap if idx < len(self._sections) else 1.0 - min_gap
            frac = max(prev_end, min(next_start, frac))

            # Update sections: this split is the end of section[idx-1] and start of section[idx]
            s_prev = self._sections[idx - 1]
            s_curr = self._sections[idx]
            self._sections[idx - 1] = (s_prev[0], frac)
            self._sections[idx] = (frac, s_curr[1])
            self.update()
            return

        if self._dragging:
            # Scrubbing playhead
            frac = self._x_to_frac(ex)
            self._playhead = frac
            if self._duration_ms > 0:
                self._player.setPosition(int(frac * self._duration_ms))
            self.update()
            return

        # Hover: change cursor near split lines
        if ey <= self.WAVE_H + self.MARGIN_Y:
            split_idx = self._find_nearest_split(ex)
            if split_idx >= 0:
                if self._hover_split_idx != split_idx:
                    self._hover_split_idx = split_idx
                    self.setCursor(Qt.CursorShape.SplitHCursor)
                    self.update()
                return

        if self._hover_split_idx >= 0:
            self._hover_split_idx = -1
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.update()

        # Section tooltip on hover
        if ey <= self.WAVE_H + self.MARGIN_Y and self._sections:
            frac = self._x_to_frac(ex)
            for si, (start, end) in enumerate(self._sections):
                if start <= frac < end:
                    tip = f"Section {si + 1}/{len(self._sections)}"
                    if si < len(self._transposes):
                        t = self._transposes[si]
                        tip += f"  ·  Transpose: {t:+d}"
                    if si < len(self._section_info):
                        info = self._section_info[si]
                        fn = info.get("from_note", "")
                        tn = info.get("to_note", "")
                        if fn and tn:
                            tip += f"  ({fn} → {tn})"
                        dist = info.get("distance", 0)
                        if dist <= 3:
                            tip += "  ✓ great match"
                        elif dist <= 6:
                            tip += "  ~ decent match"
                        else:
                            tip += "  ⚠ stretch"
                    self.setToolTip(tip)
                    return
        self.setToolTip("")

    def mouseReleaseEvent(self, event):
        if self._dragging and self._drag_split_idx >= 0:
            self.sections_changed.emit()
        self._dragging = False
        self._drag_split_idx = -1

    def contextMenuEvent(self, event):
        """Right-click (two-finger click) to add or remove a split."""
        if self._readonly or not self._samples or not self._sections:
            return
        ex = event.pos().x()
        ey = event.pos().y()
        if ey > self.WAVE_H + self.MARGIN_Y:
            return

        frac = self._x_to_frac(ex)

        # Check if right-clicking near an existing split — remove it
        split_idx = self._find_nearest_split(ex)
        if split_idx > 0:
            # Merge section[split_idx-1] and section[split_idx]
            prev_start = self._sections[split_idx - 1][0]
            curr_end = self._sections[split_idx][1]
            self._sections[split_idx - 1] = (prev_start, curr_end)
            self._sections.pop(split_idx)
            # Remove the transpose for the deleted section, keep the earlier one
            if split_idx < len(self._transposes):
                self._transposes.pop(split_idx)
            if split_idx < len(self._median_hz):
                self._median_hz.pop(split_idx)
            if split_idx < len(self._section_info):
                self._section_info.pop(split_idx)
            self._invalidate_wave_cache()
            self.sections_changed.emit()
            return

        # Otherwise, add a new split at this position
        min_gap = 0.02
        for i, (start, end) in enumerate(self._sections):
            if start + min_gap < frac < end - min_gap:
                # Split this section into two
                self._sections[i] = (start, frac)
                self._sections.insert(i + 1, (frac, end))
                # Duplicate the transpose for the new section
                t = self._transposes[i] if i < len(self._transposes) else 0
                self._transposes.insert(i + 1, t)
                hz = self._median_hz[i] if i < len(self._median_hz) else 0
                self._median_hz.insert(i + 1, hz)
                if i < len(self._section_info):
                    self._section_info.insert(i + 1, dict(self._section_info[i]))
                self._invalidate_wave_cache()
                self.sections_changed.emit()
                return

    def wheelEvent(self, event):
        """Scroll wheel on a section to shift ±12 semitones (one octave), staying in key."""
        if self._readonly or not self._sections or not self._transposes:
            return
        # Accumulate scroll delta, only trigger at full step threshold
        self._wheel_accum = getattr(self, '_wheel_accum', 0) + event.angleDelta().y()
        step = 200  # higher = slower (default scroll notch is ~120)
        if abs(self._wheel_accum) < step:
            event.accept()
            return
        delta = 12 if self._wheel_accum > 0 else -12
        self._wheel_accum = 0

        frac = self._x_to_frac(event.position().x())
        for si, (start, end) in enumerate(self._sections):
            if start <= frac < end and si < len(self._transposes):
                self._transposes[si] += delta
                self._invalidate_wave_cache()
                self.sections_changed.emit()
                event.accept()
                return
        event.ignore()

    @staticmethod
    def _fmt(ms):
        s = int(ms) // 1000
        return f"{s // 60}:{s % 60:02d}"

    def _section_distance(self, idx):
        """How far (in semitones) this section's transposed pitch is from model center."""
        if self._converted and idx < len(self._section_info):
            if self._section_info[idx].get("silent"):
                return 0  # silence — no transpose needed, show as green
            d = self._section_info[idx].get("distance", 99)
            if d < 99:
                return d
            # Unknown pitch post-conversion — fall through to pre-conversion estimate
        if idx >= len(self._median_hz) or idx >= len(self._transposes):
            return 99
        if self._median_hz[idx] <= 0 or self._model_center_hz <= 0:
            return 0 if self._is_section_silent(idx) else 99
        import math
        shifted = self._median_hz[idx] * (2 ** (self._transposes[idx] / 12))
        if shifted <= 0:
            return 99
        return abs(12 * math.log2(shifted / self._model_center_hz))

    def _is_section_silent(self, idx):
        """Check if a section is mostly silence based on waveform samples."""
        if idx >= len(self._sections) or not self._samples:
            return False
        start, end = self._sections[idx]
        n = len(self._samples)
        i_start = int(start * n)
        i_end = max(i_start + 1, int(end * n))
        section_samples = self._samples[i_start:i_end]
        if not section_samples:
            return False
        avg_amp = sum(section_samples) / len(section_samples)
        return avg_amp < 0.02

    def _color_for_distance(self, dist, alpha_bg=50, alpha_bar=160):
        """Green/yellow/orange based on match quality. Neutral for unknown.
        Output (readonly) waveform uses darker, muted versions."""
        darken = self._readonly
        if dist >= 99:
            if darken:
                return QColor(255, 255, 255, 0), QColor(140, 140, 140, 100)
            return QColor(255, 255, 255, 0), QColor(200, 200, 200, 130)
        elif dist <= 3:
            r, g, b = (50, 140, 80) if darken else (80, 200, 120)
            return QColor(r, g, b, alpha_bg), QColor(r, g, b, alpha_bar)
        elif dist <= 6:
            r, g, b = (150, 150, 50) if darken else (200, 200, 80)
            return QColor(r, g, b, alpha_bg), QColor(r, g, b, alpha_bar)
        else:
            r, g, b = (190, 100, 50) if darken else (255, 140, 80)
            return QColor(r, g, b, alpha_bg), QColor(r, g, b, alpha_bar)

    def _invalidate_wave_cache(self):
        self._wave_cache = None
        self.update()

    def _build_wave_cache(self):
        """Pre-render the static waveform (bars, sections, labels, splits) to a pixmap."""
        w, h = self.width(), self.height()
        mx, my = self.MARGIN_X, self.MARGIN_Y
        draw_w = w - mx * 2
        wave_h = self.WAVE_H
        mid_y = my + wave_h / 2
        n = len(self._samples)

        cache = QPixmap(w, h)
        cache.fill(Qt.GlobalColor.transparent)
        p = QPainter(cache)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Rounded background
        bg_path = QPainterPath()
        bg_path.addRoundedRect(QRectF(0, 0, w, h), 10, 10)
        p.fillPath(bg_path, QColor(255, 255, 255, 8))

        # Section colors based on match quality
        sec_bg = []
        sec_bar = []
        for i in range(len(self._sections)):
            dist = self._section_distance(i)
            bg, bar = self._color_for_distance(dist)
            sec_bg.append(bg)
            sec_bar.append(bar)

        # Build section lookup for bars (colors applied to bars only, no background fill)
        bar_colors = []
        for i in range(n):
            frac = i / n
            default_color = QColor(140, 140, 140, 100) if self._readonly else QColor(200, 200, 200, 130)
            color = default_color
            for si, (start, end) in enumerate(self._sections):
                if start <= frac < end:
                    color = sec_bar[si] if si < len(sec_bar) else default_color
                    break
            bar_colors.append(color)

        # Waveform bars
        bar_w = max(1, draw_w / n * 0.6)
        for i, amp in enumerate(self._samples):
            x = mx + (i / n) * draw_w
            bar_h = amp * wave_h * 0.8
            p.setPen(QPen(bar_colors[i], bar_w))
            p.drawLine(int(x), int(mid_y - bar_h / 2), int(x), int(mid_y + bar_h / 2))

        # Split lines
        for i, (start, _) in enumerate(self._sections):
            if i == 0:
                continue
            x = mx + start * draw_w
            p.setPen(QPen(QColor(255, 255, 255, 120), 1, Qt.PenStyle.DashLine))
            p.drawLine(int(x), my, int(x), my + wave_h)

        # Transpose labels
        font = QFont("Manrope", 8, QFont.Weight.Bold)
        p.setFont(font)
        p.setPen(QColor(255, 255, 255, 200))
        for i, (start, end) in enumerate(self._sections):
            if i >= len(self._transposes):
                break
            t = self._transposes[i]
            label = f"{t:+d}" if t != 0 else "0"
            x1 = mx + start * draw_w
            x2 = mx + end * draw_w
            if x2 - x1 < 20:
                continue
            p.drawText(QRectF(x1, my + 3, x2 - x1, 14), Qt.AlignmentFlag.AlignCenter, label)

        p.end()
        self._wave_cache = cache

    def paintEvent(self, event):
        if not self._samples:
            return

        w, h = self.width(), self.height()
        mx, my = self.MARGIN_X, self.MARGIN_Y
        draw_w = w - mx * 2
        wave_h = self.WAVE_H

        # Build cache if needed
        if self._wave_cache is None or self._wave_cache.size().width() != w:
            self._build_wave_cache()

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw cached static waveform
        p.drawPixmap(0, 0, self._wave_cache)

        # Active section highlight (dynamic)
        if self._active_section >= 0 and self._active_section < len(self._sections):
            start, end = self._sections[self._active_section]
            x1 = mx + start * draw_w
            x2 = mx + end * draw_w
            p.fillRect(QRectF(x1, my, x2 - x1, wave_h), QColor(255, 255, 255, 30))

        # Hovered/dragged split handle (dynamic)
        for idx in [self._hover_split_idx, self._drag_split_idx]:
            if idx > 0 and idx < len(self._sections):
                x = mx + self._sections[idx][0] * draw_w
                p.setPen(QPen(QColor(255, 255, 255, 240), 2))
                p.drawLine(int(x), my, int(x), my + wave_h)
                p.setBrush(QColor("white"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QRectF(x - 4, my + wave_h / 2 - 4, 8, 8))

        # Conversion progress overlay
        if self._progress > 0:
            prog_x = mx + self._progress * draw_w
            p.fillRect(QRectF(mx, my, prog_x - mx, wave_h), QColor(37, 99, 235, 30))
            p.setPen(QPen(QColor("#3b82f6"), 2))
            p.drawLine(int(prog_x), my, int(prog_x), my + wave_h)

        # ===== BOTTOM BAR =====
        bar_y = my + wave_h + 2

        is_playing = self._is_playing
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 160))
        bx = mx + 2
        by = bar_y + 2
        if is_playing:
            p.drawRect(int(bx), int(by), 3, 10)
            p.drawRect(int(bx + 5), int(by), 3, 10)
        else:
            triangle = QPainterPath()
            triangle.moveTo(bx, by)
            triangle.lineTo(bx, by + 10)
            triangle.lineTo(bx + 9, by + 5)
            triangle.closeSubpath()
            p.drawPath(triangle)

        if self._duration_ms > 0:
            pos_ms = self._playhead * self._duration_ms
            time_text = f"{self._fmt(pos_ms)} / {self._fmt(self._duration_ms)}"
        else:
            time_text = "0:00"
        p.setPen(QColor(255, 255, 255, 100))
        p.setFont(self._time_font)
        p.drawText(QRectF(bx + 16, bar_y, 100, self.TIME_H),
                   Qt.AlignmentFlag.AlignVCenter, time_text)

        # Playhead
        if self._playhead > 0:
            px = mx + self._playhead * draw_w
            p.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
            p.drawLine(int(px), my, int(px), my + wave_h)
            p.setBrush(QColor("white"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(px - 3, my - 3, 6, 6))

        p.end()


class _ModelCarousel(QWidget):
    """Horizontal carousel with large center image and smaller flanking images."""
    model_selected = pyqtSignal(int)
    key_badge_clicked = pyqtSignal(int)  # emitted when "?" badge is clicked

    CENTER_SIZE = 175
    SIDE_SIZE = 70
    FAR_SIZE = 48
    SPACING = 24
    BADGE_SIZE = 28
    _badge_font = QFont("Manrope", 9, QFont.Weight.Bold)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._models = []
        self._selected = 0
        self._circular_cache = {}
        self._badge_rect = None  # cached hit area for the key badge
        self._best_match_pixmap = None  # circular artist image for best match overlay
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Animated GIF for Best Match
        self._gif_movie = None
        gif_path = os.path.join(APP_DIR, "assets", "best_match.gif")
        if os.path.exists(gif_path):
            from PyQt6.QtGui import QMovie
            self._gif_movie = QMovie(gif_path)
            self._gif_movie.frameChanged.connect(self._on_gif_frame)
            self._gif_movie.start()

        # Animation state: float position that lerps toward _selected
        self._anim_pos = 0.0
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)  # ~60fps
        self._anim_timer.timeout.connect(self._anim_tick)

    def _on_gif_frame(self):
        """Update Best Match (index 0) circular cache with current GIF frame."""
        if not self._models or not self._gif_movie:
            return
        # Only repaint if Best Match is visible (near the selected item)
        if abs(0 - self._anim_pos) > 4:
            return
        frame = self._gif_movie.currentPixmap()
        if frame.isNull():
            return
        big = int(self.CENTER_SIZE * 1.4)
        self._circular_cache[(0, big)] = self._make_faded_circular(frame, big)
        self.update()

    def set_models(self, models):
        self._models = models
        self._circular_cache.clear()
        self._selected = 0
        self._anim_pos = 0.0
        self._badge_rect = None
        self.update()

    def set_vocal_key(self, idx, key):
        """Update the vocal key for a model and repaint."""
        if 0 <= idx < len(self._models):
            self._models[idx]["vocal_key"] = key
            self.update()

    def select(self, idx):
        if 0 <= idx < len(self._models):
            self._selected = idx
            self.model_selected.emit(idx)
            if not self._anim_timer.isActive():
                self._anim_timer.start()

    def _anim_tick(self):
        diff = self._selected - self._anim_pos
        if abs(diff) < 0.01:
            self._anim_pos = float(self._selected)
            self._anim_timer.stop()
        else:
            self._anim_pos += diff * 0.2  # smooth lerp
        self.update()

    def wheelEvent(self, event):
        if not self._models:
            return
        delta = event.angleDelta().y()
        if delta > 0 and self._selected > 0:
            self.select(self._selected - 1)
        elif delta < 0 and self._selected < len(self._models) - 1:
            self.select(self._selected + 1)
        event.accept()

    def keyPressEvent(self, event):
        if not self._models:
            return
        if event.key() == Qt.Key.Key_Left and self._selected > 0:
            self.select(self._selected - 1)
        elif event.key() == Qt.Key.Key_Right and self._selected < len(self._models) - 1:
            self.select(self._selected + 1)
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if not self._models:
            return
        # Check if the key badge ("?") was clicked
        if self._badge_rect and self._badge_rect.contains(event.pos()):
            model = self._models[self._selected]
            key = model.get("vocal_key", "")
            if not key or key == "Auto":
                self.key_badge_clicked.emit(self._selected)
                return

        # Figure out which item was clicked based on x position
        cx = self.width() // 2
        click_x = event.pos().x()

        # Check center
        half_center = self.CENTER_SIZE // 2
        if abs(click_x - cx) <= half_center:
            return  # Already selected

        # Check items to the right and left
        positions = self._get_positions()
        for idx, (x, size, y_off) in positions.items():
            if abs(click_x - x) <= size // 2:
                self.select(idx)
                return

    def _size_for_dist(self, dist):
        """Smooth size based on distance from center."""
        if dist <= 1.0:
            t = dist * dist * (3 - 2 * dist)  # smoothstep
            return self.CENTER_SIZE + (self.SIDE_SIZE - self.CENTER_SIZE) * t
        elif dist <= 2.0:
            t = (dist - 1.0)
            t = t * t * (3 - 2 * t)
            return self.SIDE_SIZE + (self.FAR_SIZE - self.SIDE_SIZE) * t
        else:
            return max(20.0, self.FAR_SIZE - (dist - 2.0) * 6.0)

    def _get_positions(self):
        """Calculate positions using a simple continuous offset from animated center."""
        cx = self.width() / 2.0
        positions = {}

        # Pre-calculate all sizes
        items = []
        for i in range(len(self._models)):
            dist = abs(i - self._anim_pos)
            if dist > 6:
                continue
            size = self._size_for_dist(dist)
            items.append((i, dist, size))

        # Place items by walking outward from center in both directions
        # First, find the center item's x (which is cx shifted by fractional offset)
        # The key insight: each item's center-x depends only on its distance from anim_pos

        for i, dist, size in items:
            # Signed distance (positive = right of center)
            signed = i - self._anim_pos

            # Walk from center to this item, accumulating half-widths + gaps
            if abs(signed) < 0.001:
                x = cx
            else:
                direction = 1.0 if signed > 0 else -1.0
                # Simple formula: x = cx + direction * (sum of spacing between center and item)
                # Each slot is: half of left item + gap + half of right item
                x = cx

                # Start from center edge
                center_size = self._size_for_dist(0)

                # Use continuous integration approach:
                # Position = center + integral of (size + spacing) from 0 to |signed|
                # Approximate with the actual distance
                if dist <= 1.0:
                    # Interpolate between 0 and first slot position
                    slot1_x = center_size / 2.0 + self.SPACING + self._size_for_dist(1.0) / 2.0
                    x = cx + direction * slot1_x * dist
                else:
                    # First slot
                    slot1_size = self._size_for_dist(1.0)
                    offset = center_size / 2.0 + self.SPACING + slot1_size / 2.0

                    # Additional full slots
                    for d in range(2, int(dist) + 1):
                        prev_size = self._size_for_dist(d - 1)
                        curr_size = self._size_for_dist(d)
                        offset += prev_size / 2.0 + self.SPACING + curr_size / 2.0

                    # Fractional part of the last slot
                    frac = dist - int(dist)
                    if frac > 0:
                        prev_size = self._size_for_dist(int(dist))
                        next_size = self._size_for_dist(int(dist) + 1)
                        slot_width = prev_size / 2.0 + self.SPACING + next_size / 2.0
                        offset += slot_width * frac

                    x = cx + direction * offset

            # Curve: items rise upward as they move from center
            # Parabolic curve — dist 0 = bottom, dist 3+ = max height
            curve_height = 30  # max vertical lift in pixels
            if dist <= 0.01:
                y_offset = 0
            else:
                y_offset = -curve_height * min(dist / 3.0, 1.0) ** 0.7

            positions[i] = (int(x), int(size), int(y_offset))

        return positions

    @staticmethod
    def _make_faded_circular(pixmap, size):
        """Create a circular image with soft gradient edges that fade to transparent."""
        from PyQt6.QtGui import QRadialGradient
        scaled = pixmap.scaled(size, size,
                               Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                               Qt.TransformationMode.SmoothTransformation)
        # Center-crop
        x = (scaled.width() - size) // 2
        y = (scaled.height() - size) // 2
        cropped = scaled.copy(x, y, size, size)

        result = QPixmap(size, size)
        result.fill(Qt.GlobalColor.transparent)
        p = QPainter(result)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw image
        p.drawPixmap(0, 0, cropped)

        # Apply radial gradient mask that fades edges
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        gradient = QRadialGradient(size / 2, size / 2, size / 2)
        gradient.setColorAt(0.0, QColor(0, 0, 0, 255))
        gradient.setColorAt(0.2, QColor(0, 0, 0, 180))
        gradient.setColorAt(0.45, QColor(0, 0, 0, 50))
        gradient.setColorAt(0.7, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(gradient))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(0, 0, size, size)
        p.end()
        return result

    def _get_circular(self, idx, size):
        """Get a circular pixmap for model at idx. Uses cached base size, fast-scales for animation."""
        # Cache at the largest size this item will ever render at
        base_size = int(self.CENTER_SIZE * 1.4) if idx == 0 else self.CENTER_SIZE
        base_key = (idx, base_size)

        if base_key not in self._circular_cache:
            model = self._models[idx]
            # For Best Match with GIF, use current frame with soft fade
            if idx == 0 and self._gif_movie:
                px = self._gif_movie.currentPixmap()
            else:
                px = model.get("pixmap")
            if idx == 0 and px and not px.isNull():
                result = self._make_faded_circular(px, base_size)
            elif px and not px.isNull():
                from ui.widgets.voice_card import VoiceCard
                result = VoiceCard._make_circular(px, base_size)
            else:
                result = QPixmap(base_size, base_size)
                result.fill(Qt.GlobalColor.transparent)
                painter = QPainter(result)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                path = QPainterPath()
                path.addEllipse(0, 0, base_size, base_size)
                painter.setClipPath(path)
                painter.fillRect(0, 0, base_size, base_size, QBrush(QColor("#2563eb")))
                painter.setPen(Qt.GlobalColor.white)
                font = QFont("Manrope", max(8, base_size // 3))
                font.setBold(True)
                painter.setFont(font)
                initials = model["name"][0].upper() if model["name"] else "?"
                painter.drawText(result.rect(), Qt.AlignmentFlag.AlignCenter, initials)
                painter.end()
            self._circular_cache[base_key] = result

        base = self._circular_cache[base_key]
        if size == base_size:
            return base
        # Use smooth scaling for center items, fast for animation
        mode = Qt.TransformationMode.SmoothTransformation if size >= self.CENTER_SIZE else Qt.TransformationMode.FastTransformation
        return base.scaled(size, size, Qt.AspectRatioMode.IgnoreAspectRatio, mode)

    def paintEvent(self, event):
        if not self._models:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cy = self.height() // 2
        positions = self._get_positions()

        # Draw far items first, center last
        sorted_items = sorted(positions.items(),
                              key=lambda x: abs(x[0] - self._anim_pos), reverse=True)

        for idx, (x, size, y_off) in sorted_items:
            dist = abs(idx - self._anim_pos)

            # Smooth opacity falloff
            if dist <= 1.0:
                opacity = 1.0
            elif dist <= 4.0:
                opacity = 1.0 - (dist - 1.0) / 3.0
            else:
                opacity = 0.0

            if opacity <= 0.01:
                continue

            painter.setOpacity(opacity)

            render_size = int(size * 1.4) if idx == 0 else size
            px = self._get_circular(idx, render_size)
            draw_x = x - render_size // 2
            draw_y = cy - render_size // 2 + y_off
            painter.drawPixmap(draw_x, draw_y, px)

            # Draw border (skip for Best Match — it has a faded edge)
            if idx != 0:
                is_center = dist < 0.3
                border_color = QColor(255, 255, 255, 200) if is_center else QColor(80, 80, 80, int(120 * opacity))
                pen = QPen(border_color, 2 if is_center else 1)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(draw_x, draw_y, size, size)

                # Draw artist name below non-center items
                if not is_center and idx < len(self._models):
                    artist = self._models[idx].get("artist", "")
                    if artist:
                        painter.setFont(QFont("Manrope", 8))
                        painter.setPen(QColor(220, 220, 220, int(180 * opacity)))
                        name_y = draw_y + size + 6
                        painter.drawText(QRectF(x - 70, name_y, 140, 20),
                                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                                         artist)

        # Draw vocal key badge only when settled (not during scroll animation)
        painter.setOpacity(1.0)
        self._badge_rect = None
        if not self._anim_timer.isActive() and self._selected in positions and self._selected < len(self._models) and self._selected != 0:
            cx_pos, c_size, c_yoff = positions[self._selected]
            center_draw_y = cy - c_size // 2 + c_yoff
            model = self._models[self._selected]
            key = model.get("vocal_key", "")
            badge_text = key if (key and key != "Auto") else "?"
            bs = self.BADGE_SIZE
            bx = cx_pos - bs // 2
            by = center_draw_y - bs // 3

            badge_path = QPainterPath()
            badge_path.addRoundedRect(QRectF(bx, by, bs, bs), bs / 2, bs / 2)
            from PyQt6.QtGui import QLinearGradient
            badge_grad = QLinearGradient(bx, by, bx, by + bs)
            if badge_text == "?":
                badge_grad.setColorAt(0.0, QColor(80, 80, 80, 200))
                badge_grad.setColorAt(1.0, QColor(40, 40, 40, 200))
            else:
                badge_grad.setColorAt(0.0, QColor(80, 80, 80, 220))
                badge_grad.setColorAt(1.0, QColor(20, 20, 20, 220))
            painter.fillPath(badge_path, QBrush(badge_grad))
            painter.setFont(self._badge_font)
            painter.setPen(QColor(255, 255, 255, 240))
            painter.drawText(QRectF(bx, by, bs, bs), Qt.AlignmentFlag.AlignCenter, badge_text)

            self._badge_rect = QRectF(bx, by, bs, bs).toAlignedRect()

        # Draw best match artist overlay on bottom of GIF
        if (not self._anim_timer.isActive() and self._selected == 0
                and self._best_match_pixmap and 0 in positions):
            cx_pos, c_size, c_yoff = positions[0]
            render_size = int(c_size * 1.4)
            overlay_size = 48
            ox = cx_pos - overlay_size // 2
            oy = cy + render_size // 2 - overlay_size - 20 + c_yoff
            painter.setOpacity(1.0)
            painter.drawPixmap(ox, oy, self._best_match_pixmap)
            # Artist border
            painter.setPen(QPen(QColor(0, 0, 0, 120), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(ox, oy, overlay_size, overlay_size)

        painter.end()


class _ConvertButton(QWidget):
    """Circular convert button with progress ring."""
    clicked = pyqtSignal()

    SIZE = 80
    RING_WIDTH = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE + 8, self.SIZE + 8)  # extra for ring
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._progress = 0.0  # 0.0 to 1.0
        self._converting = False
        self._hover = False
        self._update_mode = False
        self._realtime_mode = False

    def set_progress(self, value):
        self._progress = max(0.0, min(1.0, value))
        self.update()

    def set_converting(self, active):
        self._converting = active
        if not active:
            self._progress = 0.0
        self.update()

    def set_update_mode(self, active):
        self._update_mode = active
        self.update()

    def set_realtime_mode(self, active):
        self._realtime_mode = active
        self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def mousePressEvent(self, event):
        if not self._converting or self._realtime_mode:
            self.clicked.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width() / 2
        cy = self.height() / 2
        r = self.SIZE / 2

        # Background circle with gradient
        from PyQt6.QtGui import QLinearGradient
        if self._converting:
            bg = QColor("#1a1a2e")
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(bg))
        elif self._hover:
            grad = QLinearGradient(cx, cy - r, cx, cy + r)
            grad.setColorAt(0.0, QColor(70, 70, 70))
            grad.setColorAt(1.0, QColor(45, 45, 45))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
        else:
            grad = QLinearGradient(cx, cy - r, cx, cy + r)
            grad.setColorAt(0.0, QColor(60, 60, 60))
            grad.setColorAt(1.0, QColor(35, 35, 35))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
        painter.drawEllipse(int(cx - r), int(cy - r), self.SIZE, self.SIZE)

        # Progress ring
        if self._converting and self._progress > 0:
            ring_rect = QRectF(
                cx - r - self.RING_WIDTH / 2,
                cy - r - self.RING_WIDTH / 2,
                self.SIZE + self.RING_WIDTH,
                self.SIZE + self.RING_WIDTH,
            )

            # Background track
            track_pen = QPen(QColor(255, 255, 255, 20), self.RING_WIDTH)
            track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(track_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(ring_rect)

            # Progress arc
            progress_pen = QPen(QColor("#3b82f6"), self.RING_WIDTH)
            progress_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(progress_pen)
            start = 90 * 16  # top
            span = int(-self._progress * 360 * 16)  # clockwise
            painter.drawArc(ring_rect, start, span)

        # Text
        painter.setPen(QColor("white"))
        font = QFont("Helvetica", 11)
        font.setBold(True)
        painter.setFont(font)

        if self._realtime_mode and self._converting:
            text = "Stop"
        elif self._realtime_mode:
            text = "Start"
        elif self._converting:
            pct = int(self._progress * 100)
            text = f"{pct}%"
        elif self._update_mode:
            text = "Update"
        else:
            text = "Convert"

        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()


class _CreateModelPanel(QWidget):
    """Unified create-a-model panel combining dataset + training in a clean flow."""
    back_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self._init_ui()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        p.end()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(12)

        # Header row with back button
        header = QHBoxLayout()
        back = QLabel("← Back")
        back.setStyleSheet("color: rgba(255, 255, 255, 60); font-size: 12px; background: transparent;")
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.mousePressEvent = lambda e: self.back_clicked.emit()
        header.addWidget(back)
        header.addStretch()

        title = QLabel("Create a Model")
        title.setStyleSheet("color: #ddd; font-size: 18px; font-weight: bold; background: transparent;")
        header.addWidget(title)
        header.addStretch()
        # Spacer to balance the back button
        spacer = QLabel("")
        spacer.setFixedWidth(40)
        header.addWidget(spacer)
        layout.addLayout(header)

        subtitle = QLabel("Name your voice, add audio clips, and train on a cloud GPU")
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 40); font-size: 11px; background: transparent;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        # Model image grid
        from PyQt6.QtWidgets import QScrollArea
        self._model_grid = QWidget()
        self._model_grid_layout = QHBoxLayout(self._model_grid)
        self._model_grid_layout.setContentsMargins(0, 0, 0, 0)
        self._model_grid_layout.setSpacing(12)
        self._model_grid_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._model_scroll = QScrollArea()
        self._model_scroll.setWidget(self._model_grid)
        self._model_scroll.setWidgetResizable(False)
        self._model_scroll.setFixedHeight(80)
        self._model_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._model_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._model_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; } QScrollArea > QWidget > QWidget { background: transparent; }")
        self._model_scroll.viewport().setStyleSheet("background: transparent;")
        layout.addWidget(self._model_scroll)

        # Selected model label (shown below images when one is picked)
        self._lbl_selected = QLabel("")
        self._lbl_selected.setStyleSheet("color: rgba(255, 255, 255, 50); font-size: 11px; background: transparent;")
        self._lbl_selected.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_selected)

        # New name input (will be added to end of model grid in _populate)
        self._txt_new_name = QLineEdit()
        self._txt_new_name.setPlaceholderText("+ New artist")
        self._txt_new_name.setFixedSize(80, 44)
        self._txt_new_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._txt_new_name.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 5);
                border: 1px dashed rgba(255, 255, 255, 20);
                border-radius: 22px;
                color: #aaa;
                font-size: 9px;
                padding: 0 4px;
            }
            QLineEdit:focus {
                border-color: rgba(255, 255, 255, 40);
                border-style: solid;
                color: #ccc;
            }
        """)
        self._txt_new_name.returnPressed.connect(self._on_new_name_entered)

        layout.addSpacing(4)

        # Audio Clips section
        step2 = QLabel("Audio Clips")
        step2.setStyleSheet("color: rgba(255, 255, 255, 70); font-size: 11px; font-weight: bold; background: transparent;")
        layout.addWidget(step2)

        # File list showing existing clips
        self._file_list = QListWidget()
        self._file_list.setFixedHeight(200)
        self._file_list.setStyleSheet("""
            QListWidget {
                background: rgba(255, 255, 255, 3);
                border: 1px solid rgba(255, 255, 255, 10);
                border-radius: 10px;
                color: #999;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 3px 8px;
                border-bottom: 1px solid rgba(255, 255, 255, 5);
            }
            QListWidget::item:selected {
                background: rgba(255, 255, 255, 8);
                color: #ccc;
            }
        """)
        layout.addWidget(self._file_list)

        self._lbl_clips = QLabel("")
        self._lbl_clips.setStyleSheet("color: rgba(255, 255, 255, 40); font-size: 10px; background: transparent;")
        self._lbl_clips.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_clips)

        # Action row: browse + isolate
        action_row = QHBoxLayout()
        action_row.addStretch()

        self._btn_browse_clips = QLabel("+ Add files")
        self._btn_browse_clips.setStyleSheet("color: rgba(255, 255, 255, 60); font-size: 10px; background: transparent;")
        self._btn_browse_clips.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_browse_clips.mousePressEvent = lambda e: self._browse_clips()
        action_row.addWidget(self._btn_browse_clips)

        dot1 = QLabel(" · ")
        dot1.setStyleSheet("color: rgba(255, 255, 255, 30); font-size: 10px; background: transparent;")
        action_row.addWidget(dot1)

        self._btn_isolate = QLabel("Isolate vocals from songs")
        self._btn_isolate.setStyleSheet("color: rgba(94, 200, 180, 80); font-size: 10px; background: transparent;")
        self._btn_isolate.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_isolate.mousePressEvent = lambda e: self._isolate_vocals()
        action_row.addWidget(self._btn_isolate)

        dot2 = QLabel(" · ")
        dot2.setStyleSheet("color: rgba(255, 255, 255, 30); font-size: 10px; background: transparent;")
        action_row.addWidget(dot2)

        self._btn_remove_clip = QLabel("Remove selected")
        self._btn_remove_clip.setStyleSheet("color: rgba(255, 100, 100, 60); font-size: 10px; background: transparent;")
        self._btn_remove_clip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_remove_clip.mousePressEvent = lambda e: self._remove_selected_clip()
        action_row.addWidget(self._btn_remove_clip)

        dot3 = QLabel(" · ")
        dot3.setStyleSheet("color: rgba(255, 255, 255, 30); font-size: 10px; background: transparent;")
        action_row.addWidget(dot3)

        self._btn_split_clip = QLabel("Split selected")
        self._btn_split_clip.setStyleSheet("color: rgba(94, 200, 180, 80); font-size: 10px; background: transparent;")
        self._btn_split_clip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_split_clip.mousePressEvent = lambda e: self._split_selected_clip()
        action_row.addWidget(self._btn_split_clip)

        dot4 = QLabel(" · ")
        dot4.setStyleSheet("color: rgba(255, 255, 255, 30); font-size: 10px; background: transparent;")
        action_row.addWidget(dot4)

        self._btn_delete_dataset = QLabel("Delete dataset")
        self._btn_delete_dataset.setStyleSheet("color: rgba(255, 100, 100, 60); font-size: 10px; background: transparent;")
        self._btn_delete_dataset.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_delete_dataset.mousePressEvent = lambda e: self._delete_dataset()
        action_row.addWidget(self._btn_delete_dataset)

        action_row.addStretch()
        layout.addLayout(action_row)

        # Clip waveform preview
        self._clip_waveform = _WaveformWidget(readonly=True)
        self._clip_waveform.setVisible(False)
        layout.addWidget(self._clip_waveform)

        self._file_list.currentRowChanged.connect(self._on_clip_selected)

        layout.addSpacing(4)

        # Step 3: Training Options
        step3 = QLabel("3. Train")
        step3.setStyleSheet("color: rgba(255, 255, 255, 70); font-size: 11px; font-weight: bold; background: transparent;")
        layout.addWidget(step3)

        opts_row = QHBoxLayout()
        opts_row.setSpacing(12)

        type_lbl = QLabel("Type:")
        type_lbl.setStyleSheet("color: #888; font-size: 11px; background: transparent;")
        opts_row.addWidget(type_lbl)

        self._cmb_type = QComboBox()
        self._cmb_type.addItems(["Singing", "Speech"])
        self._cmb_type.setFixedHeight(28)
        self._cmb_type.setStyleSheet("""
            QComboBox {
                background: rgba(255, 255, 255, 8);
                border: 1px solid rgba(255, 255, 255, 15);
                border-radius: 6px;
                color: #ccc;
                font-size: 11px;
                padding: 0 8px;
            }
        """)
        opts_row.addWidget(self._cmb_type)

        opts_row.addSpacing(12)

        epoch_lbl = QLabel("Epochs:")
        epoch_lbl.setStyleSheet("color: #888; font-size: 11px; background: transparent;")
        opts_row.addWidget(epoch_lbl)

        self._txt_epochs = QLineEdit("Auto")
        self._txt_epochs.setFixedSize(60, 28)
        self._txt_epochs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._txt_epochs.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 8);
                border: 1px solid rgba(255, 255, 255, 15);
                border-radius: 6px;
                color: #ccc;
                font-size: 11px;
            }
        """)
        opts_row.addWidget(self._txt_epochs)
        opts_row.addStretch()
        layout.addLayout(opts_row)

        layout.addSpacing(8)

        # Train button
        self._btn_train = QPushButton("Start Training")
        self._btn_train.setFixedHeight(44)
        self._btn_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_train.setStyleSheet("""
            QPushButton {
                background: rgba(80, 200, 120, 40);
                border: 1px solid rgba(80, 200, 120, 80);
                border-radius: 12px;
                color: rgba(80, 200, 120, 200);
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(80, 200, 120, 60);
                border-color: rgba(80, 200, 120, 120);
            }
            QPushButton:disabled {
                background: rgba(255, 255, 255, 5);
                border-color: rgba(255, 255, 255, 10);
                color: #555;
            }
        """)
        self._btn_train.clicked.connect(self._start_training)

        self._btn_continue_train = QPushButton("Continue Training")
        self._btn_continue_train.setFixedHeight(44)
        self._btn_continue_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_continue_train.setVisible(False)
        self._btn_continue_train.setStyleSheet("""
            QPushButton {
                background: rgba(100, 160, 255, 30);
                border: 1px solid rgba(100, 160, 255, 70);
                border-radius: 12px;
                color: rgba(100, 160, 255, 200);
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(100, 160, 255, 50);
                border-color: rgba(100, 160, 255, 110);
            }
            QPushButton:disabled {
                background: rgba(255, 255, 255, 5);
                border-color: rgba(255, 255, 255, 10);
                color: #555;
            }
        """)
        self._btn_continue_train.clicked.connect(lambda: self._start_training(resume=True))

        train_row = QHBoxLayout()
        self._btn_stop_train = QPushButton("Stop Training")
        self._btn_stop_train.setFixedHeight(44)
        self._btn_stop_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_stop_train.setVisible(False)
        self._btn_stop_train.setStyleSheet("""
            QPushButton {
                background: rgba(255, 80, 80, 30);
                border: 1px solid rgba(255, 80, 80, 70);
                border-radius: 12px;
                color: rgba(255, 80, 80, 200);
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255, 80, 80, 50);
                border-color: rgba(255, 80, 80, 110);
            }
        """)
        self._btn_stop_train.clicked.connect(self._stop_training)

        train_row.addWidget(self._btn_train)
        train_row.addWidget(self._btn_continue_train)
        train_row.addWidget(self._btn_stop_train)
        layout.addLayout(train_row)

        # Progress / Status
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: rgba(255, 255, 255, 50); font-size: 11px; background: transparent;")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_status)

        from PyQt6.QtWidgets import QProgressBar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(255, 255, 255, 8);
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: rgba(80, 200, 120, 150);
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._progress_bar)

        from ui.widgets.log_viewer import LogViewer
        self._log = LogViewer()
        self._log.setVisible(False)
        self._log.setMaximumHeight(120)
        layout.addWidget(self._log)

        layout.addStretch()

        # State
        self._clips = []
        self._selected_name = ""
        self._training = False
        self._worker = None
        self._image_cache_dir = os.path.join(CACHE_DIR, "artist_thumbs")

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.toLocalFile().endswith(('.wav', '.flac', '.mp3', '.ogg'))]
        if paths:
            self._add_clips(paths)

    def _browse_clips(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Clips", "",
            "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if paths:
            self._add_clips(paths)

    def _add_clips(self, paths):
        self._clips.extend(paths)
        self._refresh_file_list()

    def _refresh_file_list(self):
        self._file_list.clear()
        try:
            import soundfile as _sf
            total_dur = 0
            for p in self._clips:
                name = os.path.basename(p)
                try:
                    info = _sf.info(p)
                    dur = info.duration
                    total_dur += dur
                    mins, secs = divmod(int(dur), 60)
                    self._file_list.addItem(f"{name}  ({mins}:{secs:02d})")
                except Exception:
                    self._file_list.addItem(name)
            total = len(self._clips)
            mins, secs = divmod(int(total_dur), 60)
            self._lbl_clips.setText(f"{total} clips  ·  {mins}:{secs:02d} total")
        except Exception:
            for p in self._clips:
                self._file_list.addItem(os.path.basename(p))
            self._lbl_clips.setText(f"{len(self._clips)} clips added")

    def _on_clip_selected(self, row):
        """Load waveform for the selected clip."""
        if row < 0 or row >= len(self._clips):
            self._clip_waveform.setVisible(False)
            return
        path = self._clips[row]
        if not os.path.exists(path):
            self._clip_waveform.setVisible(False)
            return
        # Load samples in background
        self._clip_sample_worker = _WaveformSamplesOnly(path)
        self._clip_sample_worker.finished.connect(self._on_clip_samples)
        self._clip_sample_worker.start()
        self._clip_waveform.load(path)

    def _on_clip_samples(self, samples):
        if samples:
            self._clip_waveform.set_data(samples, [], [], [])
            self._clip_waveform.setVisible(True)
        else:
            self._clip_waveform.setVisible(False)

    def _split_selected_clip(self):
        """Split selected clip into ~7 second chunks. Only splits if > 10 seconds."""
        row = self._file_list.currentRow()
        if row < 0 or row >= len(self._clips):
            return
        path = self._clips[row]
        if not os.path.exists(path):
            return

        import soundfile as _sf
        try:
            info = _sf.info(path)
            duration = info.duration
        except Exception:
            return

        if duration <= 10.0:
            self._lbl_clips.setText("Clip is too short to split (needs > 10s)")
            return

        # Split into ~7 second chunks
        chunk_sec = 7.0
        audio, sr = _sf.read(path)
        parent_dir = os.path.dirname(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        new_paths = []
        chunk_idx = 1
        pos = 0
        total_samples = len(audio)

        while pos < total_samples:
            end = min(pos + int(chunk_sec * sr), total_samples)
            remaining = total_samples - end
            # If remainder would be < 3 seconds, merge it into this chunk
            if remaining > 0 and remaining < int(3.0 * sr):
                end = total_samples
            chunk = audio[pos:end]
            chunk_path = os.path.join(parent_dir, f"{stem}_part{chunk_idx:02d}.wav")
            _sf.write(chunk_path, chunk, sr)
            new_paths.append(chunk_path)
            chunk_idx += 1
            pos = end

        # Replace original with splits
        self._clips.pop(row)
        os.remove(path)
        for i, p in enumerate(new_paths):
            self._clips.insert(row + i, p)
        self._refresh_file_list()

    def _remove_selected_clip(self):
        row = self._file_list.currentRow()
        if row >= 0 and row < len(self._clips):
            self._clips.pop(row)
            self._refresh_file_list()

    def _delete_dataset(self):
        name = self._selected_name.strip()
        if not name:
            return
        from services.paths import DATASETS_DIR
        dataset_dir = os.path.join(str(DATASETS_DIR), name)
        if not os.path.isdir(dataset_dir):
            return
        reply = QMessageBox.question(
            self, "Delete Dataset",
            f"Delete dataset \"{name}\" and all its audio clips?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            import shutil
            shutil.rmtree(dataset_dir, ignore_errors=True)
            self._selected_name = ""
            self._clips = []
            self._refresh_file_list()
            self._lbl_selected.setText("")
            self._populate_existing_datasets()

    def _select_model(self, name):
        """Select a model/dataset by name and load its clips."""
        self._selected_name = name
        self._lbl_selected.setText(name)
        self._lbl_selected.setStyleSheet("color: rgba(255, 255, 255, 80); font-size: 12px; font-weight: bold; background: transparent;")
        self._txt_new_name.clear()
        # Load existing clips
        from services.paths import DATASETS_DIR
        dataset_dir = os.path.join(str(DATASETS_DIR), name)
        if os.path.isdir(dataset_dir):
            self._clips = [os.path.join(dataset_dir, f) for f in sorted(os.listdir(dataset_dir))
                           if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))]
        else:
            self._clips = []
        self._refresh_file_list()
        # Highlight selected in grid
        self._update_grid_selection()
        # Show "Continue Training" if model has existing checkpoints
        self._check_existing_model(name)

    def _check_existing_model(self, name):
        """Show Continue Training button if model has G_*.pth checkpoints."""
        from services.paths import MODELS_DIR
        model_dir = os.path.join(str(MODELS_DIR), name)
        has_checkpoint = False
        if os.path.isdir(model_dir):
            has_checkpoint = any(f.startswith("G_") and f.endswith(".pth") for f in os.listdir(model_dir))
        self._btn_continue_train.setVisible(has_checkpoint)

    def _on_new_name_entered(self):
        """User typed a new artist name and pressed Enter."""
        name = self._txt_new_name.text().strip()
        if not name:
            return
        self._selected_name = name
        self._lbl_selected.setText(name)
        self._lbl_selected.setStyleSheet("color: rgba(255, 255, 255, 80); font-size: 12px; font-weight: bold; background: transparent;")
        self._clips = []
        self._refresh_file_list()
        self._update_grid_selection()
        self._btn_continue_train.setVisible(False)

    def _update_grid_selection(self):
        """Update visual selection state — brighten selected name."""
        for i in range(self._model_grid_layout.count()):
            w = self._model_grid_layout.itemAt(i).widget()
            if w and hasattr(w, '_model_name'):
                selected = w._model_name == self._selected_name
                # Find the name label (second child)
                name_lbl = w.findChild(QLabel, "name_lbl")
                if name_lbl:
                    name_lbl.setStyleSheet(f"color: {'#fff' if selected else '#888'}; font-size: 8px; background: transparent;")

    def _populate_existing_datasets(self):
        """Build visual grid of model images from downloaded models and datasets."""
        from services.paths import MODELS_DIR, DATASETS_DIR
        from ui.widgets.voice_card import VoiceCard

        # Clear existing grid
        while self._model_grid_layout.count():
            item = self._model_grid_layout.takeAt(0)
            w = item.widget()
            if w and w is not self._txt_new_name:
                w.deleteLater()

        # Collect names from both models and datasets
        names = set()
        models_dir = str(MODELS_DIR)
        if os.path.isdir(models_dir):
            for name in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, name)):
                    names.add(name)
        ds_dir = str(DATASETS_DIR)
        if os.path.isdir(ds_dir):
            for name in os.listdir(ds_dir):
                if os.path.isdir(os.path.join(ds_dir, name)):
                    names.add(name)

        for name in sorted(names):
            card = QWidget()
            card._model_name = name
            card.setFixedSize(56, 68)
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setStyleSheet("background: transparent;")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(0, 0, 0, 0)
            card_layout.setSpacing(3)

            # Image
            img_lbl = QLabel()
            img_lbl.setFixedSize(44, 44)
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setStyleSheet("background: transparent;")
            # Check model folder image first, then artist thumbs cache
            img_found = False
            model_img = os.path.join(str(MODELS_DIR), name, "image.jpg")
            thumb_path = os.path.join(self._image_cache_dir, f"{name}.jpg")
            for img_path in [model_img, thumb_path]:
                if os.path.exists(img_path):
                    px = QPixmap(img_path)
                    img_lbl.setPixmap(VoiceCard._make_circular(px, 44))
                    img_found = True
                    break
            if not img_found:
                initials = "".join(w[0].upper() for w in name.split()[:2]) if name else "?"
                img_lbl.setText(initials)
                img_lbl.setStyleSheet("background: rgba(255,255,255,12); border-radius: 22px; color: #aaa; font-size: 14px; font-weight: bold;")
            card_layout.addWidget(img_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

            # Name
            name_lbl = QLabel(name)
            name_lbl.setObjectName("name_lbl")
            name_lbl.setStyleSheet("color: #888; font-size: 8px; background: transparent;")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_lbl.setWordWrap(True)
            card_layout.addWidget(name_lbl)

            card.mousePressEvent = lambda e, n=name: self._select_model(n)
            self._model_grid_layout.addWidget(card)

        # Add the "+ New artist" input at the end
        self._txt_new_name.clear()
        self._model_grid_layout.addWidget(self._txt_new_name)

        self._model_grid.adjustSize()

    def _isolate_vocals(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Songs to Extract Vocals From", "",
            "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if not paths:
            return
        self._lbl_status.setText("Isolating vocals...")
        # Run in background
        from services.vocal_separator import VocalSeparator
        import tempfile
        self._iso_dir = tempfile.mkdtemp(prefix="svc_iso_")

        class _IsoWorker(QThread):
            finished = pyqtSignal(list)
            def __init__(self, paths, out_dir):
                super().__init__()
                self.paths, self.out_dir = paths, out_dir
            def run(self):
                sep = VocalSeparator()
                vocals = []
                for p in self.paths:
                    try:
                        stems = sep.separate(p, self.out_dir)
                        vocals.append(stems["vocals"])
                    except Exception:
                        pass
                self.finished.emit(vocals)

        self._iso_worker = _IsoWorker(paths, self._iso_dir)
        self._iso_worker.finished.connect(self._on_iso_done)
        self._iso_worker.start()

    def _on_iso_done(self, vocals):
        self._lbl_status.setText("")
        if vocals:
            self._add_clips(vocals)

    def _start_training(self, resume=False):
        name = self._selected_name.strip()
        if not name:
            QMessageBox.warning(self, "No Name", "Enter a voice name first.")
            return
        if not self._clips:
            QMessageBox.warning(self, "No Audio", "Add audio clips first.")
            return

        # Copy clips to dataset dir
        from services.paths import DATASETS_DIR
        import shutil
        dataset_dir = os.path.join(str(DATASETS_DIR), name)
        os.makedirs(dataset_dir, exist_ok=True)
        for clip in self._clips:
            dest = os.path.join(dataset_dir, os.path.basename(clip))
            if not os.path.exists(dest):
                shutil.copy2(clip, dest)

        self._btn_train.setEnabled(False)
        self._btn_continue_train.setEnabled(False)
        btn_label = "Continuing..." if resume else "Training..."
        self._btn_train.setText(btn_label)
        self._log.setVisible(True)
        self._log.clear_log()
        self._lbl_status.setText("Resuming training..." if resume else "Starting training pipeline...")

        voice_type = self._cmb_type.currentText().lower()
        f0_method = "crepe" if voice_type == "singing" else "dio"

        epochs_text = self._txt_epochs.text().strip()
        max_epochs = None
        if epochs_text and epochs_text.lower() != "auto":
            try:
                max_epochs = int(epochs_text)
            except ValueError:
                pass

        # Find latest checkpoint if resuming
        resume_from = ""
        if resume:
            from services.paths import MODELS_DIR
            model_dir = os.path.join(str(MODELS_DIR), name)
            if os.path.isdir(model_dir):
                g_files = sorted(
                    [f for f in os.listdir(model_dir) if f.startswith("G_") and f.endswith(".pth")],
                    key=lambda f: int(f.replace("G_", "").replace(".pth", "")) if f.replace("G_", "").replace(".pth", "").isdigit() else 0,
                )
                if g_files:
                    resume_from = os.path.join(model_dir, g_files[-1])
                    self._log.append_line(f"Resuming from checkpoint: {g_files[-1]}")

        try:
            from workers.training_worker import TrainingWorker
            from services.paths import MODELS_DIR, DATASETS_DIR
            from services.job_store import load_config
            from services.dataset_manager import DatasetManager
            import uuid

            config = load_config()
            api_key = config.get("runpod_api_key", "") or os.environ.get("SOMERSVC_RUNPOD_KEY", "")
            ssh_key = os.path.expanduser(config.get("ssh_key_path", "~/.ssh/id_rsa"))

            if not api_key:
                QMessageBox.warning(self, "No API Key", "Set your RunPod API key in Settings first.")
                self._btn_train.setEnabled(True)
                self._btn_continue_train.setEnabled(True)
                self._btn_train.setText("Start Training")
                return

            dataset_mgr = DatasetManager(str(DATASETS_DIR))
            job_id = str(uuid.uuid4())[:8]

            self._worker = TrainingWorker(
                job_id=job_id,
                speaker_name=name,
                api_key=api_key,
                ssh_key_path=ssh_key,
                dataset_manager=dataset_mgr,
                models_dir=str(MODELS_DIR),
                f0_method=f0_method,
                resume_from=resume_from,
            )
            self._worker.log_line.connect(self._on_train_log)
            self._worker.status_changed.connect(lambda s: self._lbl_status.setText(s))
            self._worker.progress.connect(self._progress_bar.setValue)

            # Calculate recommended epochs
            self._recommended_epochs = 2000
            epochs_text = self._txt_epochs.text().strip()
            if epochs_text and epochs_text.lower() != "auto":
                try:
                    self._recommended_epochs = int(epochs_text)
                except ValueError:
                    pass
            else:
                try:
                    import soundfile as _sf
                    total_dur = sum(_sf.info(p).duration for p in self._clips)
                    if total_dur < 180:
                        self._recommended_epochs = 3000
                    elif total_dur < 300:
                        self._recommended_epochs = 2500
                    elif total_dur < 600:
                        self._recommended_epochs = 1500
                    elif total_dur < 1800:
                        self._recommended_epochs = 500
                    else:
                        self._recommended_epochs = 300
                except Exception:
                    pass
                # Show the calculated value in the epochs field
                self._txt_epochs.setText(str(self._recommended_epochs))
            self._current_epoch = 0
            self._worker.finished_ok.connect(self._on_train_done)
            self._worker.error.connect(self._on_train_error)
            self._progress_bar.setValue(0)
            self._progress_bar.setVisible(True)
            self._worker.start()
            self._training = True
            self._btn_stop_train.setVisible(True)
            self._btn_stop_train.setEnabled(True)
            self._btn_train.setVisible(False)
            self._btn_continue_train.setVisible(False)
        except Exception as e:
            self._on_train_error(str(e))

    def _stop_training(self):
        if self._worker and self._worker.isRunning():
            self._log.append_line("Stopping training — downloading model...")
            self._lbl_status.setText("Stopping & downloading model...")
            self._btn_stop_train.setEnabled(False)
            self._worker.request_stop()

    def _on_train_log(self, line):
        self._log.append_line(line)
        import re
        match = re.search(r'Epoch (\d+)/(\d+)', line)
        if match:
            self._current_epoch = int(match.group(1))
            rec = self._recommended_epochs
            if rec > 0 and self._current_epoch > 0:
                pct = min(int((self._current_epoch / rec) * 100), 100)
                self._progress_bar.setValue(pct)
                self._lbl_status.setText(f"Training... Epoch {self._current_epoch}/{rec}")
            # Auto-stop at target
            if self._current_epoch >= rec:
                if self._worker and self._worker.isRunning():
                    self._log.append_line(f"Reached target of {rec} epochs — stopping & downloading model...")
                    self._worker.request_stop()

    def _on_train_done(self, job_id):
        self._training = False
        self._btn_stop_train.setVisible(False)
        self._btn_train.setVisible(True)
        self._btn_train.setEnabled(True)
        self._btn_train.setText("Start Training")
        self._check_existing_model(self._selected_name)
        self._progress_bar.setValue(100)
        self._lbl_status.setText("Training complete! Model is ready.")
        self._lbl_status.setStyleSheet("color: rgba(80, 200, 120, 150); font-size: 11px; background: transparent;")

    def _on_train_error(self, error):
        self._training = False
        self._btn_stop_train.setVisible(False)
        self._btn_train.setVisible(True)
        self._btn_train.setEnabled(True)
        self._btn_train.setText("Start Training")
        self._check_existing_model(self._selected_name)
        self._progress_bar.setVisible(False)
        self._lbl_status.setText(f"Error: {error}")
        self._lbl_status.setStyleSheet("color: rgba(255, 100, 100, 150); font-size: 11px; background: transparent;")


class SimplePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._original_source = ""
        self._section_cache = {}
        self._selected_model_idx = -1
        self._optimized_for_model = -1  # model idx the waveform was last optimized for
        self._model_card_widgets = []
        self._source_median_hz = 0
        self._model_center_hz = 0
        self._hf_models = []
        self._hf_loaded = False
        self._image_cache_dir = os.path.join(CACHE_DIR, "artist_thumbs")
        self._bg_pixmap = None
        self._bg_opacity = 0.35
        self._bg_cache = None       # pre-composited background
        self._bg_cache_size = None   # (w, h) the cache was built for
        self._init_ui()
        # Create model panel (overlay, hidden by default)
        self._create_panel = _CreateModelPanel(self)
        self._create_panel.back_clicked.connect(self._hide_create_model)
        self._create_panel.setVisible(False)
        # Restore last session after a delay (let layout settle)
        QTimer.singleShot(200, self.restore_session)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 25, 24, 20)

        # ===== SEARCH BAR =====
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search artists...")
        self._search.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 5);
                color: #ccc;
                border: 1px solid rgba(255, 255, 255, 8);
                border-radius: 16px;
                padding: 8px 18px;
                font-size: 13px;
            }
            QLineEdit:focus {
                background-color: rgba(255, 255, 255, 10);
                border-color: rgba(255, 255, 255, 20);
                color: #eee;
            }
        """)
        self._search.setFixedHeight(36)
        self._search.setMaximumWidth(320)
        self._search.textChanged.connect(self._on_search_changed)
        self._search.mousePressEvent = self._on_search_clicked
        # Forward spacebar to page when search is empty (for waveform play/pause)
        _orig_search_key = self._search.keyPressEvent
        def _search_key_handler(event):
            if event.key() == Qt.Key.Key_Space and not self._search.text():
                self._search.clearFocus()
                self.keyPressEvent(event)
                return
            _orig_search_key(event)
        self._search.keyPressEvent = _search_key_handler

        search_row = QHBoxLayout()
        search_row.addWidget(self._search)

        self._btn_create_model = QLabel("+ Create a Model")
        self._btn_create_model.setStyleSheet("color: #fff; font-size: 11px; background: transparent;")
        self._btn_create_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_create_model.mousePressEvent = lambda e: self._show_create_model()
        search_row.addWidget(self._btn_create_model)

        search_row.addStretch()

        self._spinner_label = QLabel("")
        self._spinner_label.setStyleSheet("color: #c9a84c; font-size: 11px; background: transparent;")
        from PyQt6.QtWidgets import QGraphicsDropShadowEffect
        glow = QGraphicsDropShadowEffect()
        glow.setColor(QColor(201, 168, 76, 160))
        glow.setBlurRadius(12)
        glow.setOffset(0, 0)
        self._spinner_label.setGraphicsEffect(glow)
        self._spinner_label.setVisible(False)
        search_row.addWidget(self._spinner_label)

        self._lbl_logo = QLabel("somersaudio")
        self._lbl_logo.setStyleSheet("color: rgba(255, 255, 255, 80); font-size: 13px; font-weight: 600; background: transparent; letter-spacing: 1px;")
        search_row.addWidget(self._lbl_logo)

        layout.addLayout(search_row)

        # Dropdown (floating)
        self._dropdown = QListWidget(self)
        self._dropdown.setStyleSheet("""
            QListWidget {
                background-color: rgba(20, 20, 20, 230);
                border: 1px solid rgba(255, 255, 255, 15);
                border-radius: 10px;
                padding: 4px;
                font-size: 12px;
                color: #ddd;
            }
            QListWidget::item { padding: 6px 10px; border-radius: 6px; }
            QListWidget::item:selected { background-color: rgba(255, 255, 255, 15); }
            QListWidget::item:hover { background-color: rgba(255, 255, 255, 10); }
        """)
        self._dropdown.setVisible(False)
        self._dropdown.setMinimumHeight(350)
        self._dropdown.setMaximumHeight(450)
        self._dropdown.setIconSize(QSize(32, 32))
        self._dropdown.itemClicked.connect(self._on_dropdown_clicked)

        # ===== MODEL CAROUSEL =====
        self._carousel = _ModelCarousel()
        self._carousel.setFixedHeight(220)
        self._carousel.model_selected.connect(self._on_carousel_select)
        self._carousel.key_badge_clicked.connect(self._on_key_badge_clicked)
        layout.addWidget(self._carousel)

        # Selected model name
        self._lbl_model_name = QLabel("")
        self._lbl_model_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_model_name.setStyleSheet("color: #ccc; font-size: 14px; font-weight: bold; background: transparent; margin-top: -10px; padding: 0;")
        self._lbl_model_name.setFixedHeight(20)
        layout.addWidget(self._lbl_model_name)

        # Match Artist's Range toggle + detail
        range_row = QHBoxLayout()
        range_row.addStretch()

        self._btn_range_match = QPushButton()
        self._btn_range_match.setCheckable(True)
        self._btn_range_match.setChecked(True)
        self._btn_range_match.setFixedSize(36, 36)
        self._btn_range_match.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_range_match.setToolTip("Match Artist's Range")
        self._btn_range_match.setIcon(QIcon(self._make_wave_icon(True)))
        self._btn_range_match.setIconSize(QSize(22, 22))
        self._btn_range_match.toggled.connect(self._on_range_match_toggled)

        self._update_range_style()
        range_row.addWidget(self._btn_range_match)

        range_row.addSpacing(8)

        self._btn_realtime = QPushButton()
        self._btn_realtime.setCheckable(True)
        self._btn_realtime.setChecked(False)
        self._btn_realtime.setVisible(False)  # hidden on Best Match, shown on actual models
        self._btn_realtime.setFixedSize(36, 36)
        self._btn_realtime.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_realtime.setToolTip("Realtime Processing")
        self._btn_realtime.setIcon(QIcon(self._make_realtime_icon(False)))
        self._btn_realtime.setIconSize(QSize(22, 22))
        self._btn_realtime.toggled.connect(self._on_realtime_toggled)
        self._update_realtime_style()
        range_row.addWidget(self._btn_realtime)

        range_row.addStretch()
        layout.addLayout(range_row)

        # Hidden combo for data
        self._cmb_model = QComboBox()
        self._cmb_model.setVisible(False)
        layout.addWidget(self._cmb_model)

        # ===== DROP ZONE / SOURCE =====
        self._drop_zone = QLabel("Drop audio here or browse")
        self._drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_zone.setFixedHeight(40)
        self._drop_zone.setCursor(Qt.CursorShape.PointingHandCursor)
        self._drop_zone_default_style = """
            QLabel {
                background-color: rgba(255, 255, 255, 3);
                border: 1px dashed rgba(255, 255, 255, 15);
                border-radius: 12px;
                color: #666;
                font-size: 13px;
            }
            QLabel:hover {
                background-color: rgba(255, 255, 255, 6);
                border-color: rgba(255, 255, 255, 25);
                color: #888;
            }
        """
        self._drop_zone.setStyleSheet(self._drop_zone_default_style)
        self._drop_zone.mousePressEvent = lambda e: self._browse_source()
        self._btn_clear_source = QPushButton("×")
        self._btn_clear_source.setFixedSize(20, 20)
        self._btn_clear_source.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_clear_source.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 10);
                border: none;
                border-radius: 10px;
                color: #888;
                font-size: 14px;
                padding: 0;
            }
            QPushButton:hover { background: rgba(255, 255, 255, 20); color: #ccc; }
        """)
        self._btn_clear_source.clicked.connect(self._clear_source)
        self._btn_clear_source.setVisible(False)

        source_row = QHBoxLayout()
        source_row.addWidget(self._drop_zone, 1)
        source_row.addWidget(self._btn_clear_source)
        layout.addLayout(source_row)

        # ===== REALTIME KNOBS PANEL =====
        from ui.widgets.knob import Knob
        self._realtime_panel = QFrame()
        self._realtime_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 20, 128);
                border: 1px solid rgba(255, 255, 255, 10);
                border-radius: 12px;
            }
        """)
        self._realtime_panel.setVisible(False)
        knobs_layout = QHBoxLayout(self._realtime_panel)
        knobs_layout.setContentsMargins(12, 8, 12, 8)
        knobs_layout.setSpacing(0)

        self._rt_knob_pitch = Knob(
            label="Pitch", min_val=-24, max_val=24, default=0, step=1,
            suffix="st", decimals=0, compact=True,
        )
        self._rt_knob_tone = Knob(
            label="Tone", min_val=0.0, max_val=1.0, default=0.33, step=0.05,
            suffix="", decimals=2, compact=True,
        )
        self._rt_knob_response = Knob(
            label="Response", min_val=0.1, max_val=1.5, default=0.3, step=0.05,
            suffix="s", decimals=2, compact=True,
        )
        self._rt_knob_buffer = Knob(
            label="Buffer", min_val=0.1, max_val=1.5, default=0.3, step=0.05,
            suffix="s", decimals=2, compact=True,
        )
        self._rt_knob_smoothing = Knob(
            label="Smooth", min_val=0.0, max_val=0.15, default=0.012, step=0.005,
            suffix="s", decimals=3, compact=True,
        )
        self._rt_knob_gate = Knob(
            label="Gate", min_val=-80, max_val=0, default=-40, step=1,
            suffix="dB", decimals=0, compact=True,
        )
        for knob in [self._rt_knob_pitch, self._rt_knob_tone, self._rt_knob_response,
                      self._rt_knob_buffer, self._rt_knob_smoothing, self._rt_knob_gate]:
            knobs_layout.addWidget(knob)

        layout.addWidget(self._realtime_panel)

        # Enable drag & drop on the whole page
        self.setAcceptDrops(True)

        # Waveform visualizer with built-in playback
        # Analysis spinner state (animates drop zone border)
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self._spinner_active = False
        self._spinner_text = "analyzing"
        self._spinner_timer = QTimer()
        self._spinner_timer.setInterval(80)
        self._spinner_timer.timeout.connect(self._spin)
        self._waveform = _WaveformWidget()
        self._waveform.sections_changed.connect(self._on_sections_changed)
        self._waveform.interacted.connect(lambda: setattr(self, '_active_waveform', self._waveform))
        layout.addWidget(self._waveform)
        self._active_waveform = self._waveform  # default
        self._waveform_worker = None


        # Optimize button (appears below waveform when sections exist)
        self._btn_optimize = QPushButton("Optimal")
        self._btn_optimize.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_optimize.setVisible(False)
        self._btn_optimize.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 12);
                border-radius: 10px;
                color: #777;
                font-size: 10px;
                padding: 3px 12px;
            }
            QPushButton:hover {
                border-color: rgba(80, 200, 120, 0.4);
                color: #aaa;
            }
        """)
        self._btn_optimize.clicked.connect(self._optimize_sections)
        opt_row = QHBoxLayout()
        opt_row.addStretch()
        opt_row.addWidget(self._btn_optimize)
        opt_row.addStretch()
        layout.addLayout(opt_row)

        # Pitch info
        self._lbl_pitch = QLabel("")
        self._lbl_pitch.setStyleSheet("color: #888; font-size: 11px; background: transparent;")
        self._lbl_pitch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_pitch)

        # ===== CONVERT BUTTON (circle with progress ring) =====
        self._convert_ring = _ConvertButton()
        self._convert_ring.clicked.connect(self._convert)
        convert_row = QHBoxLayout()
        convert_row.addStretch()
        convert_row.addWidget(self._convert_ring)
        convert_row.addStretch()
        layout.addLayout(convert_row)
        self._progress_value = 0.0
        self._total_chunks = 1
        self._chunks_done = 0

        layout.addStretch()

        # ===== BOTTOM AREA (floating overlay, doesn't affect layout) =====
        self._bottom_panel = QWidget(self)
        self._bottom_panel.setStyleSheet("background: transparent;")
        bottom_layout = QVBoxLayout(self._bottom_panel)
        bottom_layout.setContentsMargins(24, 0, 24, 16)
        bottom_layout.setSpacing(4)

        self._log = LogViewer()
        self._log.setMaximumHeight(60)
        self._log.setVisible(False)
        bottom_layout.addWidget(self._log)

        self._lbl_output_name = QLabel("")
        self._lbl_output_name.setStyleSheet("color: rgba(255,255,255,40); font-size: 9px; background: transparent;")
        self._lbl_output_name.setFixedHeight(14)
        self._lbl_output_name.setVisible(False)
        bottom_layout.addWidget(self._lbl_output_name)

        self._waveform_output = _WaveformWidget(readonly=True)
        self._waveform_output.setVisible(False)
        self._waveform_output.interacted.connect(lambda: setattr(self, '_active_waveform', self._waveform_output))
        bottom_layout.addWidget(self._waveform_output)

        folder_row = QHBoxLayout()
        folder_row.setContentsMargins(0, 0, 0, 0)
        folder_row.setSpacing(0)
        folder_row.addStretch()

        self._btn_set_folder = QLabel("Set Output Folder")
        self._btn_set_folder.setStyleSheet("color: #555; font-size: 10px; background: transparent;")
        self._btn_set_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_set_folder.mousePressEvent = self._set_output_folder
        folder_row.addWidget(self._btn_set_folder)

        dot = QLabel(" · ")
        dot.setStyleSheet("color: #444; font-size: 10px; background: transparent;")
        folder_row.addWidget(dot)

        self._btn_folder = QLabel("Open Output Folder")
        self._btn_folder.setStyleSheet("color: #555; font-size: 10px; background: transparent;")
        self._btn_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_folder.mousePressEvent = self._open_output_folder
        folder_row.addWidget(self._btn_folder)

        folder_row.addStretch()
        bottom_layout.addLayout(folder_row)

        self._bottom_panel.adjustSize()

        self._source_path = ""
        self._refresh_models()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_bottom_panel()
        if self._create_panel.isVisible():
            self._create_panel.setGeometry(self.rect())

    def _position_bottom_panel(self):
        self._bottom_panel.setFixedWidth(self.width())
        self._bottom_panel.adjustSize()
        self._bottom_panel.move(0, self.height() - self._bottom_panel.height())
        self._bottom_panel.raise_()

    def _build_bg_cache(self):
        """Pre-composite the background image with gradients into a single pixmap."""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        from PyQt6.QtGui import QLinearGradient
        cache = QPixmap(w, h)
        cache.fill(QColor("#1a1a1a"))

        if self._bg_pixmap and not self._bg_pixmap.isNull():
            p = QPainter(cache)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            scaled = self._bg_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x_off = (scaled.width() - w) // 2
            y_off = (scaled.height() - h) // 2
            cropped = scaled.copy(x_off, y_off, w, h)

            p.setOpacity(self._bg_opacity)
            p.drawPixmap(0, 0, cropped)
            p.setOpacity(1.0)

            grad = QLinearGradient(0, 0, 0, h * 0.6)
            grad.setColorAt(0.0, QColor(26, 26, 26, 0))
            grad.setColorAt(0.5, QColor(26, 26, 26, 80))
            grad.setColorAt(1.0, QColor(26, 26, 26, 255))
            p.fillRect(0, 0, w, int(h * 0.6), grad)

            p.fillRect(0, int(h * 0.6), w, h - int(h * 0.6), QColor("#1a1a1a"))

            side_w = int(w * 0.25)
            grad_left = QLinearGradient(0, 0, side_w, 0)
            grad_left.setColorAt(0.0, QColor(26, 26, 26, 255))
            grad_left.setColorAt(1.0, QColor(26, 26, 26, 0))
            p.fillRect(0, 0, side_w, int(h * 0.6), grad_left)

            grad_right = QLinearGradient(w - side_w, 0, w, 0)
            grad_right.setColorAt(0.0, QColor(26, 26, 26, 0))
            grad_right.setColorAt(1.0, QColor(26, 26, 26, 255))
            p.fillRect(w - side_w, 0, side_w, int(h * 0.6), grad_right)

            p.end()

        self._bg_cache = cache
        self._bg_cache_size = (w, h)

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        if self._bg_cache is None or self._bg_cache_size != (w, h):
            self._build_bg_cache()

        p = QPainter(self)
        if self._bg_cache:
            p.drawPixmap(0, 0, self._bg_cache)
        else:
            p.fillRect(self.rect(), QColor("#1a1a1a"))
        p.end()

    def _update_background(self, model_dir):
        """Load the artist image as a wallpaper background."""
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            p = os.path.join(model_dir, f"image{ext}")
            if os.path.exists(p):
                self._bg_pixmap = QPixmap(p)
                self._bg_cache = None  # invalidate
                self.update()
                return
        self._bg_pixmap = None
        self._bg_cache = None
        self.update()

    # ===== MODEL CARDS =====

    def _refresh_models(self):
        self._cmb_model.clear()
        models = []

        # "Best Match" placeholder as first entry
        best_match_icon = os.path.join(APP_DIR, "assets", "best_match.png")
        best_pixmap = QPixmap(best_match_icon) if os.path.exists(best_match_icon) else None
        self._cmb_model.addItem("Best Match", "")
        models.append({"name": "Best Match", "dir": "", "pixmap": best_pixmap, "vocal_key": ""})

        if os.path.exists(MODELS_DIR):
            for name in sorted(os.listdir(MODELS_DIR)):
                model_dir = os.path.join(MODELS_DIR, name)
                if not os.path.isdir(model_dir):
                    continue
                mt = detect_model_type(model_dir)
                if mt == "svc":
                    if not any(f.startswith("G_") and f.endswith(".pth") for f in os.listdir(model_dir)):
                        continue
                elif mt != "rvc":
                    continue

                self._cmb_model.addItem(name, model_dir)

                # Load image
                pixmap = None
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    p = os.path.join(model_dir, f"image{ext}")
                    if os.path.exists(p):
                        pixmap = QPixmap(p)
                        break

                # Load vocal key from metadata
                vocal_key = ""
                meta_path = os.path.join(model_dir, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        import json as _json
                        with open(meta_path) as _f:
                            _meta = _json.load(_f)
                        vocal_key = _meta.get("vocal_key", "")
                    except Exception:
                        pass

                models.append({"name": name, "dir": model_dir, "pixmap": pixmap, "vocal_key": vocal_key})

        self._carousel.set_models(models)
        if models:
            self._carousel.select(0)
            self._lbl_model_name.setText("Best Match")
            best_bg = os.path.join(APP_DIR, "assets", "best_match.png")
            if os.path.exists(best_bg):
                self._bg_pixmap = QPixmap(best_bg)
                self._bg_cache = None
            self._resolve_best_match()

    def _find_best_match(self):
        """Find the model whose vocal key is closest to the source audio's median pitch."""
        if self._source_median_hz <= 0:
            return -1  # no source to match against

        import math
        best_idx = -1
        best_dist = float("inf")

        for i in range(1, self._cmb_model.count()):  # skip index 0 (Best Match)
            model = self._carousel._models[i]
            key = model.get("vocal_key", "")
            if not key or key == "Auto":
                continue
            hz = _note_to_hz(key)
            if hz <= 0:
                continue
            dist = abs(12 * math.log2(hz / self._source_median_hz))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def _resolve_best_match(self):
        """If 'Best Match' is selected, use the best model's data for conversion."""
        if self._selected_model_idx != 0:
            return
        best = self._find_best_match()
        if best > 0:
            model = self._carousel._models[best]
            self._cmb_model.setCurrentIndex(best)
            self._lbl_model_name.setText(model['name'])
            # Show artist image
            px = model.get("pixmap")
            if px and not px.isNull():
                from ui.widgets.voice_card import VoiceCard
                self._carousel._best_match_pixmap = VoiceCard._make_circular(px, 48)
            else:
                self._carousel._best_match_pixmap = None
            self._carousel.update()
            self._detect_model_key()
        else:
            self._lbl_model_name.setText("Best Match")
            self._carousel._best_match_pixmap = None
            self._carousel.update()
            if self._cmb_model.count() > 1:
                self._cmb_model.setCurrentIndex(1)
                self._detect_model_key()

    def select_model_by_name(self, name):
        """Refresh and select a model by artist name."""
        self._refresh_models()
        for i in range(self._cmb_model.count()):
            if self._cmb_model.itemText(i) == name:
                self._carousel.select(i)
                return

    def _on_carousel_select(self, idx):
        self._selected_model_idx = idx
        self._btn_realtime.setVisible(idx != 0)
        if idx == 0:
            self._realtime_panel.setVisible(False)
            self._drop_zone.setVisible(True)
            # "Best Match" selected — resolve to actual best model
            self._lbl_model_name.setText("Best Match")
            self._cmb_model.setCurrentIndex(0)
            best_bg = os.path.join(APP_DIR, "assets", "best_match.png")
            if os.path.exists(best_bg):
                self._bg_pixmap = QPixmap(best_bg)
            else:
                self._bg_pixmap = None
            self._bg_cache = None
            self.update()
            self._resolve_best_match()
            self._section_cache = {}
            return
        self._section_cache = {}
        self._carousel._best_match_pixmap = None
        # Restore realtime panel if toggled on
        if self._btn_realtime.isChecked():
            self._realtime_panel.setVisible(True)
            self._drop_zone.setVisible(False)
        self._cmb_model.setCurrentIndex(idx)
        if idx < self._cmb_model.count():
            self._lbl_model_name.setText(self._cmb_model.itemText(idx))
            model_dir = self._cmb_model.itemData(idx)
            if model_dir:
                self._update_background(model_dir)
        self._detect_model_key()

    def _detect_model_key(self):
        model_dir = self._cmb_model.currentData()
        if not model_dir:
            return

        # Check saved key
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            import json
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                key = meta.get("vocal_key", "")
                if key and key != "Auto":
                    self._model_center_hz = _note_to_hz(key)
                    self._update_transpose_info()
                    return
            except Exception:
                pass

        # Try dataset clips
        name = self._cmb_model.currentText()
        dataset_dir = os.path.join(DATASETS_DIR, name)
        if os.path.isdir(dataset_dir):
            clips = [os.path.join(dataset_dir, f) for f in sorted(os.listdir(dataset_dir))
                     if f.endswith((".wav", ".flac", ".mp3"))][:20]
            if clips:
                self._key_worker = _KeyDetectWorker(clips)
                self._key_worker.result.connect(self._on_key_detected)
                self._key_worker.start()
                return

        # Fallback: estimate from API (GetSongBPM → Spotify genre guess)
        self._estimate_key_from_api(name)

    def _on_key_badge_clicked(self, idx):
        """User clicked the '?' badge — run API key lookup for this model."""
        if idx < self._cmb_model.count():
            name = self._cmb_model.itemText(idx)
            self._estimate_key_from_api(name)

    def _on_key_detected(self, note, hz):
        if hz > 0:
            self._model_center_hz = hz
            # Save to metadata
            model_dir = self._cmb_model.currentData()
            if model_dir:
                import json
                meta_path = os.path.join(model_dir, "metadata.json")
                meta = {}
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                meta["vocal_key"] = note
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            # Update the carousel badge
            self._carousel.set_vocal_key(self._cmb_model.currentIndex(), note)
            self._update_transpose_info()

    def _estimate_key_from_api(self, artist_name):
        """Estimate vocal key via Spotify track names + GetSongBPM keys."""
        class _KeyLookupWorker(QThread):
            result = pyqtSignal(str, float)
            def __init__(self, name):
                super().__init__()
                self.name = name
            def run(self):
                try:
                    from services.songbpm_client import estimate_artist_key
                    note, hz = estimate_artist_key(self.name)
                    if note and hz > 0:
                        self.result.emit(note, hz)
                        return
                except Exception:
                    pass
                self.result.emit("", 0)

        self._key_lookup_worker = _KeyLookupWorker(artist_name)
        self._key_lookup_worker.result.connect(self._on_key_detected)
        self._key_lookup_worker.start()

    # ===== SOURCE =====

    def _make_wave_icon(self, active: bool) -> QPixmap:
        """Paint two sine waves — apart when off, merged when on."""
        import math
        size = 44
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = size / 2, size / 2
        margin = 6
        w = size - margin * 2

        if active:
            # Merged: single bright wave
            pen = QPen(QColor("#93b4f5"))
            pen.setWidthF(2.0)
            p.setPen(pen)
            path = QPainterPath()
            for i in range(w + 1):
                x = margin + i
                y = cy + 6 * math.sin(2 * math.pi * i / w * 2)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.drawPath(path)
        else:
            # Two separate waves offset vertically
            for offset, color in [(-4, "#888"), (4, "#666")]:
                pen = QPen(QColor(color))
                pen.setWidthF(1.5)
                p.setPen(pen)
                path = QPainterPath()
                for i in range(w + 1):
                    x = margin + i
                    y = cy + offset + 5 * math.sin(2 * math.pi * i / w * 2)
                    if i == 0:
                        path.moveTo(x, y)
                    else:
                        path.lineTo(x, y)
                p.drawPath(path)

        p.end()
        return pm

    def _update_range_style(self):
        if self._btn_range_match.isChecked():
            self._btn_range_match.setStyleSheet("""
                QPushButton {
                    background: rgba(37, 99, 235, 60);
                    border: 1px solid rgba(100, 160, 255, 120);
                    border-radius: 18px;
                }
                QPushButton:hover {
                    background: rgba(37, 99, 235, 80);
                    border-color: rgba(100, 160, 255, 160);
                }
            """)
        else:
            self._btn_range_match.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: 1px solid rgba(255, 255, 255, 15);
                    border-radius: 18px;
                }
                QPushButton:hover {
                    border-color: rgba(255, 255, 255, 30);
                    background: rgba(255, 255, 255, 5);
                }
            """)

    def _on_range_match_toggled(self, checked):
        self._btn_range_match.setIcon(QIcon(self._make_wave_icon(checked)))
        self._update_range_style()
        # Disable realtime while range match is active
        self._btn_realtime.setEnabled(not checked)
        if checked and self._source_path and os.path.exists(self._source_path):
            if self._lbl_pitch.text():
                self._lbl_pitch.setVisible(True)
            if self._waveform._samples:
                # Already have data — just show it
                self._waveform.setVisible(True)
                self._btn_optimize.setVisible(len(self._waveform._sections) > 1)
            else:
                self._analyze_waveform()
        elif not checked:
            self._waveform.hide_and_stop()
            self._btn_optimize.setVisible(False)
            self._lbl_pitch.setVisible(False)

    def _make_realtime_icon(self, active: bool) -> QPixmap:
        """Paint a clock with a lightning bolt inside."""
        import math
        size = 44
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = size / 2, size / 2
        r = 15  # clock radius

        # Clock circle
        color = QColor("#e8c94a") if active else QColor("#888")
        pen = QPen(color, 2.0)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        # Hour ticks (12, 3, 6, 9 o'clock)
        tick_color = QColor(232, 201, 74, 180) if active else QColor("#777")
        tick_pen = QPen(tick_color, 1.5)
        p.setPen(tick_pen)
        for hour in [0, 3, 6, 9]:
            angle = math.radians(hour * 30 - 90)
            inner = r - 3
            outer = r - 1
            p.drawLine(
                int(cx + inner * math.cos(angle)), int(cy + inner * math.sin(angle)),
                int(cx + outer * math.cos(angle)), int(cy + outer * math.sin(angle)),
            )

        # Lightning bolt in center
        bolt_color = QColor("#e8c94a") if active else QColor("#999")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bolt_color)
        bolt = QPainterPath()
        # Stylized bolt shape
        bolt.moveTo(cx + 1, cy - 8)
        bolt.lineTo(cx - 3, cy + 1)
        bolt.lineTo(cx, cy + 1)
        bolt.lineTo(cx - 1, cy + 8)
        bolt.lineTo(cx + 3, cy - 1)
        bolt.lineTo(cx, cy - 1)
        bolt.closeSubpath()
        p.drawPath(bolt)

        p.end()
        return pm

    def _update_realtime_style(self):
        if self._btn_realtime.isChecked():
            self._btn_realtime.setStyleSheet("""
                QPushButton {
                    background: rgba(232, 201, 74, 40);
                    border: 1px solid rgba(232, 201, 74, 100);
                    border-radius: 18px;
                }
                QPushButton:hover {
                    background: rgba(232, 201, 74, 60);
                    border-color: rgba(232, 201, 74, 140);
                }
            """)
        else:
            self._btn_realtime.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: 1px solid rgba(255, 255, 255, 15);
                    border-radius: 18px;
                }
                QPushButton:hover {
                    border-color: rgba(255, 255, 255, 30);
                    background: rgba(255, 255, 255, 5);
                }
            """)

    def _on_realtime_toggled(self, checked):
        self._btn_realtime.setIcon(QIcon(self._make_realtime_icon(checked)))
        self._update_realtime_style()
        self._realtime_panel.setVisible(checked)
        self._drop_zone.setVisible(not checked)
        self._btn_clear_source.setVisible(not checked and bool(self._source_path))
        self._convert_ring.set_realtime_mode(checked)
        # Disable range match while realtime is active
        self._btn_range_match.setEnabled(not checked)

    def _toggle_realtime(self):
        """Start or stop realtime voice conversion."""
        import subprocess as _sp
        import sys

        if self._convert_ring._converting:
            # Stop realtime
            if hasattr(self, '_rt_process') and self._rt_process:
                self._rt_process.terminate()
                self._rt_process.wait(2)
                self._rt_process = None
            if hasattr(self, '_rt_read_timer') and self._rt_read_timer:
                self._rt_read_timer.stop()
            self._convert_ring.set_converting(False)
            self._hide_spinner()
            return

        # Start realtime
        if self._cmb_model.count() == 0 or not self._cmb_model.currentData():
            QMessageBox.warning(self, "No Model", "Select a model first.")
            return

        model_dir = self._cmb_model.currentData()
        mt = detect_model_type(model_dir)

        if mt == "rvc":
            # RVC doesn't support svc vc realtime — notify user
            QMessageBox.warning(self, "Not Supported", "Realtime mode requires an SVC model.")
            return

        g_files = sorted(
            [f for f in os.listdir(model_dir) if f.startswith("G_") and f.endswith(".pth")],
            key=lambda f: int(f.replace("G_", "").replace(".pth", "")) if f.replace("G_", "").replace(".pth", "").isdigit() else 0,
        )
        if not g_files:
            QMessageBox.warning(self, "No Model", "No checkpoint found.")
            return

        model_path = os.path.join(model_dir, g_files[-1])
        config_path = os.path.join(model_dir, "config.json")
        svc_bin = os.path.join(os.path.dirname(sys.executable), "svc")

        cmd = [
            svc_bin, "vc",
            "-m", model_path,
            "-c", config_path,
            "-s", "0",
            "-t", str(int(self._rt_knob_pitch.value)),
            "-fm", "dio",
            "-n", str(self._rt_knob_tone.value),
            "-db", str(int(self._rt_knob_gate.value)),
            "-ch", str(self._rt_knob_response.value),
            "-b", str(self._rt_knob_buffer.value),
            "-cr", str(self._rt_knob_smoothing.value),
            "-d", "mps",
            "-a",
        ]

        try:
            self._rt_process = _sp.Popen(
                cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT,
                text=True, bufsize=1,
            )
            self._convert_ring.set_converting(True)
            self._show_spinner("realtime")

            self._rt_read_timer = QTimer()
            self._rt_read_timer.setInterval(100)
            self._rt_read_timer.timeout.connect(self._read_rt_output)
            self._rt_read_timer.start()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _read_rt_output(self):
        """Poll realtime process for output."""
        if not hasattr(self, '_rt_process') or not self._rt_process:
            return
        ret = self._rt_process.poll()
        if ret is not None:
            # Process ended
            self._rt_process = None
            self._rt_read_timer.stop()
            self._convert_ring.set_converting(False)
            self._hide_spinner()
            return
        # Read available output (non-blocking)
        import select
        while select.select([self._rt_process.stdout], [], [], 0)[0]:
            line = self._rt_process.stdout.readline()
            if not line:
                break

    def _clear_source(self):
        self._source_path = ""
        self._original_source = ""
        self._source_median_hz = 0
        self._section_cache = {}
        # Clean up section cache files
        cache_dir = os.path.join(OUTPUT_DIR, ".section_cache")
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
        self._drop_zone.setText("Drop audio here or browse")
        self._drop_zone.setStyleSheet(self._drop_zone_default_style)
        self._btn_clear_source.setVisible(False)
        self._btn_optimize.setVisible(False)
        self._hide_spinner()
        self._waveform.clear()
        self._lbl_pitch.setText("")
        self._waveform_output.hide_and_stop()
        self._lbl_output_name.setVisible(False)
        self._convert_ring.set_update_mode(False)
        self._log.setVisible(False)
        QTimer.singleShot(50, self._position_bottom_panel)

    def _browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio", "",
            "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if path:
            self._load_source(path)

    def _on_pitch_result(self, text, median_hz):
        self._lbl_pitch.setText(text)
        if median_hz > 0:
            self._source_median_hz = median_hz
            # If "Best Match" is selected, re-evaluate with new pitch info
            if self._selected_model_idx == 0:
                self._resolve_best_match()
            else:
                self._update_transpose_info()
        # Hide spinner if no waveform analysis will follow
        if not self._btn_range_match.isChecked():
            self._hide_spinner()
        # Show waveform even without model center (just splits, no transpose colors)
        if not self._waveform.isVisible() and self._source_path:
            self._analyze_waveform()

    def _update_transpose_info(self):
        if self._source_median_hz > 0 and self._model_center_hz > 0:
            import math
            semitones = round(12 * math.log2(self._model_center_hz / self._source_median_hz))
            model_note = _hz_to_note(self._model_center_hz)
            current = self._lbl_pitch.text().split("→")[0].strip()
            self._lbl_pitch.setText(f"{current}  →  Transpose: {semitones:+d} (to {model_note})")
            # Update waveform colors without re-analyzing
            if self._waveform._samples:
                self._waveform._model_center_hz = self._model_center_hz
                self._waveform._invalidate_wave_cache()
            elif self._source_path:
                self._analyze_waveform()
        # Toggle optimize button state based on whether current model matches optimized model
        if self._waveform._samples and self._btn_optimize.isVisible():
            if self._optimized_for_model >= 0 and self._selected_model_idx == self._optimized_for_model:
                self._mark_optimize_clean()
            else:
                self._mark_optimize_dirty()

    def _spin(self):
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        frame = self._spinner_frames[self._spinner_idx]
        self._spinner_label.setText(f"{frame} {self._spinner_text} {frame}")

    def _show_spinner(self, text="analyzing"):
        self._spinner_text = text
        self._spinner_active = True
        self._spinner_label.setVisible(True)
        self._spinner_timer.start()

    def _hide_spinner(self):
        self._spinner_active = False
        self._spinner_label.setVisible(False)
        self._spinner_timer.stop()

    def _analyze_waveform(self):
        if not self._source_path or not os.path.exists(self._source_path):
            self._waveform.clear()
            self._hide_spinner()
            return
        if not self._btn_range_match.isChecked():
            self._waveform.clear()
            self._hide_spinner()
            return
        self._show_spinner("analyzing")
        # Disconnect old worker — don't wait
        if self._waveform_worker is not None:
            try:
                self._waveform_worker.disconnect()
            except Exception:
                pass
            if self._waveform_worker.isRunning():
                if not hasattr(self, '_old_workers'):
                    self._old_workers = []
                self._old_workers.append(self._waveform_worker)
        self._waveform_worker = _WaveformAnalyzer(self._source_path, self._model_center_hz)
        self._waveform_worker.finished.connect(self._on_waveform_ready)
        self._waveform_worker.start()

    def _on_waveform_ready(self, samples, sections, transposes, median_hz):
        self._hide_spinner()
        # Only apply if this is still the active worker
        if samples:
            self._waveform.set_data(samples, sections, transposes, median_hz, self._model_center_hz)
            self._btn_optimize.setVisible(len(sections) > 1)
            self._optimized_for_model = self._selected_model_idx
            self._mark_optimize_clean()

    def _optimize_sections(self):
        """For each section, try ±12, ±24, ±36 from current transpose and pick the octave closest to model center."""
        if not self._waveform._sections or not self._waveform._transposes:
            return
        if self._model_center_hz <= 0:
            QMessageBox.information(self, "No Model Key", "Select a model with a detected vocal key first.")
            return
        if not self._source_path:
            return

        self._btn_optimize.setText("Optimizing...")
        self._btn_optimize.setEnabled(False)

        try:
            import soundfile as _sf
            info = _sf.info(self._source_path)
            duration = info.duration
        except Exception:
            self._btn_optimize.setText("Optimal")
            self._btn_optimize.setEnabled(True)
            return

        sections = self._waveform._sections
        abs_sections = [(s * duration, e * duration) for s, e in sections]
        transposes = list(self._waveform._transposes)

        class _OptWorker(QThread):
            result = pyqtSignal(list, int)
            def __init__(self, path, secs, current_transposes, center_hz):
                super().__init__()
                self.path = path
                self.secs = secs
                self.transposes = current_transposes
                self.center = center_hz
            def run(self):
                import tempfile, shutil, math
                import numpy as np
                from services.section_splitter import split_audio_file, analyze_section_pitches
                tmp_dir = tempfile.mkdtemp(prefix="svc_opt_")
                try:
                    section_paths = split_audio_file(self.path, self.secs, tmp_dir)
                    pitched = analyze_section_pitches(section_paths)
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                # Find base transpose from overall median (same as section_splitter)
                voiced = [p.get("median_hz", 0) for p in pitched if p.get("median_hz", 0) > 0]
                if voiced:
                    overall_median = float(np.median(voiced))
                    base_transpose = round(12 * math.log2(self.center / overall_median))
                else:
                    base_transpose = 0

                improved = 0
                new_t = list(self.transposes)
                for i, p in enumerate(pitched):
                    if i >= len(new_t):
                        break
                    median_hz = p.get("median_hz", 0)
                    if median_hz <= 0:
                        continue
                    current = new_t[i]
                    # Only allow base ± octaves to stay in key
                    candidates = [base_transpose + offset for offset in [0, -12, 12, -24, 24]]
                    best = current
                    best_dist = 999.0
                    for candidate in candidates:
                        test_hz = median_hz * (2 ** (candidate / 12))
                        if test_hz <= 0:
                            continue
                        dist = abs(12 * math.log2(test_hz / self.center))
                        if dist < best_dist or (dist == best_dist and abs(candidate) < abs(best)):
                            best_dist = dist
                            best = candidate
                    if best != current:
                        new_t[i] = best
                        improved += 1
                self.result.emit(new_t, improved)

        self._opt_worker = _OptWorker(self._source_path, abs_sections, transposes, self._model_center_hz)
        self._opt_worker.result.connect(self._on_optimize_done)
        self._opt_worker.start()

    def _on_optimize_done(self, transposes, improved):
        self._waveform._transposes = transposes
        self._waveform._invalidate_wave_cache()
        self._btn_optimize.setEnabled(True)
        self._btn_optimize.setText(f"Optimized {improved} section{'s' if improved != 1 else ''}" if improved > 0 else "Already optimal")
        self._optimized_for_model = self._selected_model_idx
        self._mark_optimize_clean()
        QTimer.singleShot(2000, lambda: self._btn_optimize.setText("Optimal"))

    def _mark_optimize_dirty(self):
        self._btn_optimize.setText("Optimize")
        self._btn_optimize.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 10px;
                color: #ddd;
                font-size: 10px;
                padding: 3px 12px;
            }
            QPushButton:hover {
                border-color: rgba(80, 200, 120, 0.5);
                color: #fff;
            }
        """)

    def _mark_optimize_clean(self):
        self._btn_optimize.setText("Optimal")
        self._btn_optimize.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 12);
                border-radius: 10px;
                color: #777;
                font-size: 10px;
                padding: 3px 12px;
            }
            QPushButton:hover {
                border-color: rgba(80, 200, 120, 0.4);
                color: #aaa;
            }
        """)

    def _on_sections_changed(self):
        """User dragged a split marker — just invalidate the cache to redraw with new boundaries."""
        self._waveform._invalidate_wave_cache()
        self._mark_optimize_dirty()
        # Show "Update" on convert button if there's already an output
        if self._waveform_output.isVisible():
            self._convert_ring.set_update_mode(True)

    def _get_custom_sections(self):
        """Convert waveform's normalized sections back to absolute seconds."""
        if not self._btn_range_match.isChecked():
            return None
        sections = self._waveform._sections
        if not sections or len(sections) <= 1:
            return None
        try:
            import soundfile as _sf
            info = _sf.info(self._source_path)
            duration = info.duration
        except Exception:
            return None
        return [(s * duration, e * duration) for s, e in sections]

    # ===== CONVERT =====

    def _convert(self):
        # Realtime mode
        if self._btn_realtime.isChecked():
            self._toggle_realtime()
            return

        # If already converting, cancel
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
            self._worker = None
            self._convert_ring.set_converting(False)
            self._waveform.set_active_section(-1)
            self._waveform.set_progress(0.0)
            self._log.append_line("Cancelled.")
            return

        if self._cmb_model.count() == 0 or not self._cmb_model.currentData():
            QMessageBox.warning(self, "No Model", "Select a model first.")
            return
        if not self._source_path or not os.path.exists(self._source_path):
            QMessageBox.warning(self, "No Audio", "Select an audio file first.")
            return

        model_dir = self._cmb_model.currentData()
        mt = detect_model_type(model_dir)

        if mt == "svc":
            g_files = sorted(
                [f for f in os.listdir(model_dir) if f.startswith("G_") and f.endswith(".pth")],
                key=lambda f: int(f.replace("G_", "").replace(".pth", "")) if f.replace("G_", "").replace(".pth", "").isdigit() else 0,
            )
            if not g_files:
                QMessageBox.warning(self, "No Model", "No checkpoint found.")
                return
            model_path = os.path.join(model_dir, g_files[-1])
            config_path = os.path.join(model_dir, "config.json")
        else:
            rvc_files = _get_rvc_pth_files(os.listdir(model_dir))
            if not rvc_files:
                QMessageBox.warning(self, "No Model", "No RVC model found.")
                return
            model_path = os.path.join(model_dir, rvc_files[0])
            config_path = ""

        # Calculate transpose (only when matching artist's range)
        transpose = 0
        if self._btn_range_match.isChecked() and self._source_median_hz > 0 and self._model_center_hz > 0:
            import math
            transpose = round(12 * math.log2(self._model_center_hz / self._source_median_hz))

        # Estimate total chunks from source duration
        import soundfile as _sf
        try:
            info = _sf.info(self._source_path)
            self._total_chunks = max(1, int(info.duration / 30.0) + 1)
        except Exception:
            self._total_chunks = 3
        self._chunks_done = 0

        # Merge adjacent GREEN sections with the same transpose (yellow/red stay split)
        if self._waveform._sections and self._waveform._transposes and len(self._waveform._sections) == len(self._waveform._transposes):
            merged_sections = [self._waveform._sections[0]]
            merged_transposes = [self._waveform._transposes[0]]
            merged_median = [self._waveform._median_hz[0]] if self._waveform._median_hz else [0]
            for i in range(1, len(self._waveform._sections)):
                same_t = self._waveform._transposes[i] == merged_transposes[-1]
                # Only merge if both sections are green (distance ≤ 3)
                prev_idx = len(merged_sections) - 1
                prev_green = self._waveform._section_distance(i - 1) <= 3
                curr_green = self._waveform._section_distance(i) <= 3
                if same_t and prev_green and curr_green:
                    merged_sections[-1] = (merged_sections[-1][0], self._waveform._sections[i][1])
                else:
                    merged_sections.append(self._waveform._sections[i])
                    merged_transposes.append(self._waveform._transposes[i])
                    merged_median.append(self._waveform._median_hz[i] if i < len(self._waveform._median_hz) else 0)
            if len(merged_sections) < len(self._waveform._sections):
                self._waveform._sections = merged_sections
                self._waveform._transposes = merged_transposes
                self._waveform._median_hz = merged_median
                self._waveform._section_info = []
                self._waveform._invalidate_wave_cache()

        self._convert_ring.set_update_mode(False)
        self._convert_ring.set_converting(True)
        self._convert_ring.set_progress(0.02)  # show it's started
        self._log.setVisible(False)
        self._log.clear_log()
        QTimer.singleShot(50, self._position_bottom_panel)

        self._worker = InferenceWorker(
            source_wav=self._source_path,
            model_path=model_path,
            config_path=config_path,
            output_dir=OUTPUT_DIR,
            transpose=transpose,
            f0_method="crepe",
            auto_predict_f0=False,
            noise_scale=0.1,
            db_thresh=-35,
            pad_seconds=1.0,
            chunk_seconds=30.0,
            separate_vocals=False,
            model_type=mt,
            model_dir=model_dir,
            smart_transpose=self._btn_range_match.isChecked() and self._model_center_hz > 0,
            model_center_hz=self._model_center_hz,
            max_sections=20,
            custom_sections=self._get_custom_sections(),
            custom_transposes=list(self._waveform._transposes) if self._btn_range_match.isChecked() and self._waveform._transposes else None,
            section_cache=getattr(self, '_section_cache', {}),
        )
        self._worker.log_line.connect(self._on_log_line)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.section_results.connect(self._on_section_results)
        self._worker.cache_ready.connect(self._on_cache_ready)
        self._worker.start()
        self._show_spinner("converting")

    def _on_log_line(self, line):
        self._log.append_line(line)
        # Track progress from inference log
        # In smart transpose mode, use section progress (not chunk counting)
        if "Separating vocals" in line or "Separating track" in line:
            self._convert_ring.set_progress(0.05)
        elif "Vocal separation complete" in line:
            self._convert_ring.set_progress(0.15)
        elif "Detecting sections" in line:
            self._convert_ring.set_progress(0.18)
        elif "Converting section" in line:
            import re
            m = re.search(r"section (\d+)/(\d+)", line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                progress = 0.2 + 0.7 * done / total
                self._convert_ring.set_progress(progress)
                self._waveform.set_active_section(done - 1)
                self._waveform.set_progress(progress)
        elif "Rejoining" in line or "Remixing" in line:
            self._convert_ring.set_progress(0.92)
        elif "Output saved" in line:
            self._convert_ring.set_progress(1.0)
        elif "Inference time:" in line and not self._btn_range_match.isChecked():
            # Only use chunk counting in non-smart-transpose mode
            self._chunks_done += 1
            progress = min(0.95, self._chunks_done / self._total_chunks)
            self._convert_ring.set_progress(progress)

    def _on_finished(self, output_path):
        self._hide_spinner()
        self._convert_ring.set_converting(False)
        self._waveform.set_active_section(-1)
        self._waveform.set_progress(0.0)
        self._log.append_line(f"Done!")

        # Load output into waveform player
        from pathlib import Path
        stem = Path(output_path).stem
        # Format: songname_artistname_N — split to color song teal, artist gold
        artist = os.path.basename(self._cmb_model.currentData()) if self._cmb_model.currentData() else ""
        if artist and f"_{artist}_" in stem:
            parts = stem.rsplit(f"_{artist}_", 1)
            song = parts[0]
            num = parts[1] if len(parts) > 1 else ""
            self._lbl_output_name.setText(
                f'<span style="color: rgba(94,200,180,150);">{song}</span>'
                f' · <span style="color: rgba(201,168,76,150);">{artist}</span>'
                f' <span style="color: rgba(255,255,255,40);">#{num}</span>'
            )
        else:
            self._lbl_output_name.setText(f'<span style="color: rgba(94,200,180,150);">{stem}</span>')
        self._lbl_output_name.setVisible(True)
        self._waveform_output.load(output_path)
        self._output_samples_worker = _WaveformSamplesOnly(output_path)
        self._output_samples_worker.finished.connect(self._on_output_samples)
        self._output_samples_worker.start()

    def _on_output_samples(self, samples):
        if samples:
            # Copy section layout from source waveform as a frozen snapshot
            sections = list(self._waveform._sections)
            transposes = list(self._waveform._transposes)
            median_hz = list(self._waveform._median_hz) if self._waveform._median_hz else [0] * len(sections)
            self._waveform_output.set_data(samples, sections, transposes, median_hz, self._model_center_hz)
            # Copy section info for proper coloring
            self._waveform_output._section_info = list(self._waveform._section_info)
            self._waveform_output._converted = self._waveform._converted
            self._waveform_output.setVisible(True)
            # If source waveform is hidden, make output the active one for spacebar
            if not self._waveform.isVisible():
                self._active_waveform = self._waveform_output
            QTimer.singleShot(50, self._position_bottom_panel)

    def _on_section_results(self, results):
        # Preserve user's transpose values — only add match info
        self._waveform._section_info = results
        self._waveform._converted = True
        self._waveform._invalidate_wave_cache()

    def _on_cache_ready(self, cache):
        self._section_cache = cache

    def _on_error(self, error):
        self._hide_spinner()
        self._convert_ring.set_converting(False)
        self._log.append_line(f"ERROR: {error}")
        QMessageBox.critical(self, "Error", f"Conversion failed:\n{error}")

    # ===== SEARCH / BROWSE =====

    def _on_search_clicked(self, event):
        QLineEdit.mousePressEvent(self._search, event)
        if not self._hf_loaded:
            self._load_hf()
        else:
            self._show_dropdown()

    def _load_hf(self):
        from services.hf_model_browser import fetch_available_models
        self._search.setPlaceholderText("Loading artists...")
        self._show_spinner("loading")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        try:
            self._hf_models = fetch_available_models()
            self._hf_loaded = True
            self._hide_spinner()
            self._search.setPlaceholderText("Search artists...")
            self._show_dropdown()
        except Exception as e:
            self._hide_spinner()
            self._search.setPlaceholderText("Search artists...")
            QMessageBox.warning(self, "Error", str(e))

    def _on_search_changed(self, text):
        if self._hf_loaded:
            self._show_dropdown(text)

    def _show_dropdown(self, ft="", show_all=False):
        ft = ft.lower()
        self._dropdown.clear()
        shown = 0
        max_items = 50 if (not ft and not show_all) else 9999
        has_more = False
        for m in self._hf_models:
            if ft and ft not in m["artist"].lower():
                continue
            if shown >= max_items:
                has_more = True
                break
            item = QListWidgetItem(m["artist"])
            item.setData(Qt.ItemDataRole.UserRole, m)

            thumb = os.path.join(self._image_cache_dir, f"{m['artist']}.jpg")
            if os.path.exists(thumb):
                px = QPixmap(thumb)
                if not px.isNull():
                    from ui.widgets.voice_card import VoiceCard
                    item.setIcon(QIcon(VoiceCard._make_circular(px, 28)))

            self._dropdown.addItem(item)
            shown += 1

        # "See all" option
        if has_more:
            see_all = QListWidgetItem("See all artists...")
            see_all.setData(Qt.ItemDataRole.UserRole, {"_see_all": True})
            see_all.setForeground(QColor(94, 200, 180, 180))
            self._dropdown.addItem(see_all)
            shown += 1

        if shown > 0:
            bar_pos = self._search.mapTo(self, QPoint(0, self._search.height()))
            w = min(self.width() - 40, 500)
            x = max(10, bar_pos.x() + self._search.width() // 2 - w // 2)
            self._dropdown.setFixedWidth(w)
            self._dropdown.move(x, bar_pos.y() + 4)
            self._dropdown.setVisible(True)
            self._dropdown.raise_()
        else:
            self._dropdown.setVisible(False)

    def _on_dropdown_clicked(self, item):
        model_info = item.data(Qt.ItemDataRole.UserRole)
        if not model_info:
            return
        # Handle "See all" click
        if model_info.get("_see_all"):
            self._show_dropdown(show_all=True)
            return

        # Remove any existing floating widgets (download button + badges)
        for child in self._dropdown.findChildren(QWidget, "dl_btn_float"):
            child.deleteLater()
        # If clicking the same row, just toggle off
        if getattr(self, '_dl_btn_item', None) is item:
            self._dl_btn_item = None
            return
        self._dl_btn_item = item

        # Show floating badges and download button over the clicked row
        rect = self._dropdown.visualItemRect(item)
        viewport_offset = self._dropdown.viewport().mapTo(self._dropdown, QPoint(0, 0))
        vy = rect.center().y() + viewport_offset.y()
        rx = rect.right() + viewport_offset.x()

        # Model type badge (SVC/RVC)
        model_type = model_info.get("type", "").upper()
        if model_type:
            type_lbl = QLabel(model_type, self._dropdown)
            type_lbl.setObjectName("dl_btn_float")
            type_lbl.setFixedSize(32, 18)
            type_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            type_color = "rgba(180, 130, 255, 130)" if model_type == "SVC" else "rgba(255, 180, 100, 130)"
            type_lbl.setStyleSheet(f"color: {type_color}; font-size: 9px; font-weight: bold; background: transparent;")
            type_lbl.move(rx - 210, vy - 9)
            type_lbl.show()

        # Source badge (HF)
        hf_lbl = QLabel("HF", self._dropdown)
        hf_lbl.setObjectName("dl_btn_float")
        hf_lbl.setFixedSize(22, 18)
        hf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hf_lbl.setStyleSheet("color: rgba(94, 200, 180, 130); font-size: 9px; font-weight: bold; background: transparent;")
        hf_lbl.move(rx - 175, vy - 9)
        hf_lbl.show()

        # Download button
        btn = QPushButton("Download", self._dropdown)
        btn.setObjectName("dl_btn_float")
        btn.setFixedSize(100, 26)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {
                background: rgba(94, 200, 180, 40);
                border: 1px solid rgba(94, 200, 180, 80);
                border-radius: 8px;
                color: rgba(94, 200, 180, 200);
                font-size: 11px;
                font-weight: bold;
                padding: 0 12px;
            }
            QPushButton:hover {
                background: rgba(94, 200, 180, 80);
                color: white;
            }
        """)
        btn.move(rx - 110, vy - 13)
        btn.clicked.connect(lambda _, i=item, b=btn: (self._download_from_dropdown(i), b.deleteLater()))
        btn.show()
        # Hide all floats on scroll
        def _on_scroll(_):
            for c in self._dropdown.findChildren(QWidget, "dl_btn_float"):
                try:
                    c.deleteLater()
                except RuntimeError:
                    pass
            try:
                self._dropdown.verticalScrollBar().valueChanged.disconnect(_on_scroll)
            except Exception:
                pass
            self._dl_btn_item = None
        self._dropdown.verticalScrollBar().valueChanged.connect(_on_scroll)

    def _download_from_dropdown(self, item):
        model_info = item.data(Qt.ItemDataRole.UserRole)
        if not model_info:
            return

        import shutil
        artist = model_info["artist"]
        folder = model_info["folder"]
        self._dropdown.setVisible(False)
        self._search.clear()

        dest = os.path.join(MODELS_DIR, artist)
        if os.path.exists(dest):
            reply = QMessageBox.question(
                self, "Exists", f"'{artist}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            shutil.rmtree(dest)

        self._search.setPlaceholderText(f"Downloading {artist}...")
        self._show_spinner("downloading")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from services.hf_model_browser import download_model
            download_model(folder, dest, on_log=lambda m: (
                self._search.setPlaceholderText(m), QApplication.processEvents()))

            from services.model_inspector import inspect_model
            import json
            meta = inspect_model(dest)
            meta["source"] = "downloaded"
            with open(os.path.join(dest, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # Fetch artist image: try Spotify first, fall back to cached thumbnail
            save_path = os.path.join(dest, "image.jpg")
            got_image = False
            try:
                from services.spotify_client import SpotifyClient
                spotify = SpotifyClient(
                    os.environ.get("SOMERSVC_SPOTIFY_ID", ""),
                    os.environ.get("SOMERSVC_SPOTIFY_SECRET", ""),
                )
                got_image = spotify.download_artist_image(artist, save_path)
            except Exception:
                pass

            if not got_image:
                # Fall back to cached dropdown thumbnail
                thumb = os.path.join(self._image_cache_dir, f"{artist}.jpg")
                if os.path.exists(thumb):
                    import shutil as _sh
                    _sh.copy2(thumb, save_path)

            self._hide_spinner()
            self._search.setPlaceholderText("Search artists...")
            self._refresh_models()
            QMessageBox.information(self, "Downloaded", f"'{artist}' ready!")
        except Exception as e:
            self._hide_spinner()
            self._search.setPlaceholderText("Search artists...")
            QMessageBox.warning(self, "Failed", str(e))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(u.toLocalFile().lower().endswith((".wav", ".flac", ".mp3", ".ogg")) for u in urls):
                event.acceptProposedAction()
                self._drop_zone.setStyleSheet("""
                    QLabel {
                        background-color: rgba(37, 99, 235, 12);
                        border: 2px dashed rgba(37, 99, 235, 40);
                        border-radius: 12px;
                        color: #aaa;
                        font-size: 13px;
                    }
                """)
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._drop_zone.setStyleSheet(self._drop_zone_default_style)

    def dropEvent(self, event):
        self._drop_zone.setStyleSheet(self._drop_zone_default_style)
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if path.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                self._load_source(path)
                return

    def _stop_workers(self):
        """Disconnect old workers so their results are ignored. Don't wait — let them finish quietly."""
        for attr in ['_pitch_worker', '_waveform_worker', '_key_worker',
                     '_transpose_worker', '_opt_worker', '_restore_worker',
                     '_normalize_worker']:
            worker = getattr(self, attr, None)
            if worker is not None:
                try:
                    worker.disconnect()
                except Exception:
                    pass
                # Keep a ref so it doesn't get GC'd while running
                if not hasattr(self, '_old_workers'):
                    self._old_workers = []
                if worker.isRunning():
                    self._old_workers.append(worker)
                    worker.finished.connect(lambda w=worker: self._old_workers.remove(w) if w in self._old_workers else None)
                setattr(self, attr, None)


    def _load_source(self, path):
        """Load a source audio file."""
        self._stop_workers()
        self._original_source = path
        name = os.path.basename(path)
        self._drop_zone.setText(f"♪  {name}")
        self._drop_zone.setStyleSheet("""
            QLabel {
                background-color: rgba(37, 99, 235, 8);
                border: 1px solid rgba(37, 99, 235, 25);
                border-radius: 12px;
                color: #aaa;
                font-size: 12px;
            }
        """)
        self._btn_clear_source.setVisible(True)
        self._lbl_pitch.setText("Analyzing pitch...")
        self._show_spinner("analyzing")
        self._optimized_for_model = -1

        # Normalize audio in background to avoid UI freeze
        self._normalize_worker = _NormalizeWorker(path)
        self._normalize_worker.finished.connect(self._on_normalize_done)
        self._normalize_worker.start()

    def _on_normalize_done(self, norm_path):
        """Called when background normalization finishes."""
        if not self._original_source:
            return  # Source was cleared while normalizing
        self._source_path = norm_path

        self._waveform.load(norm_path)

        self._pitch_worker = _PitchWorker(norm_path)
        self._pitch_worker.result.connect(self._on_pitch_result)
        self._pitch_worker.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            event.accept()
            try:
                from PyQt6.sip import isdeleted
                wf = self._active_waveform
                if not wf or isdeleted(wf) or not wf.isVisible() or not wf._samples:
                    return
                if not wf._has_source or wf._duration_ms <= 0 or isdeleted(wf._player):
                    return
                # Stop the other waveform if playing
                other = self._waveform_output if wf is self._waveform else self._waveform
                if other._is_playing and other.isVisible() and not isdeleted(other) and not isdeleted(other._player):
                    other._player.pause()
                    other.update()
                # Toggle this one
                if wf._is_playing:
                    wf._player.pause()
                else:
                    wf._player.play()
                wf.update()
            except (RuntimeError, OSError):
                pass
            return
        super().keyPressEvent(event)

    def _set_output_folder(self, event=None):
        import services.paths as _paths
        folder = QFileDialog.getExistingDirectory(self, "Set Output Folder", str(_paths.OUTPUT_DIR))
        if folder:
            _paths.OUTPUT_DIR = folder

    def _show_create_model(self):
        self._create_panel._populate_existing_datasets()
        self._create_panel.setGeometry(self.rect())
        self._create_panel.setVisible(True)
        self._create_panel.raise_()

    def _hide_create_model(self):
        self._create_panel.setVisible(False)
        # Refresh models in case one was just trained
        self._refresh_models()

    def _open_output_folder(self, event=None):
        import subprocess
        subprocess.Popen(["open", str(OUTPUT_DIR)])

    def mousePressEvent(self, event):
        if self._dropdown.isVisible():
            if not self._dropdown.geometry().contains(event.pos()):
                bar = self._search.geometry()
                if not bar.contains(event.pos()):
                    self._dropdown.setVisible(False)
                    self._search.clear()
        super().mousePressEvent(event)

    def save_session(self):
        """Save current state so the app reopens where the user left off."""
        import json
        session = {
            "source_path": self._source_path or "",
            "selected_model": self._cmb_model.currentText() if self._cmb_model.count() > 0 else "",
            "range_match": self._btn_range_match.isChecked(),
            "source_median_hz": self._source_median_hz,
            "model_center_hz": self._model_center_hz,
            "waveform_sections": self._waveform._sections,
            "waveform_transposes": self._waveform._transposes,
            "section_info": self._waveform._section_info,
            "converted": self._waveform._converted,
        }
        from services.job_store import save_config
        save_config({"session": session})

    def restore_session(self):
        """Restore state from last session."""
        try:
            self._restore_session_inner()
        except Exception as e:
            print(f"Session restore error (non-fatal): {e}")

    def _restore_session_inner(self):
        from services.job_store import load_config
        config = load_config()
        session = config.get("session")
        if not session:
            return

        # Always default to Best Match — no audio or waveform restored
        self._carousel.select(0)
        self._btn_range_match.setChecked(session.get("range_match", True))

    def _deferred_restore(self, source_path):
        """Stage the restore: load samples in background, then load media player after."""
        worker = _WaveformSamplesOnly(source_path)
        worker.finished.connect(self._on_restore_waveform)
        self._restore_worker = worker
        worker.start()
        QTimer.singleShot(300, lambda: self._waveform.load(source_path))

    def _on_restore_waveform(self, samples):
        """Apply saved sections over freshly loaded waveform samples."""
        restore = getattr(self, "_pending_restore", None)
        if not restore or not samples:
            return
        self._waveform._samples = samples
        self._waveform._sections = restore["sections"]
        self._waveform._transposes = restore["transposes"]
        self._waveform._section_info = restore.get("section_info", [])
        self._waveform._converted = restore.get("converted", False)
        self._waveform.setVisible(True)
        self._waveform._invalidate_wave_cache()
        self._pending_restore = None
