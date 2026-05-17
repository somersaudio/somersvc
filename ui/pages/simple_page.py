"""Simple one-screen interface for SomerSVC."""

import os

from PyQt6.QtCore import Qt, QRectF, QSize, QThread, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath,
    QPen, QPixmap,
)
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
    QScrollArea,
    QStyledItemDelegate,
    QTreeWidget,
    QTreeWidgetItem,
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

            # Per-section pitches are needed for the coloring system AND
            # for transpose calculation. They don't depend on the model,
            # so always compute them — even if the user hasn't picked a
            # model yet, or if there's only a single section. Previously
            # we gated this on (model_center_hz > 0 AND len(sections) > 1)
            # which meant short clips or "audio dropped before model
            # selected" stayed gray forever even after the model arrived.
            import tempfile
            from services.section_splitter import (
                split_audio_file, analyze_section_pitches,
                calculate_section_transposes,
            )
            transposes = [0] * len(sections)
            median_hz_list = [0] * len(sections)
            if len(sections) >= 1:
                tmp_dir = tempfile.mkdtemp(prefix="svc_wf_")
                try:
                    section_paths = split_audio_file(self.audio_path, sections, tmp_dir)
                    pitched = analyze_section_pitches(section_paths)
                    median_hz_list = [s.get("median_hz", 0) for s in pitched]
                    if self.model_center_hz > 0:
                        pitched = calculate_section_transposes(
                            pitched, self.model_center_hz
                        )
                        transposes = [s.get("transpose", 0) for s in pitched]
                except Exception:
                    pass
                finally:
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)

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


class _ClipProcessWorker(QThread):
    """Background: copy each uploaded file into the app's own staging
    folder, split it into ~7s training clips, and flag silent ones.

    The user's original dropped files are never modified or deleted —
    everything happens on copies under CACHE_DIR/clip_staging/.
    """

    file_done = pyqtSignal(dict)   # one processed-file record
    all_done = pyqtSignal()

    CHUNK_SEC = 7.0
    TAIL_MERGE_SEC = 3.0     # a trailing remnant shorter than this is merged
    SPLIT_MIN_SEC = 10.0     # files at/under this are kept whole (one clip)
    SILENCE_RMS = 0.003      # ~-50 dBFS — below this a clip counts as silent

    def __init__(self, paths, staging_root):
        super().__init__()
        self._paths = list(paths)
        self._staging_root = staging_root

    def run(self):
        for path in self._paths:
            try:
                rec = self._process_one(path)
            except Exception as e:
                rec = {"name": os.path.basename(path), "source": path,
                       "clips": [], "error": str(e)}
            self.file_done.emit(rec)
        self.all_done.emit()

    def _process_one(self, path: str) -> dict:
        import tempfile
        import numpy as np
        import soundfile as sf

        os.makedirs(self._staging_root, exist_ok=True)
        work_dir = tempfile.mkdtemp(prefix="clip_", dir=self._staging_root)
        stem = os.path.splitext(os.path.basename(path))[0]

        # Keep the source's channels — do NOT downmix to mono here.
        # Vocal isolation (Demucs) needs real stereo; a mono clip makes
        # Demucs crash. Training staging downmixes to mono later anyway.
        audio, sr = sf.read(path)
        total = len(audio)
        duration = total / sr if sr else 0.0

        clip_paths = []
        if duration <= self.SPLIT_MIN_SEC:
            # Short enough to use as-is — one clip, no split.
            cp = os.path.join(work_dir, f"{stem}_part01.wav")
            sf.write(cp, audio, sr)
            clip_paths.append(cp)
        else:
            chunk = int(self.CHUNK_SEC * sr)
            tail_min = int(self.TAIL_MERGE_SEC * sr)
            pos, idx = 0, 1
            while pos < total:
                end = min(pos + chunk, total)
                if 0 < (total - end) < tail_min:
                    end = total  # absorb a too-short tail into this chunk
                cp = os.path.join(work_dir, f"{stem}_part{idx:02d}.wav")
                sf.write(cp, audio[pos:end], sr)
                clip_paths.append(cp)
                idx += 1
                pos = end

        clips = [{"path": cp, "silent": self._is_silent(cp)}
                 for cp in clip_paths]
        return {"name": os.path.basename(path), "source": path,
                "staged_dir": work_dir, "clips": clips, "error": ""}

    def _is_silent(self, path: str) -> bool:
        """RMS over the first ~10s below ~-50 dBFS — too quiet to train on."""
        import numpy as np
        import soundfile as sf
        try:
            with sf.SoundFile(path) as f:
                n = min(f.frames, f.samplerate * 10)
                if n <= 0:
                    return True
                data = f.read(n, dtype="float32")
            if getattr(data, "ndim", 1) > 1:
                data = data.mean(axis=1)
            if data.size == 0:
                return True
            return float(np.sqrt(np.mean(data ** 2))) < self.SILENCE_RMS
        except Exception:
            return False


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
        from services.audio_device_tracker import register_audio_output
        self._player = QMediaPlayer()
        self._audio_out = QAudioOutput()
        # Follow the system's default output so swapping headphones /
        # speakers in macOS routes the app's audio there too.
        register_audio_output(self._audio_out)
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

            # Clamp within the neighbouring splits. The gap is just a
            # couple of pixels — enough that a section can't collapse to
            # zero width, but small enough to butt a divider right up
            # against its neighbour.
            draw_w = max(1, self.width() - 2 * self.MARGIN_X)
            min_gap = 2.0 / draw_w
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
        Source and output waveforms render identically."""
        if dist >= 99:
            return QColor(255, 255, 255, 0), QColor(200, 200, 200, 130)
        elif dist <= 3:
            r, g, b = 80, 200, 120
            return QColor(r, g, b, alpha_bg), QColor(r, g, b, alpha_bar)
        elif dist <= 6:
            r, g, b = 200, 200, 80
            return QColor(r, g, b, alpha_bg), QColor(r, g, b, alpha_bar)
        else:
            r, g, b = 255, 140, 80
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
            default_color = QColor(200, 200, 200, 130)
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
        self._show_grade_badge = False  # opt-in per-instance (Create panel only)
        self._grade_badge_cache: dict = {}  # (grade, size) → scaled QPixmap
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
        """Update Best Match circular cache with current GIF frame."""
        if not self._models or not self._gif_movie:
            return
        # No-op if there's no Best Match entry (e.g. used in Create panel)
        if not self._models[0].get("is_best_match"):
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

    def set_grade_badges_enabled(self, enabled: bool):
        """Show small letter-grade badge on each artist circle. Used by the
        Create panel; the front-page carousel leaves it off so the picker
        stays clean.
        """
        self._show_grade_badge = bool(enabled)
        self.update()

    def _get_grade_pixmap(self, grade: str, size: int):
        """Return a scaled badge pixmap for the given grade letter, cached."""
        key = (grade, size)
        cached = self._grade_badge_cache.get(key)
        if cached is not None:
            return cached
        path = os.path.join(APP_DIR, "assets", "grade_badges", f"{grade}.png")
        if not os.path.exists(path):
            self._grade_badge_cache[key] = None
            return None
        px = QPixmap(path)
        if px.isNull():
            self._grade_badge_cache[key] = None
            return None
        scaled = px.scaled(
            size, size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._grade_badge_cache[key] = scaled
        return scaled

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

    def event(self, event):
        # Show a contextual tooltip when hovering the "?"/key badge
        from PyQt6.QtCore import QEvent
        from PyQt6.QtWidgets import QToolTip
        if event.type() == QEvent.Type.ToolTip:
            if self._badge_rect and self._badge_rect.contains(event.pos()):
                if 0 <= self._selected < len(self._models):
                    model = self._models[self._selected]
                    key = (model.get("vocal_key") or "").strip()
                    if not key or key == "Auto":
                        QToolTip.showText(
                            event.globalPos(),
                            "Vocal key not detected yet.\n\n"
                            "It populates automatically after training, when the\n"
                            "trainer scans the artist's audio to find their median\n"
                            "pitch. The app then uses that key to recommend\n"
                            "transposes that keep your songs in the artist's range.\n\n"
                            "Click to estimate it now from Spotify + GetSongBPM\n"
                            "(no training data required).",
                            self,
                        )
                    else:
                        QToolTip.showText(
                            event.globalPos(),
                            f"Detected vocal key: {key}",
                            self,
                        )
                    return True
            QToolTip.hideText()
        return super().event(event)

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

        # Pre-calculate all sizes. Per-instance cap so a narrower carousel
        # (e.g. with a metadata panel beside it) can render fewer items.
        items = []
        max_dist = getattr(self, "_max_visible_dist", 6)
        for i in range(len(self._models)):
            dist = abs(i - self._anim_pos)
            if dist > max_dist:
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
        model = self._models[idx]
        is_best = bool(model.get("is_best_match"))
        # Cache at the largest size this item will ever render at
        base_size = int(self.CENTER_SIZE * 1.4) if is_best else self.CENTER_SIZE
        base_key = (idx, base_size)

        if base_key not in self._circular_cache:
            # For Best Match with GIF, use current frame with soft fade
            if is_best and self._gif_movie:
                px = self._gif_movie.currentPixmap()
            else:
                px = model.get("pixmap")
            if is_best and px and not px.isNull():
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

            # Untrained / "pending" entries render dim so the user sees the
            # placeholder is provisional (no model yet, just a name+image).
            is_pending_local = bool(self._models[idx].get("pending"))
            painter.setOpacity(opacity * 0.2 if is_pending_local else opacity)

            is_best_local = bool(self._models[idx].get("is_best_match"))
            render_size = int(size * 1.4) if is_best_local else size
            px = self._get_circular(idx, render_size)
            draw_x = x - render_size // 2
            draw_y = cy - render_size // 2 + y_off
            painter.drawPixmap(draw_x, draw_y, px)

            # Draw border (skip for Best Match — it has a faded edge)
            if not is_best_local:
                is_center = dist < 0.3
                border_color = QColor(255, 255, 255, 200) if is_center else QColor(80, 80, 80, int(120 * opacity))
                pen = QPen(border_color, 2 if is_center else 1)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(draw_x, draw_y, size, size)

                # Grade badge (Create-panel carousel only): a small letter
                # chip on the bottom-right of each trained artist's circle.
                # Uses the pre-rendered badge images in assets/grade_badges/.
                model = self._models[idx]
                grade = model.get("grade")
                if (self._show_grade_badge and grade and grade not in ("", "--")
                        and not model.get("pending")):
                    chip_d = max(20, int(size * 0.42))
                    badge_px = self._get_grade_pixmap(grade, chip_d)
                    if badge_px is not None:
                        # Anchor the badge so it slightly overlaps the circle
                        # at the bottom-right.
                        chip_x = draw_x + size - int(chip_d * 0.78)
                        chip_y = draw_y + size - int(chip_d * 0.78)
                        painter.setOpacity(opacity * 0.5)
                        painter.drawPixmap(chip_x, chip_y, badge_px)
                        painter.setOpacity(opacity)  # restore for next items

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


# Item-data roles for clip-tree rows.
_CLIP_SILENT_ROLE = Qt.ItemDataRole.UserRole + 1   # silent-clip flag
_CLIP_RECORD_ROLE = Qt.ItemDataRole.UserRole + 2   # upload record (file nodes)


class _ClipBadgeDelegate(QStyledItemDelegate):
    """Paints an ISOLATED badge on isolated clips, and a red "silent"
    marker on clips skipped for being silent."""

    # Vintage gold (mid-saturation, slightly aged)
    _BADGE_BG = QColor(193, 154, 70, 220)         # warm gold
    _BADGE_BORDER = QColor(140, 105, 35, 230)     # darker gold edge
    _BADGE_TEXT = QColor(28, 22, 8)               # dark coffee text

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        # Silent clips — a red "silent" word on the right edge.
        if index.data(_CLIP_SILENT_ROLE):
            painter.save()
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                f = QFont(painter.font())
                f.setPointSize(8)
                f.setBold(True)
                painter.setFont(f)
                painter.setPen(QColor(224, 92, 92))
                painter.drawText(
                    option.rect.adjusted(0, 0, -10, 0),
                    Qt.AlignmentFlag.AlignRight
                    | Qt.AlignmentFlag.AlignVCenter,
                    "silent",
                )
            finally:
                painter.restore()
            return
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        # Match the current "_Isolated_Vocals" suffix and the older "_isolated" tag
        # so previously-isolated clips don't lose their badge after the rename.
        lower = text.lower()
        if "_isolated_vocals" not in lower and "_isolated" not in lower:
            return
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            badge_text = "ISOLATED"
            font = QFont(painter.font())
            font.setPointSize(8)
            font.setBold(True)
            font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 110)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(badge_text)
            badge_w = text_w + 14
            badge_h = max(16, option.rect.height() - 10)
            margin = 8
            x = option.rect.right() - badge_w - margin
            y = option.rect.center().y() - badge_h // 2
            badge_rect = QRectF(x, y, badge_w, badge_h)
            painter.setPen(QPen(self._BADGE_BORDER, 1))
            painter.setBrush(QBrush(self._BADGE_BG))
            painter.drawRoundedRect(badge_rect, 4, 4)
            painter.setPen(self._BADGE_TEXT)
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)
        finally:
            painter.restore()


class _HFArtistDelegate(QStyledItemDelegate):
    """Renders HF dropdown rows normally, then overlays the user's
    grade rating as a subtle cyan letter at the right of the row."""

    GRADE_ROLE = Qt.ItemDataRole.UserRole + 1
    _GRADE_COLOR = QColor(108, 197, 212, 165)  # subtle cyan

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        grade = index.data(self.GRADE_ROLE)
        if not grade:
            return
        painter.save()
        try:
            font = QFont(option.font)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(self._GRADE_COLOR)
            rect = option.rect.adjusted(0, 0, -12, 0)
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                grade,
            )
        finally:
            painter.restore()


class _CreateModelPanel(QWidget):
    """Unified create-a-model panel combining dataset + training in a clean flow."""
    back_clicked = pyqtSignal()
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        # Background-image state — mirrors SimplePage so the panel can
        # show the selected artist's photo behind the top half with
        # the same gradient fade.
        self._bg_pixmap = None
        self._bg_opacity = 0.35
        self._bg_cache = None
        self._bg_cache_size = None
        self._init_ui()
        # Default wallpaper before the user selects anyone — same as the
        # front page's Best Match background.
        best_bg = os.path.join(APP_DIR, "assets", "best_match.png")
        if os.path.exists(best_bg):
            pix = QPixmap(best_bg)
            if not pix.isNull():
                self._bg_pixmap = pix

    def _build_bg_cache(self):
        """Pre-composite the artist photo with gradient fades."""
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
            # Top-half fade-down gradient
            grad = QLinearGradient(0, 0, 0, h * 0.6)
            grad.setColorAt(0.0, QColor(26, 26, 26, 0))
            grad.setColorAt(0.5, QColor(26, 26, 26, 80))
            grad.setColorAt(1.0, QColor(26, 26, 26, 255))
            p.fillRect(0, 0, w, int(h * 0.6), grad)
            # Solid floor below the fade
            p.fillRect(0, int(h * 0.6), w, h - int(h * 0.6), QColor("#1a1a1a"))
            # Side fades
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Invalidate the cache so the next paint rebuilds it for the new size
        self._bg_cache = None

    def _update_panel_background(self, name: str):
        """Swap the artist photo behind the panel based on the selected model.

        Falls back to assets/best_match.png — the same wallpaper the front
        page uses when "Best Match" is selected — when no specific artist
        image is available.
        """
        from services.paths import MODELS_DIR
        self._bg_pixmap = None
        if name:
            model_dir = os.path.join(str(MODELS_DIR), name)
            thumb_dir = self._image_cache_dir if hasattr(self, "_image_cache_dir") else None
            for candidate in [
                os.path.join(model_dir, "image.jpg"),
                os.path.join(model_dir, "image.jpeg"),
                os.path.join(model_dir, "image.png"),
                os.path.join(model_dir, "image.webp"),
                os.path.join(thumb_dir, f"{name}.jpg") if thumb_dir else None,
            ]:
                if candidate and os.path.exists(candidate):
                    pix = QPixmap(candidate)
                    if not pix.isNull():
                        self._bg_pixmap = pix
                        break
        if self._bg_pixmap is None or self._bg_pixmap.isNull():
            best_bg = os.path.join(APP_DIR, "assets", "best_match.png")
            if os.path.exists(best_bg):
                pix = QPixmap(best_bg)
                if not pix.isNull():
                    self._bg_pixmap = pix
        self._bg_cache = None
        self.update()

    def _init_ui(self):
        # State — initialised before any widgets are built so UI builders
        # that preview state (e.g. _update_auto_epoch_placeholder, which
        # reads _selected_name) never hit an unset attribute on launch.
        self._clips = []
        # Grouped clip model: one record per uploaded file —
        # {name, source, staged_dir, clips:[{path,silent}], error}.
        self._processed_files = []
        self._clip_workers = []
        self._selected_name = ""
        self._training = False
        self._worker = None
        # GPU-availability status for the label above Start Training — set
        # while a cloud run provisions if the chosen GPU is unavailable.
        self._gpu_chosen_unavail = False
        self._gpu_active = None
        self._gpu_got = False
        # Initialise training-progress fields so a reattach via ResumeWorker
        # can read them in _on_train_log without an AttributeError fatal.
        self._recommended_epochs = 0
        self._current_epoch = 0
        self._auto_stop_fired = False
        self._last_log_ckpt_epoch = None
        self._last_log_was_wait = False
        # When resuming training, the trainer reports session-local epoch
        # numbers (0..delta). We add this offset so the live counter shows
        # TOTAL epochs continuous with the previous run.
        self._resume_offset = 0
        self._image_cache_dir = os.path.join(CACHE_DIR, "artist_thumbs")
        # Per-artist staged clips for pending (not-yet-trained) artists.
        # Persists clip selections across artist switches before the dataset
        # dir exists on disk.
        self._pending_clips_by_artist: dict = {}
        self.setAcceptDrops(True)

        # The panel's content is taller than the window can get once a clip
        # waveform is showing — an over-constrained QVBoxLayout was
        # overlapping widgets (clip list over the action buttons). Put
        # everything in a scroll area so it scrolls instead. Kept
        # transparent so the artist-photo background (drawn in paintEvent)
        # still shows through.
        from PyQt6.QtWidgets import QScrollArea, QFrame, QWidget
        _outer = QVBoxLayout(self)
        _outer.setContentsMargins(0, 0, 0, 0)
        _scroll = QScrollArea()
        _scroll.setWidgetResizable(True)
        _scroll.setFrameShape(QFrame.Shape.NoFrame)
        _scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _scroll.setStyleSheet("QScrollArea { background: transparent; }")
        _scroll.viewport().setStyleSheet("background: transparent;")
        _content = QWidget()
        _content.setStyleSheet("background: transparent;")
        _scroll.setWidget(_content)
        _outer.addWidget(_scroll)

        layout = QVBoxLayout(_content)
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

        title = QLabel("Models")
        title.setStyleSheet("color: #ddd; font-size: 18px; font-weight: bold; background: transparent;")
        header.addWidget(title)
        header.addStretch()
        # Import + Export icon buttons (top-right)
        icon_style = (
            "QLabel { color: rgba(255,255,255,55); font-size: 18px; "
            "background: transparent; padding: 2px 6px; }"
            "QLabel:hover { color: rgba(255,255,255,180); }"
        )
        self._btn_import_model = QLabel("⤓")
        self._btn_import_model.setStyleSheet(icon_style)
        self._btn_import_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_import_model.setToolTip("Import a .svc / .pth model from your computer")
        self._btn_import_model.mousePressEvent = lambda e: self._import_model_file()
        header.addWidget(self._btn_import_model)
        self._btn_export_model = QLabel("⤒")
        self._btn_export_model.setStyleSheet(icon_style)
        self._btn_export_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_export_model.setToolTip("Export the selected model as a .svc file")
        self._btn_export_model.mousePressEvent = lambda e: self._export_model_file()
        header.addWidget(self._btn_export_model)
        layout.addLayout(header)

        subtitle = QLabel("Name your voice, add audio clips, and train on a cloud GPU")
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 40); font-size: 11px; background: transparent;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        # Front-page-style carousel of artists. No "Best Match" entry — the
        # Create panel only shows actual trained models / dataset folders.
        # Grade badges enabled here so the user sees model quality at a glance
        # while picking which artist to add data to / continue training.
        carousel_row = QHBoxLayout()
        carousel_row.setContentsMargins(0, 0, 0, 0)
        carousel_row.setSpacing(0)

        # Left-side metadata panel for the currently selected artist.
        # Populated by _update_metadata_panel().
        self._lbl_meta_panel = QLabel("")
        self._lbl_meta_panel.setStyleSheet(
            "color: rgba(255, 255, 255, 80); font-size: 11px; "
            "background: transparent; padding: 8px 12px;"
        )
        self._lbl_meta_panel.setTextFormat(Qt.TextFormat.RichText)
        self._lbl_meta_panel.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._lbl_meta_panel.setFixedWidth(200)
        self._lbl_meta_panel.setWordWrap(True)
        # Allow hover events on inline <a href="..."> so the maturity help
        # icon can show a contextual tooltip without making clicks navigate.
        self._lbl_meta_panel.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self._lbl_meta_panel.setOpenExternalLinks(False)
        self._lbl_meta_panel.linkHovered.connect(self._on_meta_link_hover)
        self._lbl_meta_panel.linkActivated.connect(self._on_meta_link_clicked)
        self._maturity_help_html: str = ""
        carousel_row.addWidget(self._lbl_meta_panel)

        self._carousel = _ModelCarousel()
        self._carousel.setFixedHeight(220)
        self._carousel.set_grade_badges_enabled(True)
        # Narrower fan-out so a matching right spacer can true-center the
        # carousel without clipping leading / trailing artists. dist=3
        # fits inside the resulting 672px width.
        self._carousel._max_visible_dist = 3
        self._carousel.model_selected.connect(self._on_carousel_select)
        carousel_row.addWidget(self._carousel, 1)

        # Mirror-width invisible spacer so the carousel sits exactly
        # centered on the page rather than offset by the metadata panel.
        right_spacer = QLabel("")
        right_spacer.setFixedWidth(200)
        carousel_row.addWidget(right_spacer)

        layout.addLayout(carousel_row)

        # Selected model label (under the carousel)
        self._lbl_selected = QLabel("")
        self._lbl_selected.setStyleSheet("color: rgba(255, 255, 255, 50); font-size: 11px; background: transparent;")
        self._lbl_selected.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_selected)

        # "+ New artist" input — left-aligned above the Audio Clips section
        new_artist_row = QHBoxLayout()
        self._txt_new_name = QLineEdit()
        self._txt_new_name.setPlaceholderText("+ Create new artist model")
        self._txt_new_name.setFixedSize(220, 28)
        self._txt_new_name.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._txt_new_name.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 5);
                border: 1px dashed rgba(255, 255, 255, 20);
                border-radius: 14px;
                color: #ffffff;
                font-size: 11px;
                padding: 0 12px;
            }
            QLineEdit::placeholder {
                color: rgba(255, 255, 255, 110);
            }
            QLineEdit:focus {
                border-color: rgba(255, 255, 255, 60);
                border-style: solid;
                color: #ffffff;
            }
        """)
        self._txt_new_name.returnPressed.connect(self._on_new_name_entered)
        new_artist_row.addWidget(self._txt_new_name)
        new_artist_row.addStretch()
        layout.addLayout(new_artist_row)

        layout.addSpacing(4)

        # Audio Clips section
        step2 = QLabel("Audio Clips")
        step2.setStyleSheet("color: rgba(255, 255, 255, 70); font-size: 11px; font-weight: bold; background: transparent;")
        layout.addWidget(step2)

        # File list showing existing clips
        # A QTreeWidget — uploaded files are top-level nodes, their
        # split clips are expandable children. Each item carries a path
        # in UserRole (the clip path for clip rows, the original upload
        # for a file node).
        self._file_list = QTreeWidget()
        self._file_list.setHeaderHidden(True)
        self._file_list.setColumnCount(1)
        self._file_list.setRootIsDecorated(True)
        # Custom delegate paints an "ISOLATED" badge in vintage gold on the
        # right side of any clip whose basename contains "_isolated".
        self._file_list.setItemDelegate(_ClipBadgeDelegate(self._file_list))
        self._file_list.setFixedHeight(200)
        # Allow shift-click / cmd-click / Cmd+A multi-select for bulk actions
        self._file_list.setSelectionMode(
            QTreeWidget.SelectionMode.ExtendedSelection
        )
        self._file_list.setStyleSheet("""
            QTreeWidget {
                background: rgba(255, 255, 255, 3);
                border: 1px solid rgba(255, 255, 255, 10);
                border-radius: 10px;
                color: #999;
                font-size: 11px;
            }
            QTreeWidget::item {
                padding: 3px 8px;
                border-bottom: 1px solid rgba(255, 255, 255, 5);
            }
            QTreeWidget::item:selected {
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

        self._btn_isolate = QLabel("Isolate selected vocals")
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

        dot_norm = QLabel(" · ")
        dot_norm.setStyleSheet("color: rgba(255, 255, 255, 30); font-size: 10px; background: transparent;")
        action_row.addWidget(dot_norm)

        self._btn_normalize_clip = QLabel("Normalize")
        self._btn_normalize_clip.setStyleSheet("color: rgba(94, 200, 180, 80); font-size: 10px; background: transparent;")
        self._btn_normalize_clip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_normalize_clip.setToolTip("Bring the selected clips' RMS loudness to -12 dBFS (peak-safe)")
        self._btn_normalize_clip.mousePressEvent = lambda e: self._normalize_selected_clips()
        action_row.addWidget(self._btn_normalize_clip)

        # Hide isolate / remove / normalize (and the dots that separate
        # them) until the user actually selects a clip. Splitting is
        # automatic now — there's no Split button.
        self._selection_only_widgets = [
            dot1, self._btn_isolate,
            dot2, self._btn_remove_clip,
            dot_norm, self._btn_normalize_clip,
        ]
        for w in self._selection_only_widgets:
            w.setVisible(False)

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

        self._file_list.currentItemChanged.connect(self._on_clip_selected)
        self._file_list.itemSelectionChanged.connect(self._on_clip_selection_changed)

        layout.addSpacing(4)

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

        self._txt_epochs = QLineEdit()
        self._txt_epochs.setFixedSize(60, 28)
        self._txt_epochs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._txt_epochs.setToolTip(
            "Target epoch count. Left blank, the auto count (shown dim) is "
            "used. Enter an absolute number (e.g. 5626) or a delta (e.g. "
            "+1126 to add to the current epoch). On a resume run, any value "
            "below the current epoch is treated as a delta."
        )
        self._txt_epochs.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 8);
                border: 1px solid rgba(255, 255, 255, 15);
                border-radius: 6px;
                color: #ccc;
                font-size: 11px;
            }
        """)
        # "auto" caption beneath the box — same style as the field
        # captions in Settings. An empty box shows the auto-picked
        # epoch count as a dim placeholder number.
        self._lbl_epochs_auto = QLabel("auto")
        self._lbl_epochs_auto.setStyleSheet(
            "color: rgba(255, 255, 255, 90); font-size: 10px;"
            " background: transparent;"
        )
        self._lbl_epochs_auto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        _epoch_col = QVBoxLayout()
        _epoch_col.setContentsMargins(0, 0, 0, 0)
        _epoch_col.setSpacing(1)
        _epoch_col.addWidget(self._txt_epochs)
        _epoch_col.addWidget(self._lbl_epochs_auto)
        # The "auto" caption makes this column taller than the box, so the
        # row was centring the box higher than the Type field beside it.
        # Pad the top by the caption's height: the box then sits centred in
        # the column — level with the Type box — and "auto" hangs beneath.
        _epoch_col.insertSpacing(0, self._lbl_epochs_auto.sizeHint().height())
        opts_row.addLayout(_epoch_col)
        self._update_auto_epoch_placeholder()

        # Help "?" — hover shows the Auto-epoch lookup table
        epochs_help = QLabel("?")
        epochs_help.setFixedSize(16, 16)
        epochs_help.setAlignment(Qt.AlignmentFlag.AlignCenter)
        epochs_help.setCursor(Qt.CursorShape.WhatsThisCursor)
        epochs_help.setStyleSheet(
            "QLabel { color: rgba(255,255,255,80); font-size: 11px; font-weight: bold; "
            "background: rgba(255,255,255,10); border-radius: 8px; }"
            "QLabel:hover { color: #fff; background: rgba(255,255,255,25); }"
        )
        # Rich-text tooltip — Qt's tooltip renders basic HTML and flexes width
        epochs_help.setToolTip(
            "<div style='font-size:11px;'>"
            "<b>Auto epoch picks based on dataset length</b><br><br>"
            "<table cellspacing='6' cellpadding='2'>"
            "<tr><td align='right'><b>&lt; 3 minutes</b></td><td>3000 epochs</td></tr>"
            "<tr><td align='right'><b>3 – 5 minutes</b></td><td>2500 epochs</td></tr>"
            "<tr><td align='right'><b>5 – 10 minutes</b></td><td>1500 epochs</td></tr>"
            "<tr><td align='right'><b>10 – 30 minutes</b></td><td>500 epochs</td></tr>"
            "<tr><td align='right'><b>30+ minutes</b></td><td>300 epochs</td></tr>"
            "</table><br>"
            "Smaller datasets need more passes to converge. Larger ones see "
            "enough variation per epoch to stop earlier. Type a number to "
            "override."
            "</div>"
        )
        opts_row.addSpacing(6)
        opts_row.addWidget(epochs_help)

        opts_row.addStretch()
        layout.addLayout(opts_row)

        layout.addSpacing(8)

        # Which GPU training will use — reflects Settings → Cloud GPU
        # (or local training). Refreshed each time the panel is shown.
        self._lbl_gpu = QLabel("")
        self._lbl_gpu.setStyleSheet(
            "color: rgba(255, 255, 255, 90); font-size: 10px;"
            " background: transparent;"
        )
        self._lbl_gpu.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_gpu)
        self._refresh_gpu_label()

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

        # Delete Model button (only visible when a trained model exists)
        self._btn_delete_model = QPushButton("Delete Model")
        self._btn_delete_model.setFixedHeight(32)
        self._btn_delete_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_delete_model.setVisible(False)
        self._btn_delete_model.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: rgba(255, 100, 100, 110);
                border: 1px solid rgba(255, 100, 100, 50);
                border-radius: 6px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: rgba(255, 80, 80, 25);
                color: rgba(255, 130, 130, 200);
            }
        """)
        self._btn_delete_model.clicked.connect(self._delete_model)
        layout.addWidget(self._btn_delete_model)

        # Progress / Status
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: #2DD4BF; font-size: 11px; background: transparent;")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_status)

        # Epoch counter (e.g. "1/500") above the progress bar
        self._lbl_epoch = QLabel("")
        self._lbl_epoch.setStyleSheet(
            "color: #2DD4BF; font-size: 11px; "
            "font-weight: 600; background: transparent;"
        )
        self._lbl_epoch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_epoch.setVisible(False)
        layout.addWidget(self._lbl_epoch)

        from PyQt6.QtWidgets import QProgressBar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        # Tall enough to fully show the centred percentage text — an 8px
        # bar clipped the digits top and bottom. Radius scales with the
        # height so it keeps its rounded-pill shape.
        self._progress_bar.setFixedHeight(24)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(255, 255, 255, 8);
                border: none;
                border-radius: 12px;
            }
            QProgressBar::chunk {
                background: rgba(80, 200, 120, 150);
                border-radius: 12px;
            }
        """)
        layout.addWidget(self._progress_bar)

        from ui.widgets.log_viewer import LogViewer
        self._log = LogViewer()
        self._log.setVisible(False)
        self._log.setMaximumHeight(120)
        layout.addWidget(self._log)

        layout.addStretch()

        # Bottom-left model summary — shows duration trained on, epoch count,
        # and a one-line suggestion (more data vs more epochs).
        # Wrap in a row with a leading spacer so it sits clear of the
        # version label that sits in the corner of the simple container.
        self._lbl_model_info = QLabel("")
        self._lbl_model_info.setStyleSheet(
            "color: rgba(255, 255, 255, 75); font-size: 11px; "
            "background: transparent; line-height: 1.4em;"
        )
        self._lbl_model_info.setWordWrap(True)
        self._lbl_model_info.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._lbl_model_info.setVisible(False)
        # RichText so inline <img> badges render in the suggestion.
        self._lbl_model_info.setTextFormat(Qt.TextFormat.RichText)
        # Force the label to take the full remaining width so the two-line
        # summary doesn't wrap into 4–6 narrow lines.
        from PyQt6.QtWidgets import QSizePolicy
        self._lbl_model_info.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred,
        )
        self._lbl_model_info.setMinimumWidth(720)
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.addSpacing(60)  # clear the version label in the corner
        info_row.addWidget(self._lbl_model_info, 1)  # stretch=1: take the rest
        layout.addLayout(info_row)
        # Lift it off the bottom so it doesn't collide with the
        # "Open Output Folder" / "Set Output Folder" links beneath the panel.
        layout.addSpacing(36)

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
        """Uploaded audio is auto-processed in the background — copied
        into the app's staging area, split into ~7s training clips, and
        silent clips flagged. The user's original files are never
        modified or deleted."""
        if not paths:
            return
        staging_root = os.path.join(CACHE_DIR, "clip_staging")
        worker = _ClipProcessWorker(paths, staging_root)
        worker.file_done.connect(self._on_clip_file_processed)
        worker.all_done.connect(
            lambda w=worker: self._on_clip_processing_done(w)
        )
        self._clip_workers.append(worker)
        self._show_toast("Processing audio…")
        worker.start()

    def _on_clip_file_processed(self, rec: dict):
        """One uploaded file finished processing — record it and add its
        non-silent clips to the training set."""
        self._processed_files.append(rec)
        if rec.get("error"):
            self._show_toast(f"Couldn't process {rec.get('name', 'file')}")
            return
        # Step 1: feed the non-silent clips into the flat list the UI and
        # trainer already use. Grouped display lands in step 2.
        for c in rec.get("clips", []):
            if not c.get("silent"):
                self._clips.append(c["path"])
        self._refresh_file_list()
        # Persist staged clips for pending (untrained, no dataset on
        # disk) artists so closing/reopening the panel keeps the work.
        name = (self._selected_name or "").strip()
        if name and not os.path.isdir(os.path.join(str(DATASETS_DIR), name)):
            self._pending_clips_by_artist[name] = list(self._clips)

    def _on_clip_processing_done(self, worker):
        """A processing worker finished — wait for the thread to fully
        exit before dropping the reference (a bare drop mid-teardown
        would hit QThread's 'destroyed while running' abort)."""
        try:
            worker.wait(5000)
        except Exception:
            pass
        if worker in self._clip_workers:
            self._clip_workers.remove(worker)

    def _show_toast(self, text: str, duration_ms: int = 2500) -> None:
        """Floating pill-shaped message that fades in, holds, and fades out."""
        from PyQt6.QtCore import (
            QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup,
        )
        from PyQt6.QtWidgets import QGraphicsOpacityEffect

        toast = QLabel(text, self)
        toast.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        toast.setStyleSheet(
            "background: rgba(20, 20, 20, 230);"
            "color: #fff;"
            "border: 1px solid rgba(255,255,255,40);"
            "border-radius: 14px;"
            "padding: 10px 18px;"
            "font-size: 13px;"
            "font-weight: 500;"
        )
        toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toast.adjustSize()
        # Top-center of the panel so it doesn't fight the carousel/log.
        toast.move(max(0, (self.width() - toast.width()) // 2), 80)

        effect = QGraphicsOpacityEffect(toast)
        toast.setGraphicsEffect(effect)
        effect.setOpacity(0.0)
        toast.show()
        toast.raise_()

        fade_in_ms = 280
        fade_out_ms = 450
        hold_ms = max(0, duration_ms - fade_in_ms - fade_out_ms)

        fade_in = QPropertyAnimation(effect, b"opacity")
        fade_in.setDuration(fade_in_ms)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.Type.OutQuad)

        fade_out = QPropertyAnimation(effect, b"opacity")
        fade_out.setDuration(fade_out_ms)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InQuad)

        group = QSequentialAnimationGroup(toast)
        group.addAnimation(fade_in)
        group.addPause(hold_ms)
        group.addAnimation(fade_out)
        group.finished.connect(toast.deleteLater)
        group.start()

    def _trained_files_path(self, name: str) -> str:
        """Path to the per-model JSON snapshot of trained-on filenames."""
        from services.paths import MODELS_DIR
        return os.path.join(str(MODELS_DIR), name, "trained_files.json")

    def _load_trained_set(self, name: str) -> set:
        """Return the set of basenames the model was last trained on, or empty."""
        if not name:
            return set()
        p = self._trained_files_path(name)
        if not os.path.exists(p):
            return set()
        try:
            import json
            with open(p) as f:
                data = json.load(f) or {}
            return set(data.get("trained_files", []))
        except Exception:
            return set()

    def _save_trained_snapshot(self, name: str, basenames: list = None) -> None:
        """Write a snapshot of which clips this model was trained on.

        If basenames is None, walks the dataset dir and snapshots whatever
        is currently there (used right after a successful training run).
        """
        from services.paths import MODELS_DIR, DATASETS_DIR
        if not name:
            return
        model_dir = os.path.join(str(MODELS_DIR), name)
        if not os.path.isdir(model_dir):
            return
        if basenames is None:
            ds_dir = os.path.join(str(DATASETS_DIR), name)
            if not os.path.isdir(ds_dir):
                return
            basenames = sorted(
                f for f in os.listdir(ds_dir)
                if f.endswith((".wav", ".flac", ".mp3", ".ogg"))
            )
        try:
            import json
            with open(self._trained_files_path(name), "w") as f:
                json.dump({"trained_files": list(basenames)}, f, indent=2)
        except Exception:
            pass

    def _rebuild_processed_files_from_clips(self):
        """Reconstruct the file → clips grouping from clip filenames.

        The auto-processor names split clips '<source-stem>_partNN.wav'
        (older datasets use '<stem>_clipNNN.wav'). On an app restart or
        artist switch the panel only has a flat clip list — regroup by
        that shared stem so the expandable file → clips tree shows again
        instead of a bare flat list. Clips whose names don't match the
        split pattern are left ungrouped (they render as flat rows).
        """
        import re
        groups: dict = {}
        order: list = []
        for p in self._clips:
            base = os.path.splitext(os.path.basename(p))[0]
            m = re.match(r'^(.+)_(?:part|clip)\d+$', base)
            if not m:
                continue  # not a split clip — _refresh_file_list shows it flat
            stem = m.group(1)
            if stem not in groups:
                groups[stem] = []
                order.append(stem)
            groups[stem].append(p)
        self._processed_files = [
            {
                "name": stem,
                "source": "",
                "staged_dir": "",
                "clips": [{"path": p, "silent": False} for p in groups[stem]],
                "error": "",
            }
            for stem in order
        ]

    def _refresh_file_list(self):
        from PyQt6.QtGui import QBrush, QColor
        self._file_list.clear()

        # Resolve the dataset dir for the currently selected model so we can
        # tell apart "already trained on" clips from newly-added pending ones.
        from services.paths import MODELS_DIR, DATASETS_DIR
        name = self._selected_name.strip()
        dataset_dir = (
            os.path.realpath(os.path.join(str(DATASETS_DIR), name)) if name else ""
        )
        # The authoritative source: trained_files.json snapshot written after
        # the last successful training run. If it doesn't exist (e.g. legacy
        # model trained before this feature shipped), fall back to the older
        # heuristic of "any clip in dataset_dir + a checkpoint exists".
        trained_set = self._load_trained_set(name)
        has_snapshot = bool(trained_set)
        has_checkpoint = bool(name) and os.path.isdir(os.path.join(str(MODELS_DIR), name)) and any(
            f.endswith(".pth") for f in os.listdir(os.path.join(str(MODELS_DIR), name))
        )

        # Translucent overlays — readable over the dark list, accent the row
        TRAINED = QBrush(QColor(60, 200, 130, 55))   # opaque green
        PENDING = QBrush(QColor(255, 165, 60, 55))   # opaque orange

        def _classify(path: str):
            base = os.path.basename(path)
            if has_snapshot:
                return TRAINED if base in trained_set else PENDING
            # Legacy fallback for models trained before snapshots existed
            in_dataset = (
                bool(dataset_dir)
                and os.path.realpath(path).startswith(dataset_dir + os.sep)
            )
            if in_dataset and has_checkpoint:
                return TRAINED
            return PENDING

        try:
            import soundfile as _sf
        except Exception:
            _sf = None

        def _clip_item(path):
            """Build a leaf row for one clip — returns (item, duration).
            The clip's path travels in UserRole so actions and the
            waveform preview never depend on its position."""
            fname = os.path.basename(path)
            text = fname
            dur = 0.0
            if _sf is not None:
                try:
                    dur = _sf.info(path).duration
                    mins, secs = divmod(int(dur), 60)
                    text = f"{fname}  ({mins}:{secs:02d})"
                except Exception:
                    pass
            it = QTreeWidgetItem([text])
            it.setData(0, Qt.ItemDataRole.UserRole, path)
            it.setBackground(0, _classify(path))
            return it, dur

        clips_set = set(self._clips)
        total_dur = 0.0
        grouped = set()

        # Files uploaded this session — grouped, with their split clips
        # as expandable children. A clip shows if it's silent (kept as a
        # "skipped" marker) or still in the trainable set; a non-silent
        # clip the user removed just drops out. A record with nothing
        # left to show is skipped, so it stays in step with self._clips.
        for rec in self._processed_files:
            visible = []
            for c in rec.get("clips", []):
                if c.get("silent"):
                    visible.append((c["path"], True))
                elif c["path"] in clips_set:
                    visible.append((c["path"], False))
            if not visible:
                continue
            n_train = sum(1 for _, sil in visible if not sil)
            file_node = QTreeWidgetItem([
                f"✓  {rec.get('name', 'Audio file')}"
                f"    ·    {n_train} clip{'s' if n_train != 1 else ''}"
            ])
            # Selecting the file node previews the original upload; the
            # record rides along so Remove can drop the whole file.
            file_node.setData(0, Qt.ItemDataRole.UserRole, rec.get("source"))
            file_node.setData(0, _CLIP_RECORD_ROLE, rec)
            file_node.setForeground(0, QColor(125, 200, 150))
            _fnt = QFont()
            _fnt.setBold(True)
            file_node.setFont(0, _fnt)
            self._file_list.addTopLevelItem(file_node)
            for cp, silent in visible:
                if silent:
                    child = self._silent_clip_item(cp)
                else:
                    child, dur = _clip_item(cp)
                    total_dur += dur
                file_node.addChild(child)
                grouped.add(cp)

        # Clips that didn't come through the auto-processor (e.g. an
        # existing artist's dataset) stay as flat top-level rows.
        for p in self._clips:
            if p in grouped:
                continue
            item, dur = _clip_item(p)
            self._file_list.addTopLevelItem(item)
            total_dur += dur

        total = len(self._clips)
        mins, secs = divmod(int(total_dur), 60)
        self._lbl_clips.setText(f"{total} clips  ·  {mins}:{secs:02d} total")
        # Keep the dim auto-epoch hint in step with the dataset length.
        self._update_auto_epoch_placeholder(total_dur)

    def _silent_clip_item(self, path: str):
        """A leaf row for a silent clip — dimmed text, kept visible as a
        'skipped, here's why' marker but never trained on. The red
        'silent' badge is painted by _ClipBadgeDelegate."""
        item = QTreeWidgetItem([os.path.basename(path)])
        item.setData(0, Qt.ItemDataRole.UserRole, path)
        item.setData(0, _CLIP_SILENT_ROLE, True)
        item.setForeground(0, QColor(110, 110, 110))
        return item

    def _on_clip_selected(self, current, previous=None):
        """Load the waveform for the selected clip row."""
        path = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        if not path or not os.path.exists(path):
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

    def _normalize_clips_with_progress(self, pending: list) -> bool:
        """Normalize the given (src, dst) pairs in a worker with a progress dialog.

        Returns True if the user cancelled, False if completed normally.
        """
        from PyQt6.QtCore import Qt, QThread, pyqtSignal
        from PyQt6.QtWidgets import (
            QApplication, QDialog, QLabel, QProgressBar, QVBoxLayout, QPushButton,
        )

        copy_fn = self._copy_clip_normalized

        class _NormalizeWorker(QThread):
            progress = pyqtSignal(int, int, str)   # done, total, current filename
            done = pyqtSignal(bool, str)            # ok, error_or_empty

            def __init__(self, jobs):
                super().__init__()
                self.jobs = jobs
                self._cancel = False

            def cancel(self):
                self._cancel = True

            def run(self):
                try:
                    total = len(self.jobs)
                    for i, (src, dst) in enumerate(self.jobs, 1):
                        if self._cancel:
                            self.done.emit(False, "cancelled")
                            return
                        self.progress.emit(i - 1, total, os.path.basename(src))
                        copy_fn(src, dst)
                    self.progress.emit(total, total, "")
                    self.done.emit(True, "")
                except Exception as e:
                    self.done.emit(False, str(e))

        dlg = QDialog(self)
        dlg.setWindowTitle("Preparing Clips")
        dlg.setModal(True)
        dlg.setFixedSize(440, 150)
        dlg.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(10)
        lbl = QLabel(f"Preparing {len(pending)} clips...")
        lbl.setStyleSheet("color: #ddd; font-size: 13px;")
        lbl.setWordWrap(True)
        v.addWidget(lbl)
        bar = QProgressBar()
        bar.setRange(0, len(pending))
        v.addWidget(bar)
        hint = QLabel("Converting to 44.1 kHz / 16-bit / mono for fastest training.")
        hint.setStyleSheet("color: rgba(255,255,255,80); font-size: 11px;")
        v.addWidget(hint)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        v.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

        worker = _NormalizeWorker(pending)
        result = {"cancelled": False, "error": ""}

        def _on_progress(done: int, total: int, current: str):
            bar.setValue(done)
            if current:
                lbl.setText(f"Preparing clip {done + 1}/{total}: {current}")
            else:
                lbl.setText(f"Preparing clips... done")
            QApplication.processEvents()

        def _on_done(ok: bool, err: str):
            if not ok:
                if err == "cancelled":
                    result["cancelled"] = True
                else:
                    result["error"] = err
            dlg.accept()

        cancel_btn.clicked.connect(lambda: (worker.cancel(), cancel_btn.setEnabled(False)))
        worker.progress.connect(_on_progress)
        worker.done.connect(_on_done)
        worker.start()
        dlg.exec()
        worker.wait()  # ensure worker fully exits before we continue

        if result["error"]:
            QMessageBox.warning(self, "Preparation Failed", result["error"])
            return True
        return result["cancelled"]

    def _copy_clip_normalized(self, src: str, dst: str):
        """Copy a clip into the dataset normalized to 44.1 kHz / 16-bit / mono.

        That's the format so-vits-svc-fork trains on internally. Doing the
        conversion client-side saves upload bandwidth and pod preprocessing
        time. Falls back to a plain byte copy if the file can't be parsed.
        """
        import shutil
        TARGET_SR = 44100
        try:
            import soundfile as _sf
            audio, sr = _sf.read(src, always_2d=False, dtype="float32")
            channels = audio.shape[1] if getattr(audio, "ndim", 1) > 1 else 1
            info = _sf.info(src)
            already_mono = channels == 1
            already_44k = sr == TARGET_SR
            already_16bit = info.subtype == "PCM_16"
            # Nothing to do — fast path
            if already_mono and already_44k and already_16bit:
                shutil.copy2(src, dst)
                return
            # Stereo → mono (mean preserves apparent loudness, avoids clipping)
            if not already_mono:
                audio = audio.mean(axis=1)
            # Resample to 44.1 kHz if needed
            if not already_44k:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
                    sr = TARGET_SR
                except Exception:
                    # Couldn't resample — write at original sr, pod will fix
                    pass
            # Always write as 16-bit PCM (training-ready)
            _sf.write(dst, audio, sr, subtype="PCM_16")
            return
        except Exception:
            pass
        # Unreadable — just byte-copy and let the pod sort it out
        shutil.copy2(src, dst)

    def _on_clip_selection_changed(self):
        """Show isolate / remove / split only when at least one clip is selected."""
        has_selection = bool(self._file_list.selectedItems())
        for w in getattr(self, "_selection_only_widgets", []):
            w.setVisible(has_selection)

    def _selected_clip_paths(self) -> list:
        """Paths of the currently-selected clips — or the active row's
        clip if nothing is multi-selected. Empty list if no selection.

        Keeping clip actions path-based (not row-index based) lets the
        list widget change underneath without touching them.
        """
        items = self._file_list.selectedItems()
        if not items:
            cur = self._file_list.currentItem()
            if cur is not None:
                items = [cur]
        paths = []
        for it in items:
            if it.childCount() > 0:
                # A file node — resolve to all its clip children.
                for i in range(it.childCount()):
                    p = it.child(i).data(0, Qt.ItemDataRole.UserRole)
                    if p and p not in paths:
                        paths.append(p)
            else:
                p = it.data(0, Qt.ItemDataRole.UserRole)
                if p and p not in paths:
                    paths.append(p)
        return paths

    def _normalize_selected_clips(self):
        """Bring each selected clip's RMS to -12 dBFS, peak-safe (no clipping)."""
        targets = self._selected_clip_paths()
        if not targets:
            return

        TARGET_DBFS = -12.0
        target_amp = 10 ** (TARGET_DBFS / 20.0)  # ~0.2512

        from PyQt6.QtCore import QThread, pyqtSignal
        normalize_one = self._normalize_clip_to_dbfs

        class _NormWorker(QThread):
            progress = pyqtSignal(int, int, str)  # done, total, current
            done = pyqtSignal(int, int)            # done_count, skipped_count

            def __init__(self, paths, target_amp):
                super().__init__()
                self.paths = paths
                self.target_amp = target_amp

            def run(self):
                ok = 0
                skipped = 0
                total = len(self.paths)
                for i, p in enumerate(self.paths, 1):
                    self.progress.emit(i - 1, total, os.path.basename(p))
                    if normalize_one(p, self.target_amp):
                        ok += 1
                    else:
                        skipped += 1
                self.progress.emit(total, total, "")
                self.done.emit(ok, skipped)

        self._lbl_status.setText(f"Normalizing {len(targets)} clip(s) to -12 dBFS...")
        self._lbl_status.setStyleSheet(
            "color: rgba(255, 255, 255, 75); font-size: 11px; background: transparent;"
        )
        self._norm_worker = _NormWorker(targets, target_amp)

        def _on_progress(done: int, total: int, current: str):
            if current:
                self._lbl_status.setText(f"Normalizing {done + 1}/{total}: {current}")

        def _on_done(ok: int, skipped: int):
            msg = f"Normalized {ok} clip{'s' if ok != 1 else ''} to -12 dBFS"
            if skipped:
                msg += f" · skipped {skipped} (silent or unreadable)"
            self._lbl_status.setText(msg)
            self._lbl_status.setStyleSheet(
                "color: rgba(80, 200, 120, 150); font-size: 11px; background: transparent;"
            )
            # Auto-clear after a few seconds
            QTimer.singleShot(8000, lambda: self._lbl_status.setText("")
                              if self._lbl_status.text() == msg else None)
            # Refresh waveform preview if a normalized clip is the active selection
            self._refresh_file_list()

        self._norm_worker.progress.connect(_on_progress)
        self._norm_worker.done.connect(_on_done)
        self._norm_worker.start()

    def _normalize_clip_to_dbfs(self, path: str, target_amp: float) -> bool:
        """Scale audio in `path` so its RMS == target_amp, but cap so the peak
        doesn't exceed 0.99 (avoids clipping).

        Returns True if rewritten, False if skipped (silent / unreadable).
        """
        try:
            import soundfile as _sf
            import numpy as _np
            audio, sr = _sf.read(path, always_2d=False, dtype="float32")
            if audio.size == 0:
                return False
            # Compute RMS over the flattened buffer
            flat = audio if audio.ndim == 1 else audio.mean(axis=1)
            rms = float(_np.sqrt(_np.mean(flat ** 2)))
            if rms < 1e-6:
                return False  # silent
            gain = target_amp / rms
            # Peak-safety: don't let the loudest sample exceed 0.99
            peak = float(_np.max(_np.abs(audio)))
            if peak * gain > 0.99:
                gain = 0.99 / max(peak, 1e-6)
            audio = (audio * gain).astype(_np.float32, copy=False)
            info = _sf.info(path)
            subtype = info.subtype if info.subtype else "FLOAT"
            # If we'd overflow a fixed-point format, fall back to float
            if subtype.startswith("PCM_") and float(_np.max(_np.abs(audio))) > 0.999:
                subtype = "FLOAT"
            _sf.write(path, audio, sr, subtype=subtype)
            return True
        except Exception:
            return False

    def _remove_selected_clip(self):
        """Remove selected clips — or a whole uploaded file when its
        file node is selected (record, clips, and staged copies)."""
        items = self._file_list.selectedItems()
        if not items:
            cur = self._file_list.currentItem()
            if cur is not None:
                items = [cur]
        if not items:
            return
        for it in items:
            rec = it.data(0, _CLIP_RECORD_ROLE)
            if rec is not None:
                # File node — drop the whole uploaded file.
                for c in rec.get("clips", []):
                    if c["path"] in self._clips:
                        self._clips.remove(c["path"])
                staged = rec.get("staged_dir")
                if staged and os.path.isdir(staged):
                    import shutil
                    shutil.rmtree(staged, ignore_errors=True)
                if rec in self._processed_files:
                    self._processed_files.remove(rec)
            else:
                p = it.data(0, Qt.ItemDataRole.UserRole)
                if p in self._clips:
                    self._clips.remove(p)
        self._refresh_file_list()

    def _ask_export_options(self, name: str, dataset_files: list, dataset_dur: float):
        """Show a small dialog asking whether to bundle training data.

        Returns True/False for the checkbox state, or None if user cancelled.
        """
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (
            QDialog, QLabel, QCheckBox, QDialogButtonBox, QVBoxLayout,
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Model")
        dlg.setModal(True)
        dlg.setMinimumWidth(380)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(20, 20, 20, 16)
        v.setSpacing(12)

        title = QLabel(f"Export <b>{name}</b>")
        title.setStyleSheet("color: #ddd; font-size: 13px;")
        v.addWidget(title)

        # Build the checkbox label, fall back gracefully if no dataset on disk
        if dataset_files:
            mins = int(dataset_dur) // 60
            secs = int(dataset_dur) % 60
            chk_label = (
                f"Include training data ({len(dataset_files)} "
                f"{'file' if len(dataset_files) == 1 else 'files'} · {mins}:{secs:02d})"
            )
        else:
            chk_label = "Include training data (none on disk)"
        chk = QCheckBox(chk_label)
        chk.setEnabled(bool(dataset_files))
        chk.setChecked(False)
        chk.setStyleSheet("color: #ccc; font-size: 12px;")
        v.addWidget(chk)

        hint = QLabel(
            "Bundles your audio clips alongside the model so it can be "
            "retrained or extended on another machine."
        )
        hint.setStyleSheet("color: rgba(255,255,255,80); font-size: 11px;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return chk.isChecked()

    def _export_model_file(self):
        """Export the currently-selected model as a .svc zip with a progress dialog."""
        name = self._selected_name.strip()
        if not name:
            QMessageBox.information(
                self, "Pick a Model", "Select a trained model first to export it."
            )
            return
        from services.paths import MODELS_DIR, DATASETS_DIR
        model_dir = os.path.join(str(MODELS_DIR), name)
        if not os.path.isdir(model_dir):
            QMessageBox.warning(self, "No Model", f"No trained model found for \"{name}\".")
            return
        if not any(f.endswith(".pth") for f in os.listdir(model_dir)):
            QMessageBox.warning(self, "No Model", f"\"{name}\" has no checkpoint to export yet.")
            return

        # Inspect the dataset (if any) so we can show count + duration in the
        # checkbox label and decide whether to bundle it.
        dataset_dir = os.path.join(str(DATASETS_DIR), name)
        dataset_files = []
        dataset_dur = 0.0
        if os.path.isdir(dataset_dir):
            try:
                import soundfile as _sf
                for f in sorted(os.listdir(dataset_dir)):
                    if not f.endswith(('.wav', '.flac', '.mp3', '.ogg')):
                        continue
                    full = os.path.join(dataset_dir, f)
                    dataset_files.append(full)
                    try:
                        dataset_dur += _sf.info(full).duration
                    except Exception:
                        pass
            except Exception:
                pass

        include_dataset = self._ask_export_options(name, dataset_files, dataset_dur)
        if include_dataset is None:
            return  # user cancelled

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Model",
            os.path.expanduser(f"~/Desktop/{name}.svc"),
            "SomerSVC Model (*.svc);;All Files (*)",
        )
        if not save_path:
            return

        # If user opted in, hand the dataset paths to the worker
        bundled_dataset = dataset_files if include_dataset else []

        from PyQt6.QtCore import Qt, QThread, pyqtSignal
        from PyQt6.QtWidgets import (
            QApplication, QDialog, QLabel, QProgressBar, QVBoxLayout,
        )

        class _ExportWorker(QThread):
            progress = pyqtSignal(int, str)   # percent, status text
            done = pyqtSignal(bool, str)      # ok, error_or_empty

            def __init__(self, model_dir, save_path, name, dataset_paths=None):
                super().__init__()
                self.model_dir = model_dir
                self.save_path = save_path
                self.name = name
                self.dataset_paths = dataset_paths or []

            def run(self):
                import os, zipfile
                try:
                    # Gather files + total size first so we can report bytes-based progress
                    file_list = []
                    total_bytes = 0
                    for root, _dirs, files in os.walk(self.model_dir):
                        for f in files:
                            full = os.path.join(root, f)
                            arcname = os.path.join(
                                self.name, os.path.relpath(full, self.model_dir)
                            )
                            try:
                                size = os.path.getsize(full)
                            except OSError:
                                size = 0
                            file_list.append((full, arcname, size))
                            total_bytes += size
                    # Add training data files under <name>/dataset/<basename>
                    for full in self.dataset_paths:
                        if not os.path.isfile(full):
                            continue
                        arcname = os.path.join(
                            self.name, "dataset", os.path.basename(full)
                        )
                        try:
                            size = os.path.getsize(full)
                        except OSError:
                            size = 0
                        file_list.append((full, arcname, size))
                        total_bytes += size
                    if total_bytes <= 0:
                        total_bytes = 1  # avoid zero-division

                    written = 0
                    chunk_size = 1024 * 1024  # 1 MB
                    self.progress.emit(0, f"Compressing {len(file_list)} files...")

                    with zipfile.ZipFile(
                        self.save_path, "w", zipfile.ZIP_DEFLATED
                    ) as zf:
                        for idx, (full, arcname, size) in enumerate(file_list, 1):
                            # Stream large files chunk-by-chunk so the bar updates
                            # mid-file (single .pth files can be 500MB+).
                            with open(full, "rb") as src:
                                with zf.open(arcname, "w", force_zip64=True) as dst:
                                    while True:
                                        buf = src.read(chunk_size)
                                        if not buf:
                                            break
                                        dst.write(buf)
                                        written += len(buf)
                                        pct = min(int(written / total_bytes * 100), 100)
                                        self.progress.emit(
                                            pct,
                                            f"Compressing {os.path.basename(full)} "
                                            f"({idx}/{len(file_list)})... {pct}%"
                                        )
                    self.progress.emit(100, "Finalizing...")
                    self.done.emit(True, "")
                except Exception as e:
                    self.done.emit(False, str(e))

        dlg = QDialog(self)
        dlg.setWindowTitle("Exporting Model")
        dlg.setModal(True)
        dlg.setFixedSize(440, 130)
        dlg.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(10)
        lbl = QLabel("Preparing export...")
        lbl.setStyleSheet("color: #ddd; font-size: 13px;")
        lbl.setWordWrap(True)
        v.addWidget(lbl)
        bar = QProgressBar()
        bar.setRange(0, 100)
        v.addWidget(bar)
        hint = QLabel("Compression takes ~10–30s for typical models.")
        hint.setStyleSheet("color: rgba(255,255,255,80); font-size: 11px;")
        v.addWidget(hint)

        worker = _ExportWorker(model_dir, save_path, name, bundled_dataset)

        def _on_progress(pct: int, text: str):
            bar.setValue(pct)
            lbl.setText(text)
            QApplication.processEvents()

        def _on_done(ok: bool, err: str):
            dlg.accept()
            if ok:
                QMessageBox.information(self, "Exported", f"Model exported to:\n{save_path}")
            else:
                QMessageBox.warning(self, "Export Failed", err or "Unknown error.")

        worker.progress.connect(_on_progress)
        worker.done.connect(_on_done)
        worker.start()
        dlg.exec()

    def _import_model_file(self):
        """Import a .svc / .pth / .zip model from disk into the user's models dir.

        Detection is by file content, not extension. iMessage / Mail / AirDrop
        will frequently strip the file extension on the recipient's side, and
        modern torch-saved .pth files are zip-formatted internally — both of
        which made the extension-based dispatcher fall over with confusing
        "File is not a zip file" / "No model checkpoint found" errors.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Model", "",
            "All Models (*.svc *.pth *.zip);;SomerSVC Model (*.svc);;"
            "RVC Model (*.pth);;Zip Files (*.zip);;All Files (*)",
        )
        if not path:
            return

        from services.paths import MODELS_DIR
        models_root = str(MODELS_DIR)
        os.makedirs(models_root, exist_ok=True)
        import shutil
        import zipfile

        # ── Content-based type detection ────────────────────────────────
        # Read the magic bytes to decide which path to take.
        # - PK\x03\x04 : ZIP (either an .svc bundle or a modern torch .pth)
        # - \x80\x0[2-5] : Python pickle (legacy torch .pth, < 1.6)
        try:
            with open(path, "rb") as _f:
                magic = _f.read(4)
        except Exception as e:
            QMessageBox.warning(self, "Import Failed", f"Could not read file: {e}")
            return

        zip_magic = magic.startswith(b"PK\x03\x04")
        is_pickle = (
            len(magic) >= 2 and magic[0] == 0x80 and magic[1] in (2, 3, 4, 5)
        )

        # If the zip magic is there, try opening it for real to decide
        # whether the bundle is intact (and whether it's a .svc with a .pth
        # member vs a modern torch zip-format .pth with no .pth member).
        # A truncated transfer produces a file with a valid PK header but
        # invalid central directory — caught as BadZipFile.
        is_zip = False
        zip_truncated = False
        is_svc_bundle = False
        if zip_magic:
            try:
                with zipfile.ZipFile(path, "r") as _zf:
                    is_zip = True
                    is_svc_bundle = any(
                        n.endswith(".pth") for n in _zf.namelist()
                    )
            except zipfile.BadZipFile:
                zip_truncated = True  # header OK, body corrupt

        # Truncated zip → tell the user exactly what's wrong + how to fix it.
        if zip_truncated:
            try:
                sz = os.path.getsize(path)
            except OSError:
                sz = 0
            size_mb = sz / (1024 * 1024)
            QMessageBox.warning(
                self, "Model file looks incomplete",
                f"This bundle's header is valid but the rest of the file is "
                f"missing or damaged — almost always a sign the transfer was "
                f"truncated.\n\n"
                f"File size: {size_mb:.1f} MB. Trained model bundles are "
                f"usually 150–500 MB.\n\n"
                f"Ask the sender to share via Google Drive, Dropbox, or a "
                f"direct download link instead of Messages, Mail, or AirDrop."
            )
            return

        # Reject anything we definitely can't handle, with a hint about the
        # most common cause (extension stripped in transit).
        if not is_zip and not is_pickle:
            QMessageBox.warning(
                self, "Unrecognized File",
                "This doesn't look like a SomerSVC model (.svc), an RVC "
                "checkpoint (.pth), or a zip bundle.\n\n"
                "If it was sent via Mail, Messages, or AirDrop and lost its "
                "extension, try renaming it to end in .pth or .svc and try "
                "again."
            )
            return

        # Treat a torch-zip .pth identically to a legacy pickle .pth — both
        # are single-checkpoint imports. Only a real .svc bundle takes the
        # multi-file extraction path.
        treat_as_loose_pth = (is_pickle or (is_zip and not is_svc_bundle))

        try:
            # Loose .pth (RVC-style or modern torch zip) — ask for a name,
            # drop in its own folder. Force the saved filename to end in
            # .pth so the loader recognizes it even when the source file
            # had its extension stripped.
            if treat_as_loose_pth:
                from PyQt6.QtWidgets import QInputDialog
                stem = os.path.splitext(os.path.basename(path))[0]
                for suffix in ("_v2", "_v1", "-v2", "-v1"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                name, ok = QInputDialog.getText(
                    self, "Model Name", "Name for this model:", text=stem
                )
                if not ok or not name.strip():
                    return
                name = name.strip()
                dest = os.path.join(models_root, name)
                os.makedirs(dest, exist_ok=True)
                # Ensure the saved file has a .pth extension so the loader
                # finds it — the source may have arrived extension-less.
                base = os.path.basename(path)
                if not base.lower().endswith(".pth"):
                    base = base + ".pth"
                shutil.copy2(path, os.path.join(dest, base))
                # If a .index file lives next to it, bring that along too
                pth_dir = os.path.dirname(path)
                for f in os.listdir(pth_dir):
                    if f.endswith(".index"):
                        shutil.copy2(os.path.join(pth_dir, f), os.path.join(dest, f))
            else:
                # .svc or .zip - route model files into MODELS_DIR/<name>,
                # and any bundled training audio under dataset/ into
                # DATASETS_DIR/<name>.
                from services.paths import DATASETS_DIR
                datasets_root = str(DATASETS_DIR)
                os.makedirs(datasets_root, exist_ok=True)
                with zipfile.ZipFile(path, "r") as zf:
                    names = zf.namelist()
                    if not names:
                        QMessageBox.warning(self, "Invalid", "The file is empty.")
                        return
                    top = names[0].split("/")[0]
                    has_pth = any(n.endswith(".pth") for n in names)
                    if not has_pth:
                        # We only enter this branch when is_svc_bundle was
                        # True at detection time — so the .pth member must
                        # have gone missing between detection and now (rare
                        # race), OR the file was corrupted in transit and
                        # only some entries survived. Either way, point the
                        # user at the most actionable next step.
                        try:
                            sz = os.path.getsize(path)
                        except OSError:
                            sz = 0
                        size_mb = sz / (1024 * 1024)
                        QMessageBox.warning(
                            self, "Model file looks incomplete",
                            f"The bundle opened, but the model checkpoint "
                            f"(.pth) inside is missing.\n\n"
                            f"This usually means the file was truncated in "
                            f"transit — Messages, Mail, and AirDrop can drop "
                            f"large files silently. The bundle is "
                            f"{size_mb:.1f} MB; a trained model bundle is "
                            f"usually 150–500 MB.\n\n"
                            f"Ask the sender to share via Google Drive, "
                            f"Dropbox, or a download link instead."
                        )
                        return
                    model_dest = os.path.join(models_root, top)
                    dataset_dest = os.path.join(datasets_root, top)
                    if os.path.exists(model_dest):
                        reply = QMessageBox.question(
                            self, "Overwrite?",
                            f"A model named \"{top}\" already exists. Replace it?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        )
                        if reply != QMessageBox.StandardButton.Yes:
                            return
                        shutil.rmtree(model_dest, ignore_errors=True)
                    os.makedirs(model_dest, exist_ok=True)

                    # Walk entries: split on the bundled-dataset prefix
                    dataset_prefix = f"{top}/dataset/"
                    bundled_dataset = any(
                        n.startswith(dataset_prefix) and not n.endswith("/")
                        for n in names
                    )
                    if bundled_dataset:
                        os.makedirs(dataset_dest, exist_ok=True)

                    bundled_basenames = []
                    for member in names:
                        if member.endswith("/"):
                            continue
                        if member.startswith(dataset_prefix):
                            # Drop the "<top>/dataset/" prefix and write into datasets dir
                            rel = member[len(dataset_prefix):]
                            out_path = os.path.join(dataset_dest, rel)
                            if rel.endswith((".wav", ".flac", ".mp3", ".ogg")):
                                bundled_basenames.append(os.path.basename(rel))
                        elif member.startswith(f"{top}/"):
                            # Strip the top folder; rejoin under model_dest
                            rel = member[len(top) + 1:]
                            out_path = os.path.join(model_dest, rel)
                        else:
                            # Unprefixed entry — drop into model_dest as a fallback
                            out_path = os.path.join(model_dest, member)
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with zf.open(member) as src, open(out_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    name = top
                    # If the importer didn't ship a trained_files.json of its own
                    # but did bundle audio, mark those clips as trained so the
                    # recipient sees green right away.
                    snapshot_in_zip = os.path.join(model_dest, "trained_files.json")
                    if bundled_basenames and not os.path.exists(snapshot_in_zip):
                        self._save_trained_snapshot(name, sorted(set(bundled_basenames)))
        except Exception as e:
            QMessageBox.warning(self, "Import Failed", str(e))
            return

        # Refresh and select the new model
        self._populate_existing_datasets()
        self._select_model(name)
        QMessageBox.information(self, "Imported", f"Model \"{name}\" imported.")

    def _delete_model(self):
        """Delete the trained model files; then offer to delete the dataset too."""
        name = self._selected_name.strip()
        if not name:
            return
        from services.paths import MODELS_DIR, DATASETS_DIR
        model_dir = os.path.join(str(MODELS_DIR), name)
        if not os.path.isdir(model_dir):
            return
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Delete the trained model \"{name}\"? "
            f"This removes the model checkpoint from your app. "
            f"You'll be asked next whether to also delete its dataset.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)
        self._btn_continue_train.setVisible(False)
        self._btn_delete_model.setVisible(False)

        # Offer to delete the dataset too — useful when the user is fully
        # done with the artist (or starting fresh).
        dataset_dir = os.path.join(str(DATASETS_DIR), name)
        dataset_existed = os.path.isdir(dataset_dir)
        deleted_dataset = False
        if dataset_existed:
            try:
                clip_files = [
                    f for f in os.listdir(dataset_dir)
                    if f.endswith((".wav", ".flac", ".mp3", ".ogg"))
                ]
                clip_count = len(clip_files)
            except Exception:
                clip_count = 0
            duration_str = ""
            try:
                import soundfile as _sf
                total_dur = 0.0
                for f in clip_files:
                    try:
                        total_dur += _sf.info(os.path.join(dataset_dir, f)).duration
                    except Exception:
                        pass
                if total_dur > 0:
                    m, s = divmod(int(total_dur), 60)
                    duration_str = f", {m}:{s:02d} of audio"
            except Exception:
                pass
            ds_reply = QMessageBox.question(
                self, "Delete Dataset Too?",
                f"Also delete the dataset for \"{name}\" "
                f"({clip_count} clip{'s' if clip_count != 1 else ''}{duration_str})?\n\n"
                f"This permanently removes the audio clips you uploaded for "
                f"this artist. Choose No to keep them so you can train again later.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ds_reply == QMessageBox.StandardButton.Yes:
                shutil.rmtree(dataset_dir, ignore_errors=True)
                # Clear any in-memory staged clips for this artist too
                try:
                    self._pending_clips_by_artist.pop(name, None)
                except Exception:
                    pass
                self._clips = []
                self._refresh_file_list()
                deleted_dataset = True

        status_msg = (
            f"Model and dataset for \"{name}\" deleted."
            if deleted_dataset else f"Model \"{name}\" deleted."
        )
        self._lbl_status.setText(status_msg)
        self._lbl_status.setStyleSheet("color: rgba(255, 255, 255, 60); font-size: 11px; background: transparent;")
        # Auto-clear the message after 20s so it doesn't linger forever.
        # Captured text guards against clearing a later status set by other code.
        deleted_text = self._lbl_status.text()
        def _clear_if_unchanged():
            if self._lbl_status.text() == deleted_text:
                self._lbl_status.setText("")
        QTimer.singleShot(20_000, _clear_if_unchanged)

        # Targeted card update — don't full-rebuild the grid (causes overlap glitch)
        if dataset_existed and not deleted_dataset:
            # Dataset still exists, keep the card but refresh its image
            self._refresh_card_image(name)
        else:
            # No data left for this name — quietly remove just its card
            self._remove_card_widget(name)

    def _on_carousel_select(self, idx: int):
        """User picked an artist in the carousel — load that model."""
        if 0 <= idx < len(self._carousel._models):
            self._select_model(self._carousel._models[idx]["name"])

    def _remove_card_widget(self, name: str):
        """Carousel-era replacement: just rebuild the carousel from disk."""
        self._populate_existing_datasets()

    # Per-tier timing model, calibrated against the cost table:
    #   batch       — what the pod's optimize.py will set based on VRAM
    #   sec_per_step — observed wall-clock per training step on that GPU
    #   label        — short human name shown in the suggestion text
    _TIER_TIMING = {
        "cheapest":  {"batch": 96,  "sec_per_step": 4.5, "label": "A40"},
        "balanced":  {"batch": 96,  "sec_per_step": 2.8, "label": "RTX 6000 Ada"},
        "fast":      {"batch": 192, "sec_per_step": 2.5, "label": "A100 SXM"},
        "fastest":   {"batch": 192, "sec_per_step": 1.3, "label": "H100 SXM"},
    }

    def _epoch_count_for_duration(self, total_seconds: float) -> int:
        """Epoch count auto-picked for a fresh run from total clip
        duration — the table the Epochs '?' help describes."""
        if total_seconds <= 0:
            return 2000   # no clips yet — a neutral default
        if total_seconds < 180:
            return 3000
        if total_seconds < 300:
            return 2500
        if total_seconds < 600:
            return 1500
        if total_seconds < 1800:
            return 500
        return 300

    def _latest_checkpoint_epoch(self) -> int:
        """Highest epoch among the selected model's G_*.pth checkpoints,
        or 0 if it has none yet (a fresh model)."""
        name = (self._selected_name or "").strip()
        if not name:
            return 0
        best = 0
        try:
            for f in os.listdir(os.path.join(str(MODELS_DIR), name)):
                if f.startswith("G_") and f.endswith(".pth"):
                    num = f[2:-4]
                    if num.isdigit():
                        best = max(best, int(num))
        except OSError:
            pass
        return best

    def _auto_epoch_target(self, total_seconds: float = 0.0) -> int:
        """Epoch count training will auto-pick for the current model:
        the maturity-based resume target when it already has a
        checkpoint, else the fresh-run count from clip duration.
        Mirrors the auto logic in _start_training."""
        cur_ep = self._latest_checkpoint_epoch()
        if cur_ep <= 0:
            return self._epoch_count_for_duration(total_seconds)
        clips = max(len(self._clips), 1)
        meta = self._load_model_metadata(
            (self._selected_name or "").strip()
        ) or {}
        batch = int(meta.get("batch_size", 16) or 16)
        maturity = (cur_ep * batch) / clips
        if maturity >= 2000:
            target = max(maturity * 1.25, maturity + 200)
        elif maturity >= 800:
            target = 2000
        elif maturity >= 300:
            target = 800
        else:
            target = 300
        extra = int((target - maturity) * clips / max(batch, 1))
        return max(cur_ep + max(extra, 100), cur_ep + 100)

    def _update_auto_epoch_placeholder(self, total_seconds: float = 0.0):
        """Show the auto-picked epoch count as the box's dim placeholder
        so an empty (auto) box previews what training will use."""
        self._txt_epochs.setPlaceholderText(
            str(self._auto_epoch_target(total_seconds))
        )

    def _current_gpu_tier(self) -> str:
        try:
            from services.job_store import load_config
            return (load_config() or {}).get("preferred_gpu_tier", "cheapest")
        except Exception:
            return "cheapest"

    # RunPod GPU type names -> short labels for the GPU status line.
    _GPU_SHORT = {
        "NVIDIA A40": "A40",
        "NVIDIA RTX A6000": "RTX A6000",
        "NVIDIA RTX 6000 Ada Generation": "RTX 6000 Ada",
        "NVIDIA A100 80GB PCIe": "A100",
        "NVIDIA A100-SXM4-80GB": "A100 SXM",
        "NVIDIA H100 80GB HBM3": "H100",
    }

    def _short_gpu(self, full: str) -> str:
        """Shorten a RunPod GPU type name for the status line."""
        full = (full or "").strip()
        if full in self._GPU_SHORT:
            return self._GPU_SHORT[full]
        # Unknown name — drop the vendor prefix and any memory suffix.
        s = full.replace("NVIDIA ", "").strip()
        for cut in (" 80GB", " 48GB", " 24GB"):
            s = s.split(cut)[0]
        return s.strip() or full

    def _refresh_gpu_label(self):
        """GPU label above Start Training. Reflects the Settings choice; while
        a cloud run provisions, shows when the chosen GPU is unavailable and
        which GPU is being tried/used instead."""
        try:
            from services.job_store import load_config
            cfg = load_config() or {}
        except Exception:
            cfg = {}
        if cfg.get("train_locally"):
            self._lbl_gpu.setText("Training on this computer")
            return
        tier = cfg.get("preferred_gpu_tier", "cheapest")
        params = self._TIER_TIMING.get(tier, self._TIER_TIMING["cheapest"])
        chosen = params["label"]
        if getattr(self, "_gpu_chosen_unavail", False):
            active = getattr(self, "_gpu_active", None)
            if getattr(self, "_gpu_got", False) and active:
                second = f"using {active}"
            elif active:
                second = f"trying {active}…"
            else:
                second = "finding a GPU…"
            # Chosen GPU pinned on top with "unavailable"; the GPU actually
            # being tried / now in use sits on the line beneath it.
            self._lbl_gpu.setText(
                f'GPU - {chosen} '
                f'<span style="color:#e0a040;">· unavailable</span>'
                f'<br><span style="color:#9a9a9a;">{second}</span>'
            )
        else:
            self._lbl_gpu.setText(f"GPU - {chosen}")

    def _parse_gpu_provisioning(self, line: str):
        """Watch the cloud-provisioning log lines (emitted by RunPodClient's
        GPU-fallback chain) so the GPU label can show, live, when the chosen
        GPU is unavailable and which GPU is being tried/used instead."""
        import re
        s = line.strip()
        m = re.match(r'^Trying (.+?)\.\.\.\s*$', s)
        if m:
            self._gpu_active = self._short_gpu(m.group(1))
            self._refresh_gpu_label()
            return
        if re.match(r'^.+ unavailable, trying next', s):
            self._gpu_chosen_unavail = True
            self._gpu_active = None
            self._refresh_gpu_label()
            return
        m = re.match(r'^Got (.+?)!\s*$', s)
        if m:
            self._gpu_active = self._short_gpu(m.group(1))
            self._gpu_got = True
            self._refresh_gpu_label()
            return

    def showEvent(self, event):
        super().showEvent(event)
        # Keep the GPU label current with whatever Settings now says.
        self._refresh_gpu_label()

    def _estimate_training_time(
        self, total_epochs: int, current_epochs: int, clips: int,
        is_resume: bool = False, tier: str | None = None,
    ) -> tuple[str, str]:
        """Wall-clock estimate + GPU label for the user's selected tier.

        Returns (duration_string, gpu_label) — e.g. ("~42 min", "A40").
        Math: delta_epochs × batches_per_epoch × sec_per_step + ~6 min of
        pod boot/install/download overhead + ~3s per checkpoint save.
        """
        import math
        if tier is None:
            tier = self._current_gpu_tier()
        params = self._TIER_TIMING.get(tier, self._TIER_TIMING["cheapest"])
        label = params["label"]
        delta = max(0, total_epochs - max(current_epochs, 0))
        if delta <= 0:
            return "", label
        batches_per_epoch = max(1, math.ceil(max(clips, 1) / params["batch"]))
        train_sec = delta * batches_per_epoch * params["sec_per_step"]
        save_sec = (delta / 25) * 3
        overhead_sec = 6 * 60
        total_sec = train_sec + save_sec + overhead_sec
        return self._format_duration_minutes(total_sec), label

    # Back-compat shim — older call sites used the A40-specific name.
    def _estimate_training_time_a40(
        self, total_epochs: int, current_epochs: int, clips: int,
        is_resume: bool = False,
    ) -> str:
        eta, _ = self._estimate_training_time(
            total_epochs, current_epochs, clips, is_resume=is_resume,
            tier="cheapest",
        )
        return eta

    @staticmethod
    def _format_duration_minutes(seconds: float) -> str:
        m = int(seconds / 60)
        if m < 1:
            return "<1 min"
        if m < 60:
            return f"~{m} min"
        h = m // 60
        rem = m % 60
        return f"~{h}h {rem}min" if rem else f"~{h}h"

    def _load_model_metadata(self, name: str) -> dict:
        """Load metadata.json for a model, or build a minimal one from disk."""
        from services.paths import MODELS_DIR
        import json
        meta_path = os.path.join(str(MODELS_DIR), name, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    return json.load(f) or {}
            except Exception:
                pass
        # Try the model inspector for downloaded RVC/SVC checkpoints
        try:
            from services.model_inspector import inspect_model
            model_dir = os.path.join(str(MODELS_DIR), name)
            if os.path.isdir(model_dir):
                return inspect_model(model_dir) or {}
        except Exception:
            pass
        return {}

    def _add_grade_badge(self, img_lbl, name: str):
        """No-op now that the carousel handles its own painting."""
        return

    def _refresh_card_image(self, name: str):
        """Carousel-era replacement: rebuild the carousel from disk."""
        self._populate_existing_datasets()

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
        from services.paths import DATASETS_DIR
        # Before switching away, save the current artist's staged clips
        # so they persist across selections (until training creates a real
        # dataset dir).
        prev = getattr(self, "_selected_name", "")
        if prev and prev != name:
            prev_dataset = os.path.join(str(DATASETS_DIR), prev)
            if not os.path.isdir(prev_dataset):
                # Pending artist — keep their working clip list around
                self._pending_clips_by_artist[prev] = list(self._clips)

        self._selected_name = name
        self._lbl_selected.setText(name)
        self._lbl_selected.setStyleSheet("color: rgba(255, 255, 255, 80); font-size: 12px; font-weight: bold; background: transparent;")
        self._txt_new_name.clear()
        self._update_panel_background(name)
        # Kick off Spotify lookup if we don't already have an image for this name
        self._fetch_artist_image_async(name)
        # Load existing clips: dataset dir on disk wins, else fall back to
        # whatever was staged for this name during the current session.
        dataset_dir = os.path.join(str(DATASETS_DIR), name)
        if os.path.isdir(dataset_dir):
            self._clips = [os.path.join(dataset_dir, f) for f in sorted(os.listdir(dataset_dir))
                           if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))]
            # Promote: if we had pending clips for this name from this
            # session, the on-disk version is the truth now — drop them.
            self._pending_clips_by_artist.pop(name, None)
        else:
            self._clips = list(self._pending_clips_by_artist.get(name, []))
        # Rebuild the file → clips grouping from clip filenames so the
        # expandable tree survives an app restart or an artist switch.
        self._rebuild_processed_files_from_clips()
        self._refresh_file_list()
        # Highlight selected in grid
        self._update_grid_selection()
        # Show "Continue Training" if model has existing checkpoints
        self._check_existing_model(name)
        # Bottom-left summary: duration + epochs + recommendation
        self._update_model_info(name)
        self._update_metadata_panel(name)

    def _check_existing_model(self, name):
        """Continue-Training requires an SVC G_*.pth; Delete works on any .pth file."""
        from services.paths import MODELS_DIR
        model_dir = os.path.join(str(MODELS_DIR), name)
        has_svc_checkpoint = False
        has_any_model = False
        if os.path.isdir(model_dir):
            files = os.listdir(model_dir)
            has_svc_checkpoint = any(f.startswith("G_") and f.endswith(".pth") for f in files)
            has_any_model = any(f.endswith(".pth") for f in files)
        # Continue Training only makes sense for SVC checkpoints we know how to resume
        self._btn_continue_train.setVisible(has_svc_checkpoint)
        # Delete Model is available for any trained model (SVC or RVC)
        self._btn_delete_model.setVisible(has_any_model)

    def _fetch_artist_image_async(self, name: str):
        """Try Spotify in the background, drop the result into the artist
        thumb cache, then refresh the carousel + panel background.
        No-op if we already have a cached image for this name."""
        if not name:
            return
        os.makedirs(self._image_cache_dir, exist_ok=True)
        thumb_path = os.path.join(self._image_cache_dir, f"{name}.jpg")
        if os.path.exists(thumb_path):
            return

        from PyQt6.QtCore import QThread, pyqtSignal

        class _ImgFetcher(QThread):
            done = pyqtSignal(str, str)  # name, saved_path

            def __init__(self, artist, save_path):
                super().__init__()
                self.artist = artist
                self.save_path = save_path

            def run(self):
                try:
                    from services.spotify_client import SpotifyClient
                    client = SpotifyClient(
                        os.environ.get("SOMERSVC_SPOTIFY_ID", ""),
                        os.environ.get("SOMERSVC_SPOTIFY_SECRET", ""),
                    )
                    if client.download_artist_image(self.artist, self.save_path):
                        self.done.emit(self.artist, self.save_path)
                except Exception:
                    pass

        if not hasattr(self, "_img_workers"):
            self._img_workers = []
        fetcher = _ImgFetcher(name, thumb_path)
        fetcher.done.connect(self._on_artist_image_fetched)
        # Hold a reference so it isn't GC'd before run() finishes
        self._img_workers.append(fetcher)
        fetcher.start()

    def _on_artist_image_fetched(self, name: str, path: str):
        """A background fetch completed — patch the carousel + background."""
        if not path or not os.path.exists(path):
            return
        pix = QPixmap(path)
        if pix.isNull():
            return
        # Patch any carousel entry with this name
        if hasattr(self, "_carousel"):
            for m in self._carousel._models:
                if m.get("name") == name:
                    m["pixmap"] = pix
                    self._carousel._circular_cache.clear()
                    self._carousel.update()
                    break
        # Refresh the panel wallpaper if the user still has this artist selected
        if getattr(self, "_selected_name", "") == name:
            self._update_panel_background(name)

    def _on_new_name_entered(self):
        """User typed a new artist name and pressed Enter."""
        name = self._txt_new_name.text().strip()
        if not name:
            return
        # Persist the previous artist's staged clips before switching
        from services.paths import DATASETS_DIR
        prev = getattr(self, "_selected_name", "")
        if prev and prev != name:
            prev_dataset = os.path.join(str(DATASETS_DIR), prev)
            if not os.path.isdir(prev_dataset):
                self._pending_clips_by_artist[prev] = list(self._clips)
        self._selected_name = name
        self._lbl_selected.setText(name)
        self._lbl_selected.setStyleSheet("color: rgba(255, 255, 255, 80); font-size: 12px; font-weight: bold; background: transparent;")
        # Restore pending clips for this name if we already had some staged
        self._clips = list(self._pending_clips_by_artist.get(name, []))
        self._refresh_file_list()

        # Add a temporary "pending" entry to the carousel so the user sees
        # their new artist's placeholder (initials circle, dimmed) instead
        # of the previously-selected artist's image. Skip if a card with
        # the same name already exists.
        if hasattr(self, "_carousel"):
            existing = next(
                (i for i, m in enumerate(self._carousel._models) if m["name"] == name),
                None,
            )
            if existing is None:
                # Prefer a previously-cached artist thumb, then the
                # Best-Match wallpaper as a last resort placeholder.
                placeholder = None
                cached = os.path.join(self._image_cache_dir, f"{name}.jpg")
                if os.path.exists(cached):
                    pix = QPixmap(cached)
                    if not pix.isNull():
                        placeholder = pix
                if placeholder is None:
                    best_bg = os.path.join(APP_DIR, "assets", "best_match.png")
                    if os.path.exists(best_bg):
                        pix = QPixmap(best_bg)
                        if not pix.isNull():
                            placeholder = pix
                models = list(self._carousel._models) + [{
                    "name": name,
                    "dir": "",
                    "pixmap": placeholder,
                    "vocal_key": "",
                    "pending": True,
                }]
                self._carousel.set_models(models)
                idx = len(models) - 1
            else:
                idx = existing
            self._carousel.blockSignals(True)
            try:
                self._carousel.select(idx)
            finally:
                self._carousel.blockSignals(False)

        self._btn_continue_train.setVisible(False)
        self._btn_delete_model.setVisible(False)
        self._lbl_model_info.setVisible(False)
        # No image yet for a brand-new name — clear any leftover background
        self._update_panel_background("")
        # Kick off a Spotify lookup so the placeholder swaps in the real
        # artist photo within a second or two if the name matches a Spotify artist.
        self._fetch_artist_image_async(name)

    def _update_model_info(self, name: str):
        """Populate the bottom-left summary for the selected model."""
        if not name:
            self._lbl_model_info.setVisible(False)
            return
        metadata = self._load_model_metadata(name)

        # Metadata records what was used at the LAST training run. The dataset
        # folder may have changed since (clips added/removed), so always
        # measure live duration + clip count off disk for the recommendation.
        from services.paths import DATASETS_DIR
        ds_dir = os.path.join(str(DATASETS_DIR), name)
        live_duration = 0.0
        live_clips = 0
        if os.path.isdir(ds_dir):
            try:
                import soundfile as _sf
                for f in os.listdir(ds_dir):
                    if not f.endswith((".wav", ".flac", ".mp3", ".ogg")):
                        continue
                    full = os.path.join(ds_dir, f)
                    try:
                        live_duration += _sf.info(full).duration
                        live_clips += 1
                    except Exception:
                        pass
            except Exception:
                pass

        epochs = int((metadata or {}).get("epochs", 0) or 0)
        meta_duration = float((metadata or {}).get("dataset_duration_s", 0) or 0)
        meta_clips = int((metadata or {}).get("dataset_clips", 0) or 0)
        batch = int((metadata or {}).get("batch_size", 16) or 16)

        # Use live values when we have them, fall back to metadata otherwise
        duration = live_duration if live_duration > 0 else meta_duration
        clips = live_clips if live_clips > 0 else meta_clips

        if epochs <= 0 and duration <= 0:
            self._lbl_model_info.setVisible(False)
            return

        if duration >= 3600:
            h = int(duration // 3600)
            m = int((duration % 3600) // 60)
            s = int(duration % 60)
            dur_str = f"{h}:{m:02d}:{s:02d}"
        else:
            m = int(duration // 60)
            s = int(duration % 60)
            dur_str = f"{m}:{s:02d}"

        # Audio score: dataset duration in minutes
        dur_score = 3 if duration >= 600 else (
            2 if duration >= 300 else (1 if duration >= 120 else 0)
        )
        # Maturity: how many data passes the model has had per clip
        maturity = (epochs * batch / clips) if clips > 0 else 0
        train_score = 3 if maturity >= 2000 else (
            2 if maturity >= 800 else (1 if maturity >= 300 else 0)
        )

        # Suggestion phrased as "do X to reach rank Y" so the user knows
        # exactly what pushes the badge up one level. When the suggestion
        # involves adding audio, the matching epoch target is bundled in
        # because more clips dilute the maturity ratio — the new dataset
        # still needs training to hit the rank.
        GRADE_LETTERS = {6: "S", 5: "A+", 4: "A", 3: "B+",
                         2: "B", 1: "C", 0: "D"}
        AUDIO_TARGET_SECS = {0: 120, 1: 300, 2: 600}      # to score 1, 2, 3
        TRAIN_TARGET_MATURITY = {0: 300, 1: 800, 2: 2000}  # ditto

        total_score = dur_score + train_score
        current_grade = GRADE_LETTERS.get(total_score, "?")

        def _audio_plus_train_suggestion(target_total: int) -> str:
            # Audio: minutes to reach the next dur_score threshold.
            target_seconds = AUDIO_TARGET_SECS[dur_score]
            extra = max(60, target_seconds - duration)
            mins = max(1, int(round(extra / 60)))
            mins_str = f"{mins} more minute{'s' if mins != 1 else ''}"

            # New clip count after adding audio (estimated from current
            # average clip length, falling back to ~7s chunks).
            if duration > 0 and clips > 0:
                avg_clip = duration / clips
                new_clips = max(clips + 1, int(target_seconds / max(avg_clip, 1)))
            else:
                new_clips = max(clips, int(target_seconds / 7))

            # After audio bump, new dur_score = dur_score + 1. Training axis
            # must stay at (target_total - new_dur_score) to land on the rank.
            new_dur_score = dur_score + 1
            needed_train_score = max(0, target_total - new_dur_score)
            if needed_train_score >= 1:
                req_maturity = TRAIN_TARGET_MATURITY[needed_train_score - 1]
                req_epochs = int(req_maturity * new_clips / max(batch, 1))
                req_epochs = max(req_epochs, epochs + 100)
                eta, gpu_label = self._estimate_training_time(
                    req_epochs, current_epochs=epochs, clips=new_clips,
                    is_resume=True,
                )
                eta_str = f" ({eta} on {gpu_label})" if eta else ""
                return (
                    f"Add {mins_str} of audio and train to "
                    f"~{req_epochs:,} epochs{eta_str}"
                )
            return f"Add {mins_str} of audio"

        def _train_only_suggestion(target_total: int) -> str:
            needed_train_score = max(1, target_total - dur_score)
            target_mat = TRAIN_TARGET_MATURITY[needed_train_score - 1]
            extra_epochs = (
                int((target_mat - maturity) * clips / max(batch, 1))
                if clips > 0 else 0
            )
            target_epochs = max(epochs + max(extra_epochs, 100), epochs + 100)
            eta, gpu_label = self._estimate_training_time(
                target_epochs, current_epochs=epochs, clips=clips, is_resume=True,
            )
            eta_str = f" ({eta} on {gpu_label})" if eta else ""
            return f"Continue training to ~{target_epochs:,} epochs{eta_str}"

        def _grade_badge_html(grade: str, height: int = 20) -> str:
            """Inline <img> tag for the grade badge so the suggestion shows
            the actual badge artwork instead of plain text. Falls back to
            the grade letter if the asset is missing.

            margin-bottom shifts the image up 3px so it sits on the same
            visual line as the surrounding text (Qt rich text aligns the
            bottom of the image with the text baseline by default, which
            looks slightly low for badges with internal padding).
            """
            path = os.path.join(APP_DIR, "assets", "grade_badges", f"{grade}.png")
            if not os.path.exists(path):
                return grade
            return (
                f'<img src="{path}" height="{height}" '
                f'style="vertical-align:middle; margin-bottom:3px;" />'
            )

        if total_score >= 6:
            rec = (
                f"{_grade_badge_html(current_grade)} (max) — try the model "
                f"first; only retrain if quality is off."
            )
            rec_color = "rgba(80, 200, 120, 200)"
        else:
            next_total = total_score + 1
            next_grade = GRADE_LETTERS[next_total]
            badge_html = _grade_badge_html(next_grade)
            # Pick the axis with more headroom. Ties go to training because
            # the user doesn't need to gather more audio for it.
            bump_audio_first = (
                dur_score < train_score
                or (dur_score == train_score and train_score >= 3)
            )
            if bump_audio_first and dur_score < 3:
                rec = f"{_audio_plus_train_suggestion(next_total)} to reach {badge_html}."
                rec_color = "rgba(245, 158, 11, 200)"
            elif train_score < 3 and clips > 0:
                rec = f"{_train_only_suggestion(next_total)} to reach {badge_html}."
                rec_color = "rgba(85, 153, 255, 220)"
            elif dur_score < 3:
                # Training already at 3/3 but audio still has headroom.
                rec = f"{_audio_plus_train_suggestion(next_total)} to reach {badge_html}."
                rec_color = "rgba(245, 158, 11, 200)"
            else:
                rec = (
                    f"{_grade_badge_html(current_grade)} — try it before "
                    f"adding more data."
                )
                rec_color = "rgba(80, 200, 120, 200)"

        # Note when the dataset has grown since training so the user knows
        # the epoch count is referring to the older snapshot.
        out_of_date = (
            meta_duration > 0
            and live_duration > 0
            and abs(live_duration - meta_duration) > max(15.0, meta_duration * 0.05)
        )
        epochs_str = f"{epochs:,} epochs" if epochs > 0 else "no epochs yet"
        clips_str = f"{clips} clip{'s' if clips != 1 else ''}"
        if duration > 0:
            top = f"<b>{dur_str}</b> · <b>{clips_str}</b> · <b>{epochs_str}</b>"
            if out_of_date and epochs > 0:
                top += "  <span style='color:rgba(245,158,11,180);'>(dataset has changed since last training)</span>"
        else:
            top = f"<b>{epochs_str}</b>"
        html = (
            f"<div style='font-size:11px;'>"
            f"<span style='color:rgba(255,255,255,90);'>{top}</span><br>"
            f"<span style='color:{rec_color};'>{rec}</span>"
            f"</div>"
        )
        self._lbl_model_info.setText(html)
        self._lbl_model_info.setVisible(True)

    def _on_meta_link_hover(self, link: str) -> None:
        """Show / hide the maturity help tooltip when hovering the (?) icon."""
        from PyQt6.QtWidgets import QToolTip
        from PyQt6.QtGui import QCursor
        if link == "maturity-help" and self._maturity_help_html:
            QToolTip.showText(
                QCursor.pos(), self._maturity_help_html, self._lbl_meta_panel
            )
        else:
            QToolTip.hideText()

    def _on_meta_link_clicked(self, link: str) -> None:
        """Handle clicks on inline links in the metadata panel."""
        if link == "rank-override":
            self._show_rank_override_menu()
        elif link == "vocal-key-edit":
            self._edit_vocal_key()
        # vocal-key-ignore link removed from the panel; handler left out
        # so any stale clicks are silently no-op.

    def _edit_vocal_key(self) -> None:
        """Prompt for a new vocal-key note (e.g. 'D4') and persist it."""
        name = (self._selected_name or "").strip()
        if not name:
            return
        from PyQt6.QtWidgets import QInputDialog
        from services.paths import MODELS_DIR
        meta_path = os.path.join(str(MODELS_DIR), name, "metadata.json")
        if not os.path.exists(meta_path):
            return
        import json as _json
        try:
            with open(meta_path) as f:
                meta = _json.load(f) or {}
        except Exception:
            meta = {}
        current = (meta.get("vocal_key") or "").strip()
        new_key, ok = QInputDialog.getText(
            self,
            "Edit vocal key",
            "Enter a note (e.g. D4, F#3). Leave blank to clear.",
            text=current,
        )
        if not ok:
            return
        new_key = new_key.strip()
        if new_key and _note_to_hz(new_key) <= 0:
            self._lbl_status.setText(
                f"'{new_key}' is not a valid note — try D4, F#3, etc."
            )
            return
        meta["vocal_key"] = new_key
        try:
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=2)
        except Exception as e:
            self._lbl_status.setText(f"Could not save key: {e}")
            return
        # Reload the carousel-side vocal_key so Best Match sees the new
        # value, then refresh the metadata panel + transpose calc.
        self._sync_carousel_vocal_key(name, new_key)
        self._detect_model_key()
        self._update_metadata_panel(name)

    def _toggle_ignore_vocal_key(self) -> None:
        """Flip ignore_vocal_key on the selected model and refresh.
        When ignored, this model is excluded from Best Match selection
        and no auto-transpose runs."""
        name = (self._selected_name or "").strip()
        if not name:
            return
        from services.paths import MODELS_DIR
        meta_path = os.path.join(str(MODELS_DIR), name, "metadata.json")
        if not os.path.exists(meta_path):
            return
        import json as _json
        try:
            with open(meta_path) as f:
                meta = _json.load(f) or {}
        except Exception:
            meta = {}
        new_flag = not bool(meta.get("ignore_vocal_key", False))
        meta["ignore_vocal_key"] = new_flag
        try:
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=2)
        except Exception as e:
            self._lbl_status.setText(f"Could not save: {e}")
            return
        self._sync_carousel_ignore_key(name, new_flag)
        if new_flag:
            self._model_center_hz = 0
            self._update_transpose_info()
        else:
            self._detect_model_key()
        self._update_metadata_panel(name)

    def _sync_carousel_vocal_key(self, name: str, key: str) -> None:
        """Update the carousel's in-memory vocal_key for `name` so
        Best Match resolution uses the new value without a full reload."""
        try:
            for m in getattr(self._carousel, "_models", []) or []:
                if m.get("name") == name:
                    m["vocal_key"] = key
                    break
        except Exception:
            pass

    def _sync_carousel_ignore_key(self, name: str, ignore: bool) -> None:
        """Update the carousel's in-memory ignore_vocal_key for `name` and
        repaint so the badge picks up the new strikethrough / muted state."""
        try:
            for m in getattr(self._carousel, "_models", []) or []:
                if m.get("name") == name:
                    m["ignore_vocal_key"] = ignore
                    break
            self._carousel.update()
        except Exception:
            pass

    def _show_rank_override_menu(self) -> None:
        """Pop up a small menu so the user can manually pick a rank for a
        downloaded model. Persists the choice to metadata.json under
        `user_grade_override` and refreshes the carousel + panel.
        """
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QCursor
        menu = QMenu(self)
        for letter in ["S", "A+", "A", "B+", "B", "C", "D"]:
            act = menu.addAction(f"Rank {letter}")
            act.triggered.connect(
                lambda _checked=False, g=letter: self._set_rank_override(g)
            )
        menu.addSeparator()
        clear = menu.addAction("Clear override (use auto)")
        clear.triggered.connect(
            lambda _checked=False: self._set_rank_override(None)
        )
        menu.exec(QCursor.pos())

    def _set_rank_override(self, grade) -> None:
        """Write or clear the rank override on the selected model's
        metadata.json, then refresh dependent UI."""
        name = (self._selected_name or "").strip()
        if not name:
            return
        from services.paths import MODELS_DIR
        meta_path = os.path.join(str(MODELS_DIR), name, "metadata.json")
        if not os.path.exists(meta_path):
            return
        try:
            import json as _json
            with open(meta_path) as f:
                meta = _json.load(f) or {}
            if grade:
                meta["user_grade_override"] = grade
            else:
                meta.pop("user_grade_override", None)
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=2)
        except Exception as e:
            self._lbl_status.setText(f"Could not save rank: {e}")
            return
        # Refresh: metadata panel uses the new value; carousel re-pulls
        # grades when populating.
        self._update_metadata_panel(name)
        self._populate_existing_datasets()

    def _build_maturity_help_html(self, epochs: int, batch: int,
                                   clips: int, maturity: float) -> str:
        """Tooltip body shown when the user hovers the (?) by Maturity.
        Highlights the tier the model is currently in, and shows the
        calculation just above the tier list so the number isn't a black box.
        """
        if clips <= 0 or batch <= 0:
            return ""
        calc = (
            f"<span style='color:rgba(255,255,255,140);'>"
            f"epochs &times; batch &divide; clips:</span><br>"
            f"<b>{epochs:,} &times; {batch} &divide; {clips} = "
            f"{int(maturity):,}</b>"
        )

        TIERS = [
            (0, 800, "keep training, model isn't ready yet"),
            (800, 2000,
             "it's usable; might be worth continuing for the last bit of polish"),
            (2000, None,
             "diminishing returns from more training. If quality still isn't "
             "there, the answer is more or better audio, not more epochs"),
        ]
        rows = []
        for low, high, msg in TIERS:
            if high is None:
                label = f"{low:,}+"
                in_tier = maturity >= low
            elif low == 0:
                label = f"Below {high:,}"
                in_tier = maturity < high
            else:
                label = f"{low:,}–{high:,}"
                in_tier = low <= maturity < high
            if in_tier:
                rows.append(
                    f"<span style='color:#2DD4BF;'>"
                    f"&rarr; <b>{label}</b> &rarr; {msg}</span>"
                )
            else:
                rows.append(
                    f"<span style='color:rgba(255,255,255,150);'>"
                    f"{label} &rarr; {msg}</span>"
                )
        return (
            f"<div style='font-size:11px;line-height:1.6em;max-width:340px;'>"
            f"{calc}<br><br>{'<br>'.join(rows)}</div>"
        )

    def _update_metadata_panel(self, name: str):
        """Populate the left-side panel with detailed metadata for the
        selected artist's trained model."""
        if not getattr(self, "_lbl_meta_panel", None):
            return
        if not name:
            self._lbl_meta_panel.setText("")
            return

        from services.paths import DATASETS_DIR
        metadata = self._load_model_metadata(name) or {}

        # Live counts off disk so they stay accurate after add/remove
        ds_dir = os.path.join(str(DATASETS_DIR), name)
        live_duration = 0.0
        live_clips = 0
        if os.path.isdir(ds_dir):
            try:
                import soundfile as _sf
                for f in os.listdir(ds_dir):
                    if not f.endswith((".wav", ".flac", ".mp3", ".ogg")):
                        continue
                    full = os.path.join(ds_dir, f)
                    try:
                        live_duration += _sf.info(full).duration
                        live_clips += 1
                    except Exception:
                        pass
            except Exception:
                pass

        epochs = int(metadata.get("epochs", 0) or 0)
        meta_duration = float(metadata.get("dataset_duration_s", 0) or 0)
        meta_clips = int(metadata.get("dataset_clips", 0) or 0)
        batch = int(metadata.get("batch_size", 0) or 0)
        vocal_key = metadata.get("vocal_key", "") or ""
        checkpoint = metadata.get("checkpoint", "") or ""
        trained_at = metadata.get("trained_at", "") or ""

        # Pending (no checkpoint yet) — show "ready to train"
        if epochs <= 0 and not checkpoint:
            if live_clips > 0:
                m, s = divmod(int(live_duration), 60)
                lines = [
                    f"<b>{name}</b>",
                    "<span style='color:rgba(245,158,11,200);'>Not trained yet</span>",
                    "",
                    f"Clips staged: <b>{live_clips}</b>",
                    f"Duration: <b>{m}:{s:02d}</b>",
                ]
            else:
                lines = [
                    f"<b>{name}</b>",
                    "<span style='color:rgba(255,255,255,50);'>No data yet</span>",
                ]
            self._lbl_meta_panel.setText(
                "<div style='line-height:1.6em;'>" + "<br>".join(lines) + "</div>"
            )
            return

        # Format duration → "Hh Mm" or "M:SS"
        duration = live_duration if live_duration > 0 else meta_duration
        if duration >= 3600:
            h = int(duration // 3600)
            m = int((duration % 3600) // 60)
            dur_str = f"{h}h {m:02d}m"
        else:
            m = int(duration // 60)
            s = int(duration % 60)
            dur_str = f"{m}:{s:02d}"

        clips = live_clips if live_clips > 0 else meta_clips
        maturity = (epochs * batch / clips) if (clips > 0 and batch > 0) else 0

        # Grade badge color hex from voice_card helper
        try:
            from ui.widgets.voice_card import grade_for_metadata
            grade, grade_color, _tip = grade_for_metadata(metadata)
        except Exception:
            grade, grade_color = ("", "")

        # "Trained at" → friendlier date
        trained_str = ""
        if trained_at and trained_at != "unknown":
            try:
                from datetime import datetime
                # Stored as ISO 8601 (UTC). Strip TZ for display.
                dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
                trained_str = dt.strftime("%Y-%m-%d")
            except Exception:
                trained_str = trained_at[:10]

        rows = [f"<b style='font-size:13px;'>{name}</b>"]
        # Downloaded models carry no auto-grade — the user can rate one
        # by clicking, and a rated grade then shows. Auto-trained models
        # keep a read-only rank derived from their training data.
        is_downloaded = (
            clips == 0
            and (metadata.get("sample_rate") or metadata.get("rvc_version"))
        )
        if grade:
            if is_downloaded:
                rows.append(
                    f"<span style='color:{grade_color};font-weight:600;'>"
                    f"<a href=\"rank-override\" "
                    f"style=\"text-decoration:none;color:{grade_color};\">"
                    f"Rank {grade} ▾</a></span>"
                )
            else:
                rows.append(
                    f"<span style='color:{grade_color};font-weight:600;'>"
                    f"Rank {grade}</span>"
                )
        elif is_downloaded:
            rows.append(
                "<span style='font-weight:600;'>"
                "<a href=\"rank-override\" "
                "style=\"text-decoration:none;color:#888;\">"
                "Rate this model ▾</a></span>"
            )
        rows.append("")  # blank line

        # Cycle the field-name color in a 4-step palette down the rows so
        # the panel reads as a colorful at-a-glance dashboard.
        # Green → Teal → Blue → Purple, repeating.
        LABEL_COLORS = ["#22C55E", "#2DD4BF", "#5599FF", "#A855F7"]
        row_idx = [0]

        def _row(label, value, suffix=""):
            color = LABEL_COLORS[row_idx[0] % 4]
            row_idx[0] += 1
            return (
                f"<span style='color:{color};'>{label}</span> "
                f"<b>{value}</b>{suffix}"
            )

        rows.append(_row("Epochs", f"{epochs:,}"))
        if duration > 0:
            rows.append(_row("Duration", dur_str))
        if clips > 0:
            rows.append(_row("Clips", f"{clips}"))
        if batch > 0:
            rows.append(_row("Batch", f"{batch}"))
        if maturity > 0:
            # Inline (?) help icon — hovering shows the tier breakdown +
            # the live calculation. Stored on the panel so the linkHovered
            # handler can read it.
            self._maturity_help_html = self._build_maturity_help_html(
                epochs, batch, clips, maturity,
            )
            help_icon = (
                ' <a href="maturity-help" style="text-decoration:none;'
                'color:rgba(255,255,255,90);">'
                '<span style="border:1px solid currentColor;'
                'border-radius:7px;padding:0 4px;font-size:9px;">?</span>'
                '</a>'
            )
            rows.append(_row("Maturity", f"{int(maturity):,}", help_icon))
        if vocal_key:
            # Click "D4" → inline edit prompt.
            # NOTE: "Ignore this key" toggle row removed — too easy to flip
            # accidentally, and old metadata flags are now inert (see also
            # _detect_model_key, _find_best_match, carousel badge paint).
            # _toggle_ignore_vocal_key / _sync_carousel_ignore_key are kept
            # as dead code so the feature can be revived without rewriting.
            value_html = (
                f'<a href="vocal-key-edit" '
                f'style="text-decoration:none;color:inherit;">'
                f'{vocal_key} ▾</a>'
            )
            rows.append(_row("Vocal key", value_html))
        if checkpoint:
            rows.append(_row("Checkpoint", checkpoint))
        if trained_str:
            rows.append(_row("Trained", trained_str))

        # Flag "out of date" when the dataset on disk has drifted from the
        # snapshot used at training time.
        out_of_date = (
            meta_duration > 0
            and live_duration > 0
            and abs(live_duration - meta_duration) > max(15.0, meta_duration * 0.05)
        )
        if out_of_date:
            rows.append("")
            rows.append(
                "<span style='color:rgba(245,158,11,180);font-size:10px;'>"
                "Dataset has changed since last training.</span>"
            )

        self._lbl_meta_panel.setText(
            "<div style='line-height:1.7em;'>" + "<br>".join(rows) + "</div>"
        )

    def _update_grid_selection(self):
        """Move the carousel to the currently-selected name (no-op if absent).

        Uses blockSignals so the call doesn't re-emit model_selected and
        re-enter _on_carousel_select → _select_model → here → ... (infinite
        recursion → SIGABRT)."""
        if not getattr(self, "_carousel", None):
            return
        for i, m in enumerate(self._carousel._models):
            if m["name"] == self._selected_name:
                if i != getattr(self._carousel, "_selected", -1):
                    self._carousel.blockSignals(True)
                    try:
                        self._carousel.select(i)
                    finally:
                        self._carousel.blockSignals(False)
                return

    def _populate_existing_datasets(self):
        """Feed the carousel with all known model/dataset names."""
        from services.paths import MODELS_DIR, DATASETS_DIR

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
        # Include in-memory pending artists (typed names with staged clips
        # but no folder on disk yet) so they survive panel close/reopen.
        for n in getattr(self, "_pending_clips_by_artist", {}).keys():
            if n:
                names.add(n)

        models = []
        for name in sorted(names, key=str.casefold):
            pixmap = None
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                p = os.path.join(models_dir, name, f"image{ext}")
                if os.path.exists(p):
                    pixmap = QPixmap(p)
                    break
            if pixmap is None:
                thumb = os.path.join(self._image_cache_dir, f"{name}.jpg")
                if os.path.exists(thumb):
                    pixmap = QPixmap(thumb)

            vocal_key = ""
            metadata = {}
            meta_path = os.path.join(models_dir, name, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    import json as _json
                    with open(meta_path) as _f:
                        metadata = _json.load(_f) or {}
                    vocal_key = metadata.get("vocal_key", "")
                except Exception:
                    pass

            # Compute the letter-grade badge so the carousel can paint it.
            try:
                from ui.widgets.voice_card import grade_for_metadata
                grade, grade_color, grade_tip = grade_for_metadata(metadata)
            except Exception:
                grade, grade_color, grade_tip = ("", "", "")

            # An entry is "pending" if no trained checkpoint exists yet —
            # the carousel renders these at 20% opacity to signal "name
            # reserved, model not trained yet".
            artist_dir = os.path.join(models_dir, name)
            has_checkpoint = os.path.isdir(artist_dir) and any(
                f.endswith(".pth") for f in os.listdir(artist_dir)
            )
            pending = not has_checkpoint

            # Pending entries without their own image fall back to the
            # Best-Match wallpaper so the carousel matches the background.
            if pending and pixmap is None:
                best_bg = os.path.join(APP_DIR, "assets", "best_match.png")
                if os.path.exists(best_bg):
                    pixmap = QPixmap(best_bg)

            models.append({
                "name": name,
                "dir": artist_dir,
                "pixmap": pixmap,
                "vocal_key": vocal_key,
                "pending": pending,
                "grade": grade,
                "grade_color": grade_color,
                "grade_tooltip": grade_tip,
            })

        self._carousel.set_models(models)
        # Restore the previously-selected artist if it still exists; if not
        # (panel just opened for the first time), default to the first
        # artist so the metadata panel + clip list aren't empty.
        target_idx = -1
        if self._selected_name:
            for i, m in enumerate(models):
                if m["name"] == self._selected_name:
                    target_idx = i
                    break
        if target_idx < 0 and models:
            target_idx = 0
        if target_idx >= 0:
            # blockSignals so we don't bounce back through _on_carousel_select
            # while the carousel reflects its new selection.
            self._carousel.blockSignals(True)
            try:
                self._carousel.select(target_idx)
            finally:
                self._carousel.blockSignals(False)
            # Drive the dependent UI (metadata panel, clip list, buttons)
            # via _select_model — which is what _on_carousel_select would
            # have called if signals weren't blocked.
            self._select_model(models[target_idx]["name"])
        self._txt_new_name.clear()

    def _demucs_model_present(self) -> bool:
        """True if Demucs's htdemucs weights are already cached locally.

        Demucs uses torch.hub which caches under ~/.cache on macOS too
        (NOT ~/Library/Caches). Check both for safety.
        """
        import glob
        candidates = [
            os.path.expanduser("~/.cache/torch/hub/checkpoints"),
            os.path.expanduser("~/Library/Caches/torch/hub/checkpoints"),
            os.path.expanduser("~/.cache/torch/hub/demucs"),
        ]
        # htdemucs weights ship as hashed .th files (e.g. 955717e8-*.th)
        for d in candidates:
            if glob.glob(os.path.join(d, "*.th")):
                return True
        return False

    def _isolate_vocals(self):
        # Operate on whatever the user has selected in the file list.
        paths = self._selected_clip_paths()
        if not paths:
            return

        # Skip files that are already isolated outputs — there's nothing
        # useful to extract from them.
        already = [p for p in paths if "_Isolated_Vocals" in os.path.basename(p)
                   or "_isolated" in os.path.basename(p).lower()]
        paths = [p for p in paths if p not in already]
        if not paths:
            QMessageBox.information(
                self, "Already Isolated",
                "The selected clip(s) are already isolated vocals — "
                "nothing more to extract."
            )
            return

        # Confirm — also asks whether to drop the pre-isolation source
        # clips from the file list once the isolated vocals come back.
        n = len(paths)
        sample = "\n".join("  • " + os.path.basename(p) for p in paths[:5])
        more = f"\n  ...and {n - 5} more" if n > 5 else ""
        from PyQt6.QtCore import Qt as _Qt
        from PyQt6.QtWidgets import (
            QDialog, QLabel, QCheckBox, QDialogButtonBox, QVBoxLayout,
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Isolate Vocals")
        dlg.setModal(True)
        dlg.setMinimumWidth(420)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(20, 20, 20, 16)
        v.setSpacing(10)
        msg = QLabel(
            f"Isolate vocals from {n} song{'s' if n != 1 else ''}?\n\n"
            f"{sample}{more}\n\n"
            f"Each song takes ~30-90 seconds depending on length."
        )
        msg.setStyleSheet("color: #ddd; font-size: 12px;")
        msg.setWordWrap(True)
        v.addWidget(msg)
        chk_replace = QCheckBox(
            f"Remove original {'song' if n == 1 else 'songs'} after isolation"
        )
        chk_replace.setChecked(True)  # default: keep only the isolated vocals
        chk_replace.setStyleSheet("color: #ccc; font-size: 12px;")
        v.addWidget(chk_replace)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Isolate")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # Stash the per-source-path replacement preference so _on_iso_done
        # can act on it after the worker finishes.
        self._iso_replace_originals = chk_replace.isChecked()
        self._iso_source_paths = list(paths)

        first_run = not self._demucs_model_present()
        if first_run:
            from PyQt6.QtWidgets import QMessageBox
            ok = QMessageBox.information(
                self, "One-time Vocal Model Download",
                "The vocal isolation model needs to download once "
                "(about 80 MB). Subsequent runs are instant.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if ok != QMessageBox.StandardButton.Yes:
                return

        from services.vocal_separator import VocalSeparator
        import tempfile
        self._iso_dir = tempfile.mkdtemp(prefix="svc_iso_")

        class _IsoWorker(QThread):
            progress = pyqtSignal(str)              # human status line
            download_pct = pyqtSignal(int)          # 0-100 during model download
            phase = pyqtSignal(str)                 # "downloading" / "separating"
            queue_progress = pyqtSignal(int, int, str)  # done, total, current_name
            finished_with = pyqtSignal(list, list)

            def __init__(self, paths, out_dir, first_run):
                super().__init__()
                self.paths, self.out_dir = paths, out_dir
                self._download_phase = first_run

            def run(self):
                import re
                sep = VocalSeparator()
                vocals = []
                errors = []
                total = len(self.paths)

                def parse(line: str):
                    if self._download_phase:
                        m = re.search(r'(\d+)%\|', line)
                        if m:
                            self.download_pct.emit(int(m.group(1)))
                            return
                        if "Separating" in line or "track" in line.lower():
                            self._download_phase = False
                            self.phase.emit("separating")
                    self.progress.emit(line)

                for i, p in enumerate(self.paths, 1):
                    name = os.path.basename(p)
                    self.queue_progress.emit(i - 1, total, name)
                    self.progress.emit(f"Isolating {i}/{total}: {name}")
                    try:
                        stems = sep.separate(p, self.out_dir, on_log=parse)
                        vocals.append(stems["vocals"])
                        self._download_phase = False
                    except Exception as e:
                        errors.append(f"{name}: {e}")
                    self.queue_progress.emit(i, total, name)
                self.finished_with.emit(vocals, errors)

        # Container for the download progress dialog (only used on first run)
        dl_dialog = None
        dl_bar = None
        dl_label = None
        if first_run:
            from PyQt6.QtCore import Qt
            from PyQt6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout
            dl_dialog = QDialog(self)
            dl_dialog.setWindowTitle("Downloading Vocal Model")
            dl_dialog.setModal(True)
            dl_dialog.setFixedSize(420, 130)
            dl_dialog.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
            v = QVBoxLayout(dl_dialog)
            v.setContentsMargins(20, 20, 20, 20)
            v.setSpacing(10)
            dl_label = QLabel("Downloading vocal isolation model (one-time, ~80 MB)...")
            dl_label.setStyleSheet("color: #ddd; font-size: 13px;")
            dl_label.setWordWrap(True)
            v.addWidget(dl_label)
            dl_bar = QProgressBar()
            dl_bar.setRange(0, 100)
            dl_bar.setValue(0)
            v.addWidget(dl_bar)
            hint = QLabel("Future runs use the cached model — instant.")
            hint.setStyleSheet("color: rgba(255,255,255,80); font-size: 11px;")
            v.addWidget(hint)

        self._iso_worker = _IsoWorker(paths, self._iso_dir, first_run)

        # Use the existing training progress bar to track queue progress
        # (training and isolation never run at the same time).
        self._progress_bar.setRange(0, len(paths))
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._lbl_epoch.setVisible(False)

        def _on_progress(msg: str):
            self._lbl_status.setText(msg[:120])
            self._lbl_status.setStyleSheet(
                "color: rgba(255, 255, 255, 75); font-size: 11px; background: transparent;"
            )

        def _on_queue(done: int, total: int, name: str):
            self._progress_bar.setValue(done)
            remaining = total - done
            if remaining > 0:
                self._lbl_status.setText(
                    f"Isolating {done + 1}/{total}: {name}"
                    f"  ·  {remaining} song{'s' if remaining != 1 else ''} left"
                )
            else:
                self._lbl_status.setText(f"Isolated {total} song{'s' if total != 1 else ''}.")

        def _on_dl_pct(pct: int):
            if dl_bar is not None:
                dl_bar.setValue(pct)
                dl_label.setText(f"Downloading vocal isolation model... {pct}%")

        def _on_phase(p: str):
            if dl_dialog is not None and p == "separating":
                dl_dialog.accept()

        self._iso_worker.progress.connect(_on_progress)
        self._iso_worker.queue_progress.connect(_on_queue)
        self._iso_worker.download_pct.connect(_on_dl_pct)
        self._iso_worker.phase.connect(_on_phase)
        self._iso_worker.finished_with.connect(self._on_iso_done)
        self._iso_worker.finished_with.connect(
            lambda *_: dl_dialog.accept() if dl_dialog and dl_dialog.isVisible() else None
        )
        self._iso_worker.start()
        if dl_dialog is not None:
            dl_dialog.exec()

    def _on_iso_done(self, vocals, errors):
        # Drop the queue progress bar now that we're done.
        self._progress_bar.setVisible(False)
        # Rename Demucs's bare "vocals.wav" to "<song>_Isolated_Vocals.wav" so
        # the filename carries the provenance — both for the badge in the file
        # list AND survives the copy into dataset_dir during training.
        renamed = []
        for v in vocals:
            try:
                song_dir = os.path.dirname(v)
                song_stem = os.path.basename(song_dir)  # "Ain't No Sunshine_spotdown.org"
                tagged = os.path.join(song_dir, f"{song_stem}_Isolated_Vocals.wav")
                if v != tagged:
                    os.rename(v, tagged)
                renamed.append(tagged)
            except Exception:
                renamed.append(v)
        # If the user opted to replace the pre-isolation sources, drop them
        # from the clip list. We only remove ones that actually got an
        # isolated counterpart — files that errored out stay so the user
        # can still see them.
        replace = getattr(self, "_iso_replace_originals", False)
        sources = list(getattr(self, "_iso_source_paths", []))
        if replace and renamed and sources:
            ok_count = min(len(renamed), len(sources))  # in queue order
            originals_to_drop = set(sources[:ok_count])
            self._clips = [c for c in self._clips if c not in originals_to_drop]
        # Reset the per-run state
        self._iso_replace_originals = False
        self._iso_source_paths = []

        if renamed:
            self._add_clips(renamed)
            self._lbl_status.setText(f"Isolated vocals from {len(renamed)} song(s).")
            self._lbl_status.setStyleSheet(
                "color: rgba(80, 200, 120, 150); font-size: 11px; background: transparent;"
            )
        if errors:
            QMessageBox.warning(
                self, "Vocal Isolation Failed",
                "Some songs couldn't be processed:\n\n" + "\n".join(errors[:5]),
            )
            if not vocals:
                self._lbl_status.setText("Vocal isolation failed — see error.")
                self._lbl_status.setStyleSheet(
                    "color: rgba(255, 100, 100, 150); font-size: 11px; background: transparent;"
                )

    def _start_training(self, resume=False):
        name = self._selected_name.strip()
        if not name:
            QMessageBox.warning(self, "No Name", "Enter a voice name first.")
            return
        if not self._clips:
            QMessageBox.warning(self, "No Audio", "Add audio clips first.")
            return

        # Stage clips into the dataset dir, normalized to 44.1 kHz / 16-bit / mono.
        # Done in a background thread with a progress dialog so the UI stays
        # responsive even on big datasets that need resampling.
        from services.paths import DATASETS_DIR
        dataset_dir = os.path.join(str(DATASETS_DIR), name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Build the work list: only files that don't already exist in dataset_dir
        pending = []
        for clip in self._clips:
            dest = os.path.join(dataset_dir, os.path.basename(clip))
            if not os.path.exists(dest):
                pending.append((clip, dest))

        if pending:
            cancelled = self._normalize_clips_with_progress(pending)
            if cancelled:
                return

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
            from services.job_store import load_config, create_job
            from services.dataset_manager import DatasetManager

            config = load_config()
            api_key = config.get("runpod_api_key", "") or os.environ.get("SOMERSVC_RUNPOD_KEY", "")
            ssh_key = os.path.expanduser(config.get("ssh_key_path", "~/.ssh/id_rsa"))

            # Train locally? Settings flag — pod is the default otherwise.
            # A local run executes entirely on this machine and never
            # contacts RunPod, so the API-key requirement is gated to the
            # cloud path only.
            train_locally = bool(config.get("train_locally", False))

            if not train_locally and not api_key:
                QMessageBox.warning(self, "No API Key", "Set your RunPod API key in Settings first.")
                self._btn_train.setEnabled(True)
                self._btn_continue_train.setEnabled(True)
                self._btn_train.setText("Start Training")
                return

            dataset_mgr = DatasetManager(str(DATASETS_DIR))
            job = create_job(name)
            job_id = job["job_id"]

            # The local worker mirrors the orchestrator's surface so the
            # rest of the UI plumbing (Stop, log parser, progress, etc.)
            # works without changes.
            if train_locally:
                from workers.local_training_worker import LocalTrainingWorker
                self._log.append_line(
                    "Train Locally is on — running the full pipeline on this "
                    "machine. This will be much slower than a cloud GPU."
                )
                self._worker = LocalTrainingWorker(
                    job_id=job_id,
                    speaker_name=name,
                    dataset_manager=dataset_mgr,
                    models_dir=str(MODELS_DIR),
                    f0_method=f0_method,
                    resume_from=resume_from,
                )
            else:
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
            self._worker.status_changed.connect(self._on_status_text)
            self._worker.progress.connect(self._progress_bar.setValue)

            # New run — clear any GPU-availability status from a prior run so
            # the label starts at the plain "GPU - <chosen>".
            self._gpu_chosen_unavail = False
            self._gpu_active = None
            self._gpu_got = False
            self._refresh_gpu_label()

            # Calculate recommended epochs
            self._recommended_epochs = 2000
            # When resuming, we know the current checkpoint epoch — used to
            # interpret the user's input either as an absolute target or a
            # "+N more epochs" delta.
            cur_ep_for_input = 0
            if resume and resume_from:
                try:
                    import re as _re
                    _m = _re.search(r'G_(\d+)\.pth', os.path.basename(resume_from))
                    if _m:
                        cur_ep_for_input = int(_m.group(1))
                except Exception:
                    pass
            epochs_text = self._txt_epochs.text().strip()
            if epochs_text and epochs_text.lower() != "auto":
                try:
                    raw = epochs_text
                    # "+1126" → add 1126 to current.
                    if raw.startswith("+"):
                        delta = int(raw[1:])
                        self._recommended_epochs = cur_ep_for_input + delta
                    else:
                        n = int(raw)
                        # Forgiving: a value <= current_epoch on a resume run
                        # is almost certainly a delta the user typed by mistake
                        # (would otherwise auto-stop instantly). Treat as +N.
                        if cur_ep_for_input > 0 and n <= cur_ep_for_input:
                            self._recommended_epochs = cur_ep_for_input + n
                            self._log.append_line(
                                f"Interpreting '{n}' as +{n} more epochs "
                                f"(current: {cur_ep_for_input}, target: "
                                f"{self._recommended_epochs})."
                            )
                        else:
                            self._recommended_epochs = n
                except ValueError:
                    pass
            else:
                # Continue Training + Auto: target the next quality tier
                # rather than the duration-default (which may be far below
                # the model's current epoch count and would auto-stop on
                # the first log line).
                auto_target = None
                if resume and resume_from:
                    try:
                        import re as _re
                        m = _re.search(r'G_(\d+)\.pth', os.path.basename(resume_from))
                        cur_ep = int(m.group(1)) if m else 0
                    except Exception:
                        cur_ep = 0
                    clips_count = max(len(self._clips), 1)
                    last_meta = self._load_model_metadata(name) or {}
                    cur_batch = int(last_meta.get("batch_size", 16) or 16)
                    maturity = (cur_ep * cur_batch) / clips_count
                    if maturity >= 2000:
                        target_maturity = max(maturity * 1.25, maturity + 200)
                    elif maturity >= 800:
                        target_maturity = 2000
                    elif maturity >= 300:
                        target_maturity = 800
                    else:
                        target_maturity = 300
                    extra_epochs = int(
                        (target_maturity - maturity) * clips_count / max(cur_batch, 1)
                    )
                    auto_target = max(cur_ep + max(extra_epochs, 100), cur_ep + 100)
                if auto_target is not None:
                    self._recommended_epochs = auto_target
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
                # The box stays empty (auto) — its dim placeholder
                # already previews this count.
            # Print the target + an A40 wall-clock estimate so the user
            # knows roughly how long the run will take.
            current_ep_for_est = 0
            if resume and resume_from:
                try:
                    import re as _re
                    _m = _re.search(r'G_(\d+)\.pth', os.path.basename(resume_from))
                    if _m:
                        current_ep_for_est = int(_m.group(1))
                except Exception:
                    pass
            eta, gpu_label = self._estimate_training_time(
                self._recommended_epochs,
                current_epochs=current_ep_for_est,
                clips=len(self._clips),
                is_resume=resume,
            )
            try:
                import soundfile as _sf
                total_dur_log = sum(_sf.info(p).duration for p in self._clips)
                m, s = divmod(int(total_dur_log), 60)
                self._log.append_line(
                    f"Target: {self._recommended_epochs} epochs "
                    f"(dataset duration: {m}:{s:02d}, est. {eta} on {gpu_label}) — "
                    f"training will auto-stop here."
                )
            except Exception:
                self._log.append_line(
                    f"Target: {self._recommended_epochs} epochs "
                    f"(est. {eta} on {gpu_label}) — training will auto-stop here."
                )
            self._current_epoch = 0
            self._auto_stop_fired = False
            self._last_log_ckpt_epoch = None
            self._last_log_was_wait = False
            # Persist the auto-stop target so app close + reopen restores it.
            try:
                from services.job_store import update_job
                update_job(job_id, recommended_epochs=int(self._recommended_epochs))
            except Exception:
                pass
            # Push the target into the worker BEFORE start so the orchestrator
            # bakes it into the pod's config and runs the auto-stop watcher.
            # Without this, training would run forever if the user closes the
            # app — the UI auto-stop is gone.
            try:
                self._worker.target_epochs = int(self._recommended_epochs)
            except Exception:
                pass
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
            self.training_started.emit()
        except Exception as e:
            self._on_train_error(str(e))

    def _stop_training(self):
        # Either a fresh TrainingWorker or a reattached ResumeWorker may be
        # the live one. Find whichever has request_stop() and is running.
        active = None
        for attr in ("_worker", "_resume_worker"):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, "isRunning") and w.isRunning() \
                    and hasattr(w, "request_stop"):
                active = w
                break
        if active is None:
            return
        if getattr(self, "_auto_stop_fired", False):
            return  # already in progress
        reply = QMessageBox.question(
            self, "Stop Training",
            "Stop training now?\n\n"
            "If a checkpoint has been saved, it will be downloaded before "
            "the cloud GPU shuts down. If training hasn't started yet, the "
            "GPU will just be released — nothing to save.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._auto_stop_fired = True
        # IMMEDIATE feedback so the user knows the click registered.
        self._btn_stop_train.setText("Stopping...")
        self._btn_stop_train.setEnabled(False)
        self._lbl_status.setText("Stopping — saving checkpoint, then downloading...")
        # Pulse the status text so it reads as active rather than stuck.
        self._lbl_status.setStyleSheet(
            "color: #FBBF24; font-size: 12px; font-weight: 600; "
            "background: transparent;"
        )
        self._log.append_line(">>> Stop requested. Saving checkpoint, then downloading model...")
        # Force an immediate paint so the feedback shows before request_stop's
        # synchronous SSH connect blocks the UI thread.
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        active.request_stop()

    def _format_log_line_for_display(self, line: str):
        """Suppress verbose svc/Lightning/pip noise; synthesize clean checkpoint
        lines. Returns the display string, or None to skip.

        The raw line still flows through the epoch parser below — this only
        controls what shows up in the user-facing log.
        """
        import re
        if not line.strip():
            return None

        # Clean checkpoint event from Lightning's "Saving ... at epoch N to .../G_N.pth"
        m = re.search(r'epoch\s+(\d+).*?\bG_(\d+)\.pth\b', line, re.IGNORECASE)
        if m:
            ep = m.group(1)
            if getattr(self, "_last_log_ckpt_epoch", None) == ep:
                return None
            self._last_log_ckpt_epoch = ep
            return f"Training in progress... latest checkpoint: G_{ep}.pth"

        # ResumeWorker emits these directly — dedupe consecutive identical lines
        m = re.match(r'^Training in progress\.\.\. latest checkpoint: G_(\d+)\.pth', line)
        if m:
            ep = m.group(1)
            if getattr(self, "_last_log_ckpt_epoch", None) == ep:
                return None
            self._last_log_ckpt_epoch = ep
            return line

        # Collapse repeated "Waiting for training to produce checkpoints..." into one
        if line.strip() == "Waiting for training to produce checkpoints...":
            if getattr(self, "_last_log_was_wait", False):
                return None
            self._last_log_was_wait = True
            return line
        self._last_log_was_wait = False

        # Drop svc/Lightning/torch/rich timestamped log lines and wrapped continuations
        if re.match(r'^\s*\[\d{2}:\d{2}:\d{2}\]\s+(INFO|WARNING|DEBUG|ERROR)\b', line):
            return None
        if re.match(r'^\s+(INFO|WARNING|DEBUG|ERROR)\s+\[\d{2}:\d{2}:\d{2}\]', line):
            return None
        # Heavily-indented wrap continuations from rich's column wrapping
        if re.match(r'^\s{15,}\S', line):
            return None
        # pip install chatter
        if re.match(r'^\s*(Collecting|Downloading|Requirement already|Successfully (installed|built|uninstalled)|Building wheel|Getting requirements|Preparing metadata|Using cached|Installing collected|Attempting uninstall|Found existing installation)\b', line):
            return None

        return line

    def _on_train_log(self, line):
        display = self._format_log_line_for_display(line)
        if display is not None:
            self._log.append_line(display)
        # Cloud GPU provisioning status -> the label above Start Training.
        self._parse_gpu_provisioning(line)
        import re
        epoch = None
        # ResumeWorker polling line: "Training in progress... latest checkpoint: G_25.pth"
        m = re.match(
            r'^Training in progress\.\.\. latest checkpoint: G_(\d+)\.pth', line
        )
        if m:
            epoch = int(m.group(1))
        # Live progress line: "Epoch 37/9999"
        if epoch is None:
            m = re.search(r'Epoch (\d+)/(\d+)', line)
            if m:
                epoch = int(m.group(1))
        # Checkpoint save: "Saving ... at epoch 100 to /workspace/.../G_100.pth"
        if epoch is None:
            m = re.search(r'(?:saving|state at).*?epoch\s+(\d+)', line, re.IGNORECASE)
            if m:
                epoch = int(m.group(1))
        # Checkpoint filename fallback: G_100.pth or D_100.pth
        # Word-boundary-anchored so we don't match URLs like "clean_G_320000.pth"
        # in the pretrained-model warning. Also require the line to look like a
        # save event so we never chew on the bare URL.
        if epoch is None and re.search(r'\b(saving|writing|to)\b', line, re.IGNORECASE):
            m = re.search(r'\b[GD]_(\d+)\.pth\b', line)
            if m:
                epoch = int(m.group(1))

        if epoch is not None and epoch > 0:
            # so-vits-svc-fork restores current_epoch from the checkpoint on
            # resume (set_current_epoch in its train.py), so every epoch it
            # reports — the live "Epoch X/Y" line and the G_<epoch>.pth
            # checkpoints alike — is already the running TOTAL. Use it
            # directly; adding a resume offset double-counted (e.g. 222/125).
            self._current_epoch = epoch
            rec = self._recommended_epochs
            if rec > 0:
                # Training can overshoot the (estimated) target before the
                # stop actually lands — clamp the denominator so the counter
                # never reads a nonsensical past-target ratio like 222/125.
                shown_total = max(rec, self._current_epoch)
                pct = min(int((self._current_epoch / rec) * 100), 100)
                self._progress_bar.setValue(pct)
                self._lbl_epoch.setText(f"{self._current_epoch}/{shown_total}")
                self._lbl_epoch.setVisible(True)
                self._lbl_status.setText(
                    f"Training... Epoch {self._current_epoch}/{shown_total}")
            # Auto-stop at target. Latch so we don't re-fire when the matching
            # D_<epoch>.pth save (or any later log line) is parsed too.
            if (rec > 0 and self._current_epoch >= rec
                    and not getattr(self, "_auto_stop_fired", False)):
                if self._worker and self._worker.isRunning():
                    self._auto_stop_fired = True
                    self._log.append_line(f"Reached target of {rec} epochs — stopping & downloading model...")
                    self._worker.request_stop()

    def _on_status_text(self, s):
        """Update the status line. Once the run leaves the training phase
        (uploading/downloading the model, or complete), hide the epoch
        counter so it can't sit there showing a stale, past-target ratio."""
        self._lbl_status.setText(s)
        low = (s or "").lower()
        if any(k in low for k in ("upload", "download", "complete")):
            self._lbl_epoch.setVisible(False)

    def _on_train_done(self, job_id):
        self._training = False
        self._btn_stop_train.setVisible(False)
        self._btn_train.setVisible(True)
        self._btn_train.setEnabled(True)
        self._btn_train.setText("Start Training")
        self._check_existing_model(self._selected_name)
        # Snapshot which clips this run actually trained on — the dataset dir
        # contents at this moment are exactly what got sent to the pod.
        self._save_trained_snapshot(self._selected_name)
        # Clips that were just trained on now live inside the dataset dir —
        # repoint paths and re-color the list so pending/orange flips to trained/green.
        try:
            from services.paths import DATASETS_DIR
            ds_dir = os.path.join(str(DATASETS_DIR), self._selected_name)
            if os.path.isdir(ds_dir):
                self._clips = [
                    os.path.join(ds_dir, f) for f in sorted(os.listdir(ds_dir))
                    if f.endswith(('.wav', '.flac', '.mp3', '.ogg'))
                ]
                self._refresh_file_list()
        except Exception:
            pass
        # The model's metadata.json was just rewritten with the new epoch
        # count — refresh the left-side panels so they show it now instead
        # of staying stale until the next artist reselect.
        try:
            self._update_model_info(self._selected_name)
            self._update_metadata_panel(self._selected_name)
        except Exception:
            pass
        self._progress_bar.setValue(100)
        self._lbl_epoch.setVisible(False)
        self._lbl_status.setText("Training complete! Model is ready.")
        self._lbl_status.setStyleSheet("color: rgba(80, 200, 120, 150); font-size: 11px; background: transparent;")
        self.training_stopped.emit()

    def _on_train_error(self, error):
        self._training = False
        self._btn_stop_train.setVisible(False)
        # Reset Stop button text in case it was changed to "Stopping..."
        self._btn_stop_train.setText("Stop Training")
        self._btn_stop_train.setEnabled(True)
        self._btn_train.setVisible(True)
        self._btn_train.setEnabled(True)
        self._btn_train.setText("Start Training")
        self._check_existing_model(self._selected_name)
        self._progress_bar.setVisible(False)
        self._lbl_epoch.setVisible(False)
        # User-cancelled errors get a friendlier message than "Error: stopped".
        was_cancelled = (
            getattr(self, "_auto_stop_fired", False)
            or (isinstance(error, str)
                and error.lower() in ("stopped", "cancelled"))
        )
        if was_cancelled:
            self._lbl_status.setText("Training cancelled. Cloud GPU stopped.")
            self._lbl_status.setStyleSheet(
                "color: #2DD4BF; font-size: 11px; background: transparent;"
            )
        else:
            self._lbl_status.setText(f"Error: {error}")
            self._lbl_status.setStyleSheet(
                "color: rgba(255, 100, 100, 150); font-size: 11px; "
                "background: transparent;"
            )
        self._auto_stop_fired = False
        self.training_stopped.emit()


class _OptimizeButton(QPushButton):
    """The Optimize / Optimal button below the waveform. In the
    optimized ('Optimal') state the word carries a slow left-to-right
    shimmer; otherwise it renders as a normal stylesheet button."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._shimmer = False
        self._phase = 0.0  # 0..1 sweep position
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._advance)

    def set_shimmer(self, on: bool):
        if on == self._shimmer:
            return
        self._shimmer = on
        if on and self.isVisible():
            self._timer.start()
        elif not on:
            self._timer.stop()
        self.update()

    def _advance(self):
        self._phase = (self._phase + 0.016) % 1.0
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        if self._shimmer and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._timer.stop()

    def paintEvent(self, event):
        if not self._shimmer:
            super().paintEvent(event)
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        # Border — matches the 'Optimal' stylesheet, green-tinted on hover.
        hover = self.underMouse()
        border = QColor(80, 200, 120, 110) if hover else QColor(255, 255, 255, 12)
        p.setPen(QPen(border, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(
            QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 10, 10
        )
        # Shimmering text: a bright band sweeps a dim base, entering from
        # off-left and exiting off-right.
        base = QColor(125, 125, 125)
        shine = QColor(240, 240, 240)
        band = 0.22
        center = self._phase * (1.0 + 2 * band) - band
        grad = QLinearGradient(0, 0, self.width(), 0)
        for stop, col in (
            (0.0, base),
            (center - band, base),
            (center, shine),
            (center + band, base),
            (1.0, base),
        ):
            grad.setColorAt(min(1.0, max(0.0, stop)), col)
        p.setPen(QPen(QBrush(grad), 1))
        p.setFont(self.font())
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
        p.end()


class _ConvertQueueRow(QWidget):
    """One file row in the batch-convert queue list.

    Carries a status the batch runner drives (queued → converting →
    done/failed). The status maps to a coloured dot + filename so
    completion reads at a glance; the converting row also gets a
    faint highlight and an animated dot.
    """
    remove_requested = pyqtSignal(str)  # emits this row's path
    play_requested = pyqtSignal(str)    # emits this row's path (▶ click)

    # status -> (dot glyph, text colour, dot colour)
    _STATUS = {
        "queued":     ("○", "#888888", "rgba(255,255,255,40)"),
        "converting": ("◐", "#cfd8ff", "#5e8cff"),
        "done":       ("●", "#4ade80", "#4ade80"),
        "failed":     ("●", "#ef4444", "#ef4444"),
    }
    # Rotating frames for the converting-row dot animation.
    _SPIN = ["◐", "◓", "◑", "◒"]

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.status = "queued"
        self.output_path = ""  # converted file, set once this row is done
        self.sections = None  # how many sections Range-Match split it into
        self._spin_idx = 0
        # Required for the converting-row highlight to actually paint.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 5, 8, 5)
        row.setSpacing(8)

        self._dot = QLabel()
        self._dot.setFixedWidth(14)
        row.addWidget(self._dot)

        self._name = QLabel(os.path.basename(path))
        self._name.setToolTip(path)
        row.addWidget(self._name, 1)

        # Status word — shows "Queued" while a file waits its turn.
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            "color: #888; font-size: 10px; background: transparent;"
        )
        self._status_lbl.setVisible(False)
        row.addWidget(self._status_lbl)

        # How many sections Range-Match split the clip into — filled in
        # mid-conversion, teal so it reads as a result, not a control.
        self._sections_lbl = QLabel("")
        self._sections_lbl.setStyleSheet(
            "color: #5ec8b4; font-size: 10px; font-weight: 600;"
            " background: transparent;"
        )
        self._sections_lbl.setVisible(False)
        row.addWidget(self._sections_lbl)

        # ▶ to preview the converted result — shown only on done rows.
        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedSize(18, 18)
        self._btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_play.setStyleSheet(
            "QPushButton { background: rgba(74,222,128,20); border: none;"
            " border-radius: 9px; color: #4ade80; font-size: 9px; padding: 0; }"
            "QPushButton:hover { background: rgba(74,222,128,36); }"
        )
        self._btn_play.clicked.connect(
            lambda: self.play_requested.emit(self.path)
        )
        self._btn_play.setVisible(False)
        row.addWidget(self._btn_play)

        self._btn_remove = QPushButton("×")
        self._btn_remove.setFixedSize(18, 18)
        self._btn_remove.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_remove.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,10); border: none;"
            " border-radius: 9px; color: #888; font-size: 12px; padding: 0; }"
            "QPushButton:hover { background: rgba(255,255,255,20); color: #ccc; }"
        )
        self._btn_remove.clicked.connect(
            lambda: self.remove_requested.emit(self.path)
        )
        row.addWidget(self._btn_remove)

        self.set_status("queued")

    def set_status(self, status: str):
        self.status = status
        glyph, text_color, dot_color = self._STATUS.get(
            status, self._STATUS["queued"]
        )
        if status == "converting":
            self._spin_idx = 0
            glyph = self._SPIN[0]
        self._dot.setText(glyph)
        self._dot.setStyleSheet(
            f"color: {dot_color}; font-size: 11px; background: transparent;"
        )
        weight = "600" if status in ("converting", "done") else "400"
        self._name.setStyleSheet(
            f"color: {text_color}; font-size: 12px; font-weight: {weight};"
            " background: transparent;"
        )
        # Faint highlight marks the file currently being converted.
        if status == "converting":
            self.setStyleSheet(
                "_ConvertQueueRow { background-color: rgba(94,140,255,20);"
                " border-radius: 6px; }"
            )
        else:
            self.setStyleSheet("")
        # "Queued" marks a file still waiting to be converted.
        self._status_lbl.setText("Queued" if status == "queued" else "")
        self._status_lbl.setVisible(status == "queued")
        self._refresh_buttons()

    def _refresh_buttons(self):
        # Remove only while queued; play only once a result exists.
        self._btn_remove.setVisible(self.status == "queued")
        self._btn_play.setVisible(
            self.status == "done" and bool(self.output_path)
        )

    def set_output(self, output_path: str):
        """Attach the converted file so the row can offer playback."""
        self.output_path = output_path or ""
        self._refresh_buttons()

    def set_sections(self, count: int):
        """Show how many sections Range-Match split the clip into."""
        self.sections = int(count)
        plural = "" if self.sections == 1 else "s"
        self._sections_lbl.setText(
            f"Transposed in {self.sections} section{plural}"
        )
        self._sections_lbl.setVisible(True)

    def set_playing(self, playing: bool):
        self._btn_play.setText("⏸" if playing else "▶")

    def tick_spinner(self):
        """Advance the converting-row dot by one animation frame."""
        if self.status != "converting":
            return
        self._spin_idx = (self._spin_idx + 1) % len(self._SPIN)
        self._dot.setText(self._SPIN[self._spin_idx])


class _ConvertQueueList(QWidget):
    """Scrollable list of files queued for batch conversion, with a
    live N-of-M header. Shown in place of the waveform editor when
    2+ files are loaded."""
    remove_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: dict[str, _ConvertQueueRow] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._header = QLabel("")
        self._header.setStyleSheet(
            "color: #999; font-size: 11px; font-weight: 600;"
            " background: transparent; padding: 0 4px;"
        )
        outer.addWidget(self._header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMaximumHeight(220)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setStyleSheet(
            "QScrollArea { background: rgba(255,255,255,3);"
            " border: 1px solid rgba(255,255,255,10); border-radius: 12px; }"
        )

        self._inner = QWidget()
        self._list_layout = QVBoxLayout(self._inner)
        self._list_layout.setContentsMargins(4, 4, 4, 4)
        self._list_layout.setSpacing(2)
        self._list_layout.addStretch()  # keep rows top-aligned
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)

        # Drives the converting-row dot animation while a file runs.
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(130)
        self._spin_timer.timeout.connect(self._tick_spinner)

        # Shared player for per-row result playback — one file at a time.
        from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
        from services.audio_device_tracker import register_audio_output
        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        register_audio_output(self._audio_out)
        self._player.setAudioOutput(self._audio_out)
        self._player.playbackStateChanged.connect(self._on_play_state)
        self._playing_path = None

    def set_header(self, text: str):
        """Set the line above the list (accepts rich text for colour)."""
        self._header.setText(text)

    def _tick_spinner(self):
        active = [r for r in self._rows.values() if r.status == "converting"]
        if not active:
            self._spin_timer.stop()
            return
        for r in active:
            r.tick_spinner()

    def _on_row_play(self, path: str):
        """Play — or pause/resume — the converted result for a row."""
        row = self._rows.get(path)
        if row is None or not row.output_path:
            return
        from PyQt6.QtCore import QUrl
        from PyQt6.QtMultimedia import QMediaPlayer
        if self._playing_path == path:
            # Same file — toggle pause/resume.
            if (self._player.playbackState()
                    == QMediaPlayer.PlaybackState.PlayingState):
                self._player.pause()
            else:
                self._player.play()
            return
        # Different file — stop the old one, start this one.
        self._player.stop()
        old = self._rows.get(self._playing_path)
        if old is not None:
            old.set_playing(False)
        self._playing_path = path
        self._player.setSource(QUrl.fromLocalFile(row.output_path))
        self._player.play()

    def _on_play_state(self, state):
        from PyQt6.QtMultimedia import QMediaPlayer
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        row = self._rows.get(self._playing_path)
        if row is not None:
            row.set_playing(playing)

    def set_files(self, paths: list):
        """Rebuild the list to match `paths`, carrying over the status,
        converted-output path and section count of any row whose path
        is unchanged."""
        prev = {p: (r.status, r.output_path, r.sections)
                for p, r in self._rows.items()}
        self._rows = {}
        # Drop existing row widgets, keeping the trailing stretch item.
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        for p in paths:
            row = _ConvertQueueRow(p)
            if p in prev:
                status, out, secs = prev[p]
                row.set_status(status)
                if out:
                    row.set_output(out)
                if secs is not None:
                    row.set_sections(secs)
            row.remove_requested.connect(self.remove_requested)
            row.play_requested.connect(self._on_row_play)
            self._list_layout.insertWidget(
                self._list_layout.count() - 1, row
            )
            self._rows[p] = row

    def row(self, path: str):
        return self._rows.get(path)

    def set_status(self, path: str, status: str, output_path: str = None):
        row = self._rows.get(path)
        if row is not None:
            row.set_status(status)
            if output_path:
                row.set_output(output_path)
        if status == "converting" and not self._spin_timer.isActive():
            self._spin_timer.start()

    def set_sections(self, path: str, count: int):
        row = self._rows.get(path)
        if row is not None:
            row.set_sections(count)

    def clear(self):
        self._spin_timer.stop()
        try:
            self._player.stop()
        except Exception:
            pass
        self._playing_path = None
        self.set_files([])
        self._header.setText("")


class SimplePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._original_source = ""
        self._source_paths = []  # batch convert queue (raw file paths)
        self._batch_running = False
        self._batch_files = []
        self._batch_index = 0
        self._batch_done = 0
        self._batch_failed = []
        self._batch_results = {}  # source path -> converted output path
        self._batch_output_dir = ""  # per-batch <artist>_Batch_<N> folder
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
        self._create_panel.training_started.connect(lambda: self._show_spinner("modeling"))
        self._create_panel.training_stopped.connect(self._hide_spinner)
        self._create_panel.setVisible(False)
        # Restore last session after a delay (let layout settle)
        QTimer.singleShot(200, self.restore_session)
        # Check for active training jobs on launch
        QTimer.singleShot(600, self._check_active_training)
        # Clean up orphaned pods on launch
        QTimer.singleShot(800, self._cleanup_orphaned_pods)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 25, 24, 20)

        # ===== SEARCH BAR =====
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search artists on HuggingFace...")
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

        self._btn_create_model = QLabel("My Models")
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
        self._dropdown.setItemDelegate(_HFArtistDelegate(self._dropdown))
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

        # Batch-convert queue — replaces the waveform editor when 2+
        # files are loaded. Hidden in the single-file flow.
        self._convert_queue = _ConvertQueueList()
        self._convert_queue.remove_requested.connect(self._remove_source)
        self._convert_queue.setVisible(False)
        layout.addWidget(self._convert_queue)


        # Optimize button (appears below waveform when sections exist)
        self._btn_optimize = _OptimizeButton("Optimal")
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
        # Wrap the optimize row in a QWidget so we can hide the whole
        # container — Qt collapses layout spacing around hidden widgets
        # but not around bare sub-layouts, which used to leave ~32px of
        # dead space between the waveform and the Convert button on any
        # clip short enough to be a single section.
        self._opt_container = QWidget()
        opt_row = QHBoxLayout(self._opt_container)
        opt_row.setContentsMargins(0, 0, 0, 0)
        opt_row.addStretch()
        opt_row.addWidget(self._btn_optimize)
        opt_row.addStretch()
        self._opt_container.setVisible(False)
        layout.addWidget(self._opt_container)

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

        # Output (converted) waveform sits in the MAIN layout right under
        # the Convert button so the result appears close to where the user
        # was just looking, instead of pinned to the window bottom inside
        # the floating overlay. Hidden until a conversion completes.
        self._lbl_output_name = QLabel("")
        self._lbl_output_name.setStyleSheet("color: rgba(255,255,255,40); font-size: 9px; background: transparent;")
        self._lbl_output_name.setFixedHeight(14)
        self._lbl_output_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_output_name.setVisible(False)
        layout.addWidget(self._lbl_output_name)

        self._waveform_output = _WaveformWidget(readonly=True)
        self._waveform_output.setVisible(False)
        self._waveform_output.interacted.connect(lambda: setattr(self, '_active_waveform', self._waveform_output))
        layout.addWidget(self._waveform_output)

        layout.addStretch()

        # ===== BOTTOM AREA (floating overlay, doesn't affect layout) =====
        # Keeps the log viewer + folder buttons pinned to the bottom.
        # The output waveform was moved out of here (see above) so it
        # appears next to the Convert button rather than at the bottom.
        self._bottom_panel = QWidget(self)
        self._bottom_panel.setStyleSheet("background: transparent;")
        bottom_layout = QVBoxLayout(self._bottom_panel)
        bottom_layout.setContentsMargins(24, 0, 24, 16)
        bottom_layout.setSpacing(4)

        self._log = LogViewer()
        self._log.setMaximumHeight(110)
        self._log.setVisible(False)
        bottom_layout.addWidget(self._log)

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
        # Lift the panel 25px off the page bottom so the log doesn't
        # overlap the settings button.
        self._bottom_panel.move(
            0, self.height() - self._bottom_panel.height() - 25
        )
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
        models.append({"name": "Best Match", "dir": "", "pixmap": best_pixmap, "vocal_key": "", "is_best_match": True})

        if os.path.exists(MODELS_DIR):
            for name in sorted(os.listdir(MODELS_DIR), key=str.casefold):
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

                # Load image — prefer model-dir image, fall back to the
                # Spotify-fetched artist thumbnail cache so newly trained
                # models pick up the photo the Create panel already cached.
                pixmap = None
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    p = os.path.join(model_dir, f"image{ext}")
                    if os.path.exists(p):
                        pixmap = QPixmap(p)
                        break
                if pixmap is None or pixmap.isNull():
                    thumb = os.path.join(CACHE_DIR, "artist_thumbs", f"{name}.jpg")
                    if os.path.exists(thumb):
                        pixmap = QPixmap(thumb)

                # Load vocal key from metadata
                vocal_key = ""
                ignore_vocal_key = False
                meta_path = os.path.join(model_dir, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        import json as _json
                        with open(meta_path) as _f:
                            _meta = _json.load(_f)
                        vocal_key = _meta.get("vocal_key", "")
                        ignore_vocal_key = bool(_meta.get("ignore_vocal_key", False))
                    except Exception:
                        pass

                models.append({
                    "name": name, "dir": model_dir, "pixmap": pixmap,
                    "vocal_key": vocal_key,
                    "ignore_vocal_key": ignore_vocal_key,
                })

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
            # ignore_vocal_key skip removed — feature hidden.
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
                # ignore_vocal_key check removed — the user-facing toggle
                # was hidden because it was too easy to flip accidentally.
                # Any stale `ignore_vocal_key: true` in metadata is now
                # ignored itself, so the saved key is always honored.
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
                vis = len(self._waveform._sections) > 1
                self._btn_optimize.setVisible(vis)
                self._opt_container.setVisible(vis)
            else:
                self._analyze_waveform()
        elif not checked:
            self._waveform.hide_and_stop()
            self._btn_optimize.setVisible(False)
            self._opt_container.setVisible(False)
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

        # Resolve the svc invocation in both dev (venv with bin/svc) and
        # bundled-app modes. In the .app there's no separate svc binary, so
        # we re-exec ourselves with --svc-mode and main.py hands off to
        # so-vits-svc-fork's CLI. (Same pattern as services.inference_runner
        # and ui.pages.realtime_page.)
        dev_bin = os.path.join(os.path.dirname(sys.executable), "svc")
        if os.path.exists(dev_bin):
            cmd_prefix = [dev_bin]
        else:
            cmd_prefix = [sys.executable, "--svc-mode"]

        cmd = cmd_prefix + [
            "vc",
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
        self._source_paths = []
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
        self._opt_container.setVisible(False)
        self._hide_spinner()
        # Drop the batch queue and restore the single-file view.
        self._convert_queue.clear()
        self._convert_queue.setVisible(False)
        self._waveform.setVisible(True)
        self._lbl_pitch.setVisible(True)
        self._waveform.clear()
        self._lbl_pitch.setText("")
        self._waveform_output.hide_and_stop()
        self._lbl_output_name.setVisible(False)
        self._convert_ring.set_update_mode(False)
        self._log.setVisible(False)
        QTimer.singleShot(50, self._position_bottom_panel)

    def _browse_source(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio", "",
            "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if paths:
            self._add_sources(paths)

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
        # Sync the waveform widget's model_center_hz any time it changes,
        # regardless of whether the global source median was detected.
        # Per-section pitches are computed independently and need this
        # sync to colorize. Without it, picking a model AFTER dropping
        # audio left the waveform stuck on the stale value (usually 0,
        # so all sections rendered gray).
        if self._waveform._samples:
            if self._waveform._model_center_hz != self._model_center_hz:
                self._waveform._model_center_hz = self._model_center_hz
                self._waveform._invalidate_wave_cache()
        elif self._source_path and self._model_center_hz > 0:
            self._analyze_waveform()

        if self._source_median_hz > 0 and self._model_center_hz > 0:
            import math
            semitones = round(12 * math.log2(self._model_center_hz / self._source_median_hz))
            model_note = _hz_to_note(self._model_center_hz)
            current = self._lbl_pitch.text().split("→")[0].strip()
            self._lbl_pitch.setText(f"{current}  →  Transpose: {semitones:+d} (to {model_note})")
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
            vis = len(sections) > 1
            self._btn_optimize.setVisible(vis)
            self._opt_container.setVisible(vis)
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
        self._btn_optimize.set_shimmer(False)
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
        self._btn_optimize.set_shimmer(True)
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

        # Already running? A second click cancels — batch or single.
        if self._batch_running:
            self._cancel_batch()
            return
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
            self._worker = None
            self._convert_ring.set_converting(False)
            self._waveform.set_active_section(-1)
            self._waveform.set_progress(0.0)
            self._log.append_line("Cancelled.")
            return

        # Batch mode — convert the whole queue, one file at a time.
        if len(self._source_paths) >= 2:
            self._start_batch()
            return

        if not self._source_path or not os.path.exists(self._source_path):
            QMessageBox.warning(self, "No Audio", "Select an audio file first.")
            return

        resolved = self._resolve_selected_model()
        if resolved is None:
            return
        model_dir, mt, model_path, config_path = resolved

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
        # Pulse the ring to 100% so the user sees a clear "done" beat
        # before it resets. Chunk-counting caps progress at 95%, and
        # set_converting(False) clears the ring instantly, so without
        # this the bar appears to hang at 95% and then vanish.
        self._convert_ring.set_progress(1.0)
        QTimer.singleShot(500, lambda: self._convert_ring.set_converting(False))
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

    # ===== BATCH CONVERT =====

    def _resolve_selected_model(self):
        """Resolve the carousel's current model to
        (model_dir, model_type, model_path, config_path). Shows a
        warning and returns None if nothing usable is selected."""
        if self._cmb_model.count() == 0 or not self._cmb_model.currentData():
            QMessageBox.warning(self, "No Model", "Select a model first.")
            return None
        model_dir = self._cmb_model.currentData()
        mt = detect_model_type(model_dir)
        if mt == "svc":
            g_files = sorted(
                [f for f in os.listdir(model_dir) if f.startswith("G_") and f.endswith(".pth")],
                key=lambda f: int(f.replace("G_", "").replace(".pth", "")) if f.replace("G_", "").replace(".pth", "").isdigit() else 0,
            )
            if not g_files:
                QMessageBox.warning(self, "No Model", "No checkpoint found.")
                return None
            model_path = os.path.join(model_dir, g_files[-1])
            config_path = os.path.join(model_dir, "config.json")
        else:
            rvc_files = _get_rvc_pth_files(os.listdir(model_dir))
            if not rvc_files:
                QMessageBox.warning(self, "No Model", "No RVC model found.")
                return None
            model_path = os.path.join(model_dir, rvc_files[0])
            config_path = ""
        return model_dir, mt, model_path, config_path

    def _is_converted(self, path: str) -> bool:
        """True if this file's queue row is already done (green)."""
        row = self._convert_queue.row(path)
        return row is not None and row.status == "done"

    def _start_batch(self):
        """Kick off a sequential conversion of the queued files. One
        model for the whole batch; Range-Match (if on) computes each
        file's pitch shift individually inside its InferenceWorker.

        Files already converted (green) are skipped — they never
        re-convert; only not-yet-done files run."""
        resolved = self._resolve_selected_model()
        if resolved is None:
            return
        (self._batch_model_dir, self._batch_mt,
         self._batch_model_path, self._batch_config_path) = resolved

        # Convert only files not already done — a green/converted row is
        # never re-run. Queued and failed files are (re)included.
        self._batch_files = [
            p for p in self._source_paths if not self._is_converted(p)
        ]
        if not self._batch_files:
            QMessageBox.information(
                self, "Already Converted",
                "Every queued file is already converted. Drop in new "
                "files to convert more.",
            )
            return

        # Each batch lands in its own OUTPUT_DIR/<artist>_Batch_<N>
        # folder so runs stay organised and never overwrite each other.
        artist = (os.path.basename(self._batch_model_dir)
                  if self._batch_model_dir else "converted")
        self._batch_output_dir = self._make_batch_folder(artist)

        self._batch_index = 0
        self._batch_done = 0
        self._batch_failed = []
        self._batch_results = {}
        self._batch_running = True

        # Reset only the files about to convert; converted (green) rows
        # are left exactly as they are.
        for p in self._batch_files:
            self._convert_queue.set_status(p, "queued")

        self._convert_ring.set_update_mode(False)
        self._convert_ring.set_converting(True)
        self._convert_ring.set_progress(0.02)
        self._show_spinner("converting")
        self._log.clear_log()
        self._log.append_line(
            f"Batch: converting {len(self._batch_files)} files "
            f"→ {os.path.basename(self._batch_output_dir)}/"
        )
        QTimer.singleShot(50, self._position_bottom_panel)
        self._batch_next()

    def _make_batch_folder(self, artist: str) -> str:
        """Create and return the next free OUTPUT_DIR/<artist>_Batch_<N>
        directory. N ascends so a new batch never lands on an older one."""
        base = artist or "converted"
        n = 1
        while True:
            folder = os.path.join(OUTPUT_DIR, f"{base}_Batch_{n}")
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
                return folder
            n += 1

    def _batch_next(self):
        """Advance to the next queued file, or finish the batch."""
        if not self._batch_running:
            return
        if self._batch_index >= len(self._batch_files):
            self._batch_finish()
            return
        path = self._batch_files[self._batch_index]
        self._convert_queue.set_status(path, "converting")
        self._convert_queue.set_header(
            f"Converting {self._batch_index + 1} of "
            f"{len(self._batch_files)} — {os.path.basename(path)}"
        )
        self._log.append_line(
            f"[{self._batch_index + 1}/{len(self._batch_files)}] "
            f"{os.path.basename(path)}"
        )
        # Normalize this file, then convert it.
        self._batch_norm = _NormalizeWorker(path)
        self._batch_norm.finished.connect(self._batch_convert_file)
        self._batch_norm.start()

    def _batch_convert_file(self, norm_path):
        """Run one InferenceWorker for the current batch file."""
        if not self._batch_running:
            return
        self._worker = InferenceWorker(
            source_wav=norm_path,
            model_path=self._batch_model_path,
            config_path=self._batch_config_path,
            output_dir=self._batch_output_dir,
            transpose=0,
            f0_method="crepe",
            auto_predict_f0=False,
            noise_scale=0.1,
            db_thresh=-35,
            pad_seconds=1.0,
            chunk_seconds=30.0,
            separate_vocals=False,
            model_type=self._batch_mt,
            model_dir=self._batch_model_dir,
            smart_transpose=self._btn_range_match.isChecked() and self._model_center_hz > 0,
            model_center_hz=self._model_center_hz,
            max_sections=20,
        )
        self._worker.log_line.connect(self._on_batch_log)
        self._worker.finished_ok.connect(self._on_batch_file_done)
        self._worker.error.connect(self._on_batch_file_error)
        self._worker.sections_used.connect(self._on_batch_sections)
        self._worker.start()

    def _on_batch_sections(self, count):
        """Tag the current row with how many sections Range-Match split
        the clip into. Emitted mid-conversion, so _batch_index still
        points at the file being converted."""
        if not self._batch_running:
            return
        if 0 <= self._batch_index < len(self._batch_files):
            path = self._batch_files[self._batch_index]
            self._convert_queue.set_sections(path, count)

    def _on_batch_log(self, line):
        self._log.append_line(line)
        # Overall ring progress = whole files done + this file's fraction.
        total = max(1, len(self._batch_files))
        frac = 0.0
        import re
        m = re.search(r"section (\d+)/(\d+)", line)
        if m:
            frac = int(m.group(1)) / max(1, int(m.group(2)))
        elif "Output saved" in line:
            frac = 1.0
        self._convert_ring.set_progress(
            min(0.99, (self._batch_index + frac) / total)
        )

    def _release_worker(self):
        """Drop the finished InferenceWorker safely.

        finished_ok / error are emitted from inside the worker thread's
        run(), so when the slot runs the QThread can still be tearing
        down. Dropping the last Python reference then makes ~QThread
        abort the process ("QThread: Destroyed while thread is still
        running"). wait() blocks — releasing the GIL so the worker can
        finish — until the thread has fully exited; only then is the
        drop (when `w` falls out of scope) safe.
        """
        w = self._worker
        self._worker = None
        if w is not None:
            w.wait(5000)

    def _on_batch_file_done(self, output_path):
        if not self._batch_running:
            return
        path = self._batch_files[self._batch_index]
        self._batch_results[path] = output_path
        self._convert_queue.set_status(path, "done", output_path)
        self._batch_done += 1
        self._log.append_line(f"  ✓ {os.path.basename(output_path)}")
        self._release_worker()
        self._batch_index += 1
        self._batch_next()

    def _on_batch_file_error(self, error):
        if not self._batch_running:
            return
        path = self._batch_files[self._batch_index]
        self._convert_queue.set_status(path, "failed")
        self._batch_failed.append(os.path.basename(path))
        self._log.append_line(f"  ✗ {os.path.basename(path)}: {error}")
        self._release_worker()
        self._batch_index += 1
        # Per the design: keep going, the failed file just stays red.
        self._batch_next()

    def _batch_finish(self):
        self._batch_running = False
        self._worker = None
        self._hide_spinner()
        self._convert_ring.set_progress(1.0)
        QTimer.singleShot(500, lambda: self._convert_ring.set_converting(False))
        # Summarise the whole queue — files done in earlier runs were
        # intentionally skipped this run but still count toward the total.
        def _count(status):
            return sum(
                1 for p in self._source_paths
                if (r := self._convert_queue.row(p)) and r.status == status
            )
        done = _count("done")
        failed = _count("failed")
        total = len(self._source_paths)
        # The header doubles as the end-of-batch summary.
        if failed:
            self._convert_queue.set_header(
                f'<span style="color:#4ade80;">✓ {done} of {total} '
                f'converted</span>'
                f'<span style="color:#ef4444;">　·　{failed} failed</span>'
            )
        else:
            self._convert_queue.set_header(
                f'<span style="color:#4ade80;">✓ All {done} files '
                f'converted</span>'
            )
        summary = f"Batch complete — {done}/{total} converted"
        if failed:
            summary += f", {failed} failed"
        self._log.append_line(summary)
        self._log.setVisible(True)
        QTimer.singleShot(50, self._position_bottom_panel)

    def _cancel_batch(self):
        """Stop a running batch after a second Convert click."""
        self._batch_running = False
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
        self._worker = None
        self._hide_spinner()
        # Any file still mid-flight reverts to queued.
        for p in self._batch_files:
            row = self._convert_queue.row(p)
            if row is not None and row.status == "converting":
                self._convert_queue.set_status(p, "queued")
        self._convert_ring.set_converting(False)
        self._convert_ring.set_progress(0.0)
        self._convert_queue.set_header(
            f'<span style="color:#f59e0b;">Cancelled — {self._batch_done} '
            f'of {len(self._batch_files)} converted</span>'
        )
        self._log.append_line(
            f"Batch cancelled — {self._batch_done} of "
            f"{len(self._batch_files)} converted."
        )

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
            self._search.setPlaceholderText("Search artists on HuggingFace...")
            self._show_dropdown()
        except Exception as e:
            self._hide_spinner()
            self._search.setPlaceholderText("Search artists on HuggingFace...")
            QMessageBox.warning(self, "Error", str(e))

    def _on_search_changed(self, text):
        if self._hf_loaded:
            self._show_dropdown(text)

    def _hf_warning_dismissed(self) -> bool:
        """Has the user opted out of the community-model download warning?"""
        from services.job_store import load_config
        return bool(load_config().get("hf_download_warning_dismissed", False))

    def _set_hf_warning_dismissed(self):
        from services.job_store import save_config
        save_config({"hf_download_warning_dismissed": True})

    def _rated_grade_for_artist(self, artist: str):
        """The user's grade rating for a downloaded model of `artist`,
        or None if it isn't downloaded or hasn't been rated."""
        try:
            import json
            meta_path = os.path.join(MODELS_DIR, artist, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    return json.load(f).get("user_grade_override") or None
        except Exception:
            pass
        return None

    def _badge_icon(self, base_px, grade: str):
        """28px dropdown icon: the artist thumbnail (if any) with the
        grade badge in the bottom-right corner — or just the badge when
        there's no thumbnail."""
        badge = QPixmap(
            os.path.join(APP_DIR, "assets", "grade_badges", f"{grade}.png")
        )
        if badge.isNull():
            return base_px
        canvas = QPixmap(28, 28)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if base_px is not None and not base_px.isNull():
            painter.drawPixmap(0, 0, base_px)
            bsize = 15
        else:
            bsize = 22
        badge = badge.scaled(
            bsize, bsize, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(28 - badge.width(), 28 - badge.height(), badge)
        painter.end()
        return canvas

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

            icon_px = None
            thumb = os.path.join(self._image_cache_dir, f"{m['artist']}.jpg")
            if os.path.exists(thumb):
                px = QPixmap(thumb)
                if not px.isNull():
                    from ui.widgets.voice_card import VoiceCard
                    icon_px = VoiceCard._make_circular(px, 28)
            # If the user has downloaded and rated this model, ride its
            # grade badge on the row icon and its letter (cyan, via the
            # delegate) at the right of the row, next to the name.
            grade = self._rated_grade_for_artist(m["artist"])
            if grade:
                icon_px = self._badge_icon(icon_px, grade)
                item.setData(_HFArtistDelegate.GRADE_ROLE, grade)
            if icon_px is not None:
                item.setIcon(QIcon(icon_px))

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

        # Download — plain clickable text, no button chrome
        btn = QLabel("Download", self._dropdown)
        btn.setObjectName("dl_btn_float")
        btn.setFixedSize(100, 26)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn.setStyleSheet("""
            QLabel {
                background: transparent;
                color: rgba(94, 200, 180, 200);
                font-size: 11px;
                font-weight: bold;
            }
            QLabel:hover {
                color: white;
            }
        """)
        btn.move(rx - 110, vy - 13)
        # Defined as a named function (not a tuple-returning lambda) so the
        # mousePressEvent override returns None like Qt expects, and any
        # exception stays contained behind a try/except.
        def _on_dl_click(_e, i=item, b=btn):
            try:
                self._download_from_dropdown(i)
            except Exception as exc:
                self._lbl_status.setText(f"Download failed: {exc}")
            finally:
                try:
                    b.deleteLater()
                except Exception:
                    pass
        btn.mousePressEvent = _on_dl_click
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

        # Community models are outside uploads — warn before downloading,
        # once, unless the user has opted out.
        if not self._hf_warning_dismissed():
            from PyQt6.QtWidgets import QCheckBox
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Community model")
            box.setText(f"'{artist}' is a community model from Hugging Face.")
            box.setInformativeText(
                "These are uploaded by outside users — not tested or "
                "verified by SomerSVC, and quality varies. After it "
                "downloads, you can rate it by clicking its grade in your "
                "models; your rating then shows next to the artist here "
                "in the download list."
            )
            dont_ask = QCheckBox("Don't show this again")
            box.setCheckBox(dont_ask)
            box.setStandardButtons(
                QMessageBox.StandardButton.Cancel
                | QMessageBox.StandardButton.Ok
            )
            box.button(QMessageBox.StandardButton.Ok).setText("Download")
            box.setDefaultButton(QMessageBox.StandardButton.Ok)
            if box.exec() != QMessageBox.StandardButton.Ok:
                return
            if dont_ask.isChecked():
                self._set_hf_warning_dismissed()

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
            self._search.setPlaceholderText("Search artists on HuggingFace...")
            self._refresh_models()
            QMessageBox.information(
                self, "Downloaded",
                f"'{artist}' is ready.\n\nIt's an untested community "
                "model — once you've tried it, rate it by clicking its "
                "grade in your models.",
            )
        except Exception as e:
            self._hide_spinner()
            self._search.setPlaceholderText("Search artists on HuggingFace...")
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
        paths = [
            u.toLocalFile() for u in event.mimeData().urls()
            if u.toLocalFile().lower().endswith((".wav", ".flac", ".mp3", ".ogg"))
        ]
        if paths:
            self._add_sources(paths)

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


    def _add_sources(self, paths):
        """Queue one or more audio files for conversion. A single file
        is the classic single-file flow; 2+ switches to the batch
        queue list. New, non-duplicate paths are appended in order.

        If a batch is already running, the new files fold into it —
        they convert with the current run, no second Convert click."""
        added = []
        for p in paths:
            if p and p not in self._source_paths:
                self._source_paths.append(p)
                added.append(p)
        if not added:
            return
        if self._batch_running:
            self._batch_files.extend(added)
        self._sync_convert_view()

    def _remove_source(self, path):
        """Drop one file from the batch queue (a row's × button)."""
        if path in self._source_paths:
            self._source_paths.remove(path)
            # The × only shows on still-queued rows, so a removal
            # mid-batch is always a not-yet-converted file — safe to
            # pull from the live run without disturbing the index.
            if self._batch_running and path in self._batch_files:
                self._batch_files.remove(path)
            self._sync_convert_view()

    def _sync_convert_view(self):
        """Keep the convert UI in step with self._source_paths: the
        single-file waveform editor for exactly one file, the batch
        queue list for 2+, a full reset for none."""
        n = len(self._source_paths)
        if n == 0:
            self._clear_source()
            return
        if n == 1:
            # Single-file mode — today's behaviour, untouched. Only
            # reload if the lone file actually changed.
            self._convert_queue.setVisible(False)
            self._convert_queue.clear()
            self._waveform.setVisible(True)
            self._lbl_pitch.setVisible(True)
            only = self._source_paths[0]
            if self._original_source != only:
                self._load_source(only)
            return
        # Batch mode — the queue list stands in for the waveform editor.
        self._convert_queue.set_files(self._source_paths)
        self._convert_queue.setVisible(True)
        self._btn_clear_source.setVisible(True)
        self._waveform.setVisible(False)
        self._waveform_output.hide_and_stop()
        self._lbl_output_name.setVisible(False)
        self._lbl_pitch.setVisible(False)
        self._btn_optimize.setVisible(False)
        self._opt_container.setVisible(False)
        # Mid-run, leave the live "Converting…" header and the spinner
        # alone — set_files already slotted the new rows in as Queued
        # and _batch_files has been extended, so they convert with the
        # current batch.
        if not self._batch_running:
            self._convert_queue.set_header(f"{n} files queued")
            self._hide_spinner()
        self._drop_zone.setText(f"♪  {n} files queued — drop more to add")
        self._drop_zone.setStyleSheet("""
            QLabel {
                background-color: rgba(37, 99, 235, 8);
                border: 1px solid rgba(37, 99, 235, 25);
                border-radius: 12px;
                color: #aaa;
                font-size: 12px;
            }
        """)
        QTimer.singleShot(50, self._position_bottom_panel)

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

    def _cleanup_orphaned_pods(self):
        """Terminate any leftover RunPod instances that have no active job."""
        from services.job_store import load_config
        config = load_config()
        api_key = config.get("runpod_api_key", "") or os.environ.get("SOMERSVC_RUNPOD_KEY", "")
        if not api_key:
            return

        class _CleanupWorker(QThread):
            done = pyqtSignal(list)
            def __init__(self, key):
                super().__init__()
                self.key = key
            def run(self):
                try:
                    from services.pod_cleanup import cleanup_orphaned_pods
                    actions = cleanup_orphaned_pods(self.key)
                    self.done.emit(actions)
                except Exception:
                    self.done.emit([])

        self._cleanup_worker = _CleanupWorker(api_key)
        self._cleanup_worker.done.connect(self._on_cleanup_done)
        self._cleanup_worker.start()

    def _on_cleanup_done(self, actions):
        for pod_id, action, reason in actions:
            print(f"Pod cleanup: {pod_id} {action} ({reason})")

    def _check_active_training(self):
        """Auto-open Create panel and resume if there is an active training job."""
        from services.job_store import get_active_jobs, load_config
        active_jobs = get_active_jobs()
        if not active_jobs:
            return
        config = load_config()
        api_key = config.get("runpod_api_key", "") or os.environ.get("SOMERSVC_RUNPOD_KEY", "")
        ssh_key = os.path.expanduser(config.get("ssh_key_path", "~/.ssh/id_rsa"))
        if not api_key:
            return
        job = active_jobs[0]
        speaker = job["speaker_name"]
        self._show_create_model()
        panel = self._create_panel
        panel._select_model(speaker)
        panel._log.setVisible(True)
        panel._log.clear_log()
        panel._log.append_line(f"Found active job: {job['job_id'][:8]} ({job['status']})")
        panel._log.append_line(f"Speaker: {speaker}")
        panel._log.append_line("Reconnecting...")
        panel._lbl_status.setText("Reconnecting to training pod...")
        panel._btn_train.setEnabled(False)
        panel._btn_continue_train.setEnabled(False)
        panel._btn_stop_train.setVisible(True)
        panel._btn_stop_train.setEnabled(True)
        panel._btn_train.setVisible(False)
        panel._btn_continue_train.setVisible(False)
        panel._progress_bar.setVisible(True)
        panel._progress_bar.setValue(55)
        panel._training = True
        # Restore the auto-stop target persisted at training start so the
        # parser still fires request_stop at the right epoch on reopen.
        rec = job.get("recommended_epochs")
        if isinstance(rec, int) and rec > 0:
            panel._recommended_epochs = rec
            panel._auto_stop_fired = False
            panel._log.append_line(f"Restored auto-stop target: {rec} epochs")
        # Restore the resume offset so the live counter shows TOTAL epochs
        # while the orchestrator's runner is reporting session-local numbers.
        prev_epochs = job.get("previous_epochs")
        if isinstance(prev_epochs, int) and prev_epochs > 0:
            panel._resume_offset = prev_epochs
        else:
            panel._resume_offset = 0
        self._show_spinner("modeling")
        from workers.resume_worker import ResumeWorker
        panel._resume_worker = ResumeWorker(
            api_key=api_key,
            ssh_key_path=ssh_key,
            models_dir=str(MODELS_DIR),
        )
        panel._resume_worker.log_line.connect(panel._on_train_log)
        panel._resume_worker.status_changed.connect(
            lambda jid, st: panel._on_status_text(f"Training ({st})")
        )
        panel._resume_worker.progress.connect(panel._progress_bar.setValue)
        panel._resume_worker.job_finished.connect(panel._on_train_done)
        panel._resume_worker.job_failed.connect(
            lambda jid, err: panel._on_train_error(err)
        )
        panel._resume_worker.start()

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
        # If a batch has run this session, jump straight to its
        # <artist>_Batch_<N> folder; otherwise the output root.
        target = OUTPUT_DIR
        if self._batch_output_dir and os.path.isdir(self._batch_output_dir):
            target = self._batch_output_dir
        subprocess.Popen(["open", str(target)])

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
