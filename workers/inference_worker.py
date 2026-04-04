"""QThread worker for async local inference."""

import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from PyQt6.QtCore import QThread, pyqtSignal

from services.inference_runner import InferenceRunner
from services.rvc_inference_runner import RVCInferenceRunner
from services.vocal_separator import VocalSeparator


class InferenceWorker(QThread):
    log_line = pyqtSignal(str)
    finished_ok = pyqtSignal(str)  # output file path
    error = pyqtSignal(str)

    def __init__(
        self,
        source_wav: str,
        model_path: str,
        config_path: str,
        output_dir: str,
        speaker: str = "",
        transpose: int = 0,
        f0_method: str = "dio",
        auto_predict_f0: bool = True,
        noise_scale: float = 0.4,
        db_thresh: int = -20,
        pad_seconds: float = 0.5,
        chunk_seconds: float = 0.5,
        separate_vocals: bool = False,
        model_type: str = "svc",
        model_dir: str = "",
        smart_transpose: bool = False,
        model_center_hz: float = 0,
    ):
        super().__init__()
        self.source_wav = source_wav
        self.model_path = model_path
        self.config_path = config_path
        self.output_dir = output_dir
        self.speaker = speaker
        self.transpose = transpose
        self.f0_method = f0_method
        self.auto_predict_f0 = auto_predict_f0
        self.noise_scale = noise_scale
        self.db_thresh = db_thresh
        self.pad_seconds = pad_seconds
        self.chunk_seconds = chunk_seconds
        self.separate_vocals = separate_vocals
        self.model_type = model_type
        self.model_dir = model_dir
        self.smart_transpose = smart_transpose
        self.model_center_hz = model_center_hz

    def run(self):
        try:
            log = self.log_line.emit

            if self.smart_transpose and self.model_center_hz > 0:
                output = self._run_smart_transpose(log)
            elif self.separate_vocals:
                output = self._run_with_separation(log)
            elif self.model_type == "rvc":
                output = self._run_rvc(self.source_wav, log)
            else:
                output = self._run_svc(self.source_wav, log)

            self.finished_ok.emit(output)
        except Exception as e:
            self.error.emit(str(e))

    def _run_svc(self, source: str, log, output_dir: str = "") -> str:
        runner = InferenceRunner()
        return runner.run(
            source_wav=source,
            model_path=self.model_path,
            config_path=self.config_path,
            output_dir=output_dir or self.output_dir,
            speaker=self.speaker,
            transpose=self.transpose,
            f0_method=self.f0_method,
            auto_predict_f0=self.auto_predict_f0,
            noise_scale=self.noise_scale,
            db_thresh=self.db_thresh,
            pad_seconds=self.pad_seconds,
            chunk_seconds=self.chunk_seconds,
            on_log=log,
        )

    def _run_rvc(self, source: str, log, output_dir: str = "") -> str:
        runner = RVCInferenceRunner()
        # Find .index file if present
        index_path = ""
        if self.model_dir:
            for f in os.listdir(self.model_dir):
                if f.endswith(".index"):
                    index_path = os.path.join(self.model_dir, f)
                    break

        return runner.run(
            source_wav=source,
            model_path=self.model_path,
            output_dir=output_dir or self.output_dir,
            transpose=self.transpose,
            f0_method="rmvpe",  # best RVC pitch method
            index_path=index_path,
            on_log=log,
        )

    def _run_with_separation(self, log):
        """Separate vocals, convert them, remix with instrumentals."""
        import tempfile

        source_name = Path(self.source_wav).stem
        output_path = os.path.join(self.output_dir, f"{source_name}.out.wav")
        tmp_dir = tempfile.mkdtemp(prefix="svc_sep_")

        try:
            # Step 1: Separate vocals
            log("Step 1/3: Separating vocals with Demucs...")
            separator = VocalSeparator()
            stems = separator.separate(self.source_wav, tmp_dir, on_log=log)

            vocals_path = stems["vocals"]
            instrumentals_path = stems["instrumentals"]

            # Step 2: Convert vocals
            log("Step 2/3: Converting vocals...")
            if self.model_type == "rvc":
                self._run_rvc(vocals_path, log, output_dir=tmp_dir)
            else:
                self._run_svc(vocals_path, log, output_dir=tmp_dir)

            # The runner saves as {stem}.out.wav
            vocals_converted = os.path.join(tmp_dir, "vocals.out.wav")

            # Step 3: Remix
            log("Step 3/3: Remixing vocals with instrumentals...")
            converted, sr = sf.read(vocals_converted)
            instrumentals, sr2 = sf.read(instrumentals_path)

            # Match lengths
            min_len = min(len(converted), len(instrumentals))
            converted = converted[:min_len]
            instrumentals = instrumentals[:min_len]

            # Handle mono/stereo mismatch
            if converted.ndim == 1 and instrumentals.ndim == 2:
                converted = np.column_stack([converted, converted])
            elif converted.ndim == 2 and instrumentals.ndim == 1:
                instrumentals = np.column_stack([instrumentals, instrumentals])

            # Mix
            mixed = converted + instrumentals

            # Normalize to prevent clipping
            peak = np.max(np.abs(mixed))
            if peak > 0.95:
                mixed = mixed * (0.95 / peak)

            os.makedirs(self.output_dir, exist_ok=True)
            sf.write(output_path, mixed, sr)
            log(f"Output saved: {output_path}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return output_path

    def _run_smart_transpose(self, log):
        """Split audio into sections, transpose each by ±12 to match model, then rejoin."""
        import tempfile
        from services.section_splitter import (
            find_section_splits, split_audio_file,
            analyze_section_pitches, calculate_section_transposes, rejoin_sections,
        )

        source_name = Path(self.source_wav).stem
        output_path = os.path.join(self.output_dir, f"{source_name}.out.wav")
        tmp_dir = tempfile.mkdtemp(prefix="svc_smart_")

        try:
            # Step 1: Optionally separate vocals first
            process_path = self.source_wav
            instrumentals_path = None
            if self.separate_vocals:
                log("Separating vocals...")
                from services.vocal_separator import VocalSeparator
                separator = VocalSeparator()
                stems = separator.separate(self.source_wav, tmp_dir, on_log=log)
                process_path = stems["vocals"]
                instrumentals_path = stems["instrumentals"]

            # Step 2: Find section splits
            log("Detecting sections...")
            sections = find_section_splits(process_path)
            log(f"Found {len(sections)} sections")

            if len(sections) <= 1:
                # No splits found — fall back to normal inference
                log("No section breaks detected — using standard transpose")
                if self.model_type == "rvc":
                    converted = self._run_rvc(process_path, log)
                else:
                    converted = self._run_svc(process_path, log)

                if instrumentals_path:
                    self._remix(converted, instrumentals_path, output_path, log)
                    return output_path
                return converted

            # Step 3: Split audio
            section_dir = os.path.join(tmp_dir, "sections")
            section_paths = split_audio_file(process_path, sections, section_dir)

            # Step 4: Analyze pitch of each section
            log("Analyzing section pitches...")
            section_info = analyze_section_pitches(section_paths)

            # Step 5: Calculate per-section transpose
            section_info = calculate_section_transposes(section_info, self.model_center_hz)

            base = section_info[0].get("base_transpose", 0)
            from ui.pages.inference_page import _hz_to_note
            model_note = _hz_to_note(self.model_center_hz)
            log(f"Base transpose: {base:+d} semitones (to match {model_note})")

            for i, info in enumerate(section_info):
                dur = sections[i][1] - sections[i][0]
                note = _hz_to_note(info["median_hz"]) if info["median_hz"] > 0 else "?"
                extra = info["transpose"] - base
                extra_str = f" + octave {extra:+d}" if extra != 0 else ""
                log(f"  Section {i + 1}: {dur:.1f}s, center {note}, transpose {info['transpose']:+d}{extra_str}")

            # Step 6: Run inference on each section with its transpose
            converted_paths = []
            conv_dir = os.path.join(tmp_dir, "converted")
            os.makedirs(conv_dir, exist_ok=True)

            for i, info in enumerate(section_info):
                log(f"Converting section {i + 1}/{len(section_info)} (transpose {info['transpose']:+d})...")
                # Temporarily override transpose
                original_transpose = self.transpose
                self.transpose = info["transpose"]

                if self.model_type == "rvc":
                    out = self._run_rvc(info["path"], log, output_dir=conv_dir)
                else:
                    out = self._run_svc(info["path"], log, output_dir=conv_dir)

                self.transpose = original_transpose
                converted_paths.append(out)

            # Step 7: Rejoin sections
            log("Rejoining sections...")
            joined_path = os.path.join(tmp_dir, "joined.wav")
            rejoin_sections(converted_paths, joined_path)

            # Step 8: Remix with instrumentals if separated
            if instrumentals_path:
                self._remix(joined_path, instrumentals_path, output_path, log)
            else:
                shutil.copy2(joined_path, output_path)

            log(f"Output saved: {output_path}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return output_path

    def _remix(self, vocals_path, instrumentals_path, output_path, log):
        """Mix converted vocals with instrumentals."""
        log("Remixing with instrumentals...")
        converted, sr = sf.read(vocals_path)
        instrumentals, _ = sf.read(instrumentals_path)

        min_len = min(len(converted), len(instrumentals))
        converted = converted[:min_len]
        instrumentals = instrumentals[:min_len]

        if converted.ndim == 1 and instrumentals.ndim == 2:
            converted = np.column_stack([converted, converted])
        elif converted.ndim == 2 and instrumentals.ndim == 1:
            instrumentals = np.column_stack([instrumentals, instrumentals])

        mixed = converted + instrumentals
        peak = np.max(np.abs(mixed))
        if peak > 0.95:
            mixed = mixed * (0.95 / peak)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, mixed, sr)
