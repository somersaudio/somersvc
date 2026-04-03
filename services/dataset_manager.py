"""Manage local datasets for voice training."""

import math
import os
import shutil
import struct
import tarfile
import tempfile
import wave
from pathlib import Path

CLIP_DURATION = 7  # seconds — sweet spot for so-vits-svc-fork
SILENCE_THRESHOLD = 0.01  # RMS below this = silence (range 0.0 - 1.0)


class DatasetManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_speaker_dir(self, speaker_name: str) -> Path:
        return self.base_dir / speaker_name

    def create_speaker(self, speaker_name: str) -> Path:
        speaker_dir = self.get_speaker_dir(speaker_name)
        speaker_dir.mkdir(parents=True, exist_ok=True)
        return speaker_dir

    def add_files(self, speaker_name: str, file_paths: list[str]) -> list[str]:
        """Add audio files — auto-splits anything longer than CLIP_DURATION into 7s chunks."""
        speaker_dir = self.create_speaker(speaker_name)
        warnings = []

        for fp in file_paths:
            fp = Path(fp)
            if fp.suffix.lower() not in (".wav", ".flac", ".mp3", ".ogg"):
                warnings.append(f"Skipped {fp.name}: unsupported format")
                continue

            if fp.suffix.lower() == ".wav":
                try:
                    duration = self._get_wav_duration(str(fp))
                    if duration > CLIP_DURATION + 1:
                        # Auto-split into 7-second clips
                        count = self._split_wav(str(fp), speaker_dir, fp.stem)
                        warnings.append(f"{fp.name}: split into {count} clips ({duration:.0f}s total)")
                        continue
                except Exception:
                    pass

            # Short enough or non-WAV — just copy
            dest = self._unique_path(speaker_dir, fp.name)
            shutil.copy2(str(fp), str(dest))

        return warnings

    def _split_wav(self, wav_path: str, output_dir: Path, base_name: str) -> int:
        """Split a WAV file into CLIP_DURATION-second chunks. Returns number of clips created."""
        with wave.open(wav_path, "r") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            total_frames = wf.getnframes()

            frames_per_clip = int(framerate * CLIP_DURATION)
            num_clips = math.ceil(total_frames / frames_per_clip)
            count = 0

            for i in range(num_clips):
                remaining = total_frames - (i * frames_per_clip)
                chunk_frames = min(frames_per_clip, remaining)

                # Skip very short tail clips (under 1 second)
                if chunk_frames < framerate:
                    continue

                data = wf.readframes(chunk_frames)

                # Skip silent clips
                if self._is_silent(data, sampwidth):
                    continue

                clip_name = f"{base_name}_clip{i + 1:03d}.wav"
                clip_path = self._unique_path(output_dir, clip_name)

                with wave.open(str(clip_path), "w") as out:
                    out.setnchannels(n_channels)
                    out.setsampwidth(sampwidth)
                    out.setframerate(framerate)
                    out.writeframes(data)

                count += 1

        return count

    @staticmethod
    def _is_silent(data: bytes, sampwidth: int) -> bool:
        """Check if audio data is effectively silent by computing RMS."""
        if not data:
            return True

        # Unpack samples based on bit depth
        if sampwidth == 2:
            fmt = f"<{len(data) // 2}h"
            samples = struct.unpack(fmt, data)
            max_val = 32768.0
        elif sampwidth == 3:
            # 24-bit: unpack manually
            samples = []
            for j in range(0, len(data), 3):
                sample = int.from_bytes(data[j:j + 3], byteorder="little", signed=True)
                samples.append(sample)
            max_val = 8388608.0
        elif sampwidth == 4:
            fmt = f"<{len(data) // 4}i"
            samples = struct.unpack(fmt, data)
            max_val = 2147483648.0
        elif sampwidth == 1:
            # 8-bit unsigned
            samples = [s - 128 for s in data]
            max_val = 128.0
        else:
            return False

        if not samples:
            return True

        # Compute RMS normalized to 0.0 - 1.0
        sum_sq = sum(s * s for s in samples)
        rms = (sum_sq / len(samples)) ** 0.5 / max_val

        return rms < SILENCE_THRESHOLD

    @staticmethod
    def _unique_path(directory: Path, filename: str) -> Path:
        """Return a unique file path, adding numeric suffix if needed."""
        dest = directory / filename
        if not dest.exists():
            return dest
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        i = 1
        while dest.exists():
            dest = directory / f"{stem}_{i}{suffix}"
            i += 1
        return dest

    def remove_file(self, speaker_name: str, filename: str):
        filepath = self.get_speaker_dir(speaker_name) / filename
        if filepath.exists():
            filepath.unlink()

    def list_files(self, speaker_name: str) -> list[dict]:
        """Return list of files with metadata."""
        speaker_dir = self.get_speaker_dir(speaker_name)
        if not speaker_dir.exists():
            return []

        files = []
        for fp in sorted(speaker_dir.iterdir()):
            if fp.is_file() and fp.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
                info = {"name": fp.name, "path": str(fp), "size_mb": fp.stat().st_size / (1024 * 1024)}
                if fp.suffix.lower() == ".wav":
                    try:
                        info["duration"] = self._get_wav_duration(str(fp))
                    except Exception:
                        info["duration"] = None
                files.append(info)
        return files

    def get_file_count(self, speaker_name: str) -> int:
        return len(self.list_files(speaker_name))

    def package(self, speaker_name: str) -> str:
        """Create a tar.gz of the dataset_raw directory structure for upload."""
        speaker_dir = self.get_speaker_dir(speaker_name)
        if not speaker_dir.exists() or not any(speaker_dir.iterdir()):
            raise ValueError(f"No files found for speaker '{speaker_name}'")

        # Create dataset_raw structure in a temp dir
        tmp_dir = tempfile.mkdtemp()
        dataset_raw = Path(tmp_dir) / "dataset_raw" / speaker_name
        dataset_raw.mkdir(parents=True)

        for fp in speaker_dir.iterdir():
            if fp.is_file():
                shutil.copy2(str(fp), str(dataset_raw / fp.name))

        # Create tar.gz
        tar_path = os.path.join(tmp_dir, "dataset.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(Path(tmp_dir) / "dataset_raw"), arcname="dataset_raw")

        return tar_path

    def validate(self, speaker_name: str) -> list[str]:
        """Validate dataset and return warnings."""
        warnings = []
        files = self.list_files(speaker_name)

        if len(files) == 0:
            warnings.append("No audio files found")
            return warnings

        if len(files) < 10:
            warnings.append(f"Only {len(files)} files. Recommend at least 20-50 for good results")

        total_duration = 0
        for f in files:
            dur = f.get("duration")
            if dur:
                total_duration += dur

        if total_duration > 0 and total_duration < 60:
            warnings.append(f"Total audio is only {total_duration:.0f}s. Recommend at least 5-10 minutes")

        return warnings

    @staticmethod
    def _get_wav_duration(path: str) -> float:
        with wave.open(path, "r") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
