"""Split audio into sections by detecting silence gaps."""

import numpy as np
import soundfile as sf


def find_section_splits(
    audio_path: str,
    min_silence_sec: float = 0.08,
    silence_thresh_db: float = -25,
    min_section_sec: float = 3,
    max_sections: int = 20,
) -> list[tuple[float, float]]:
    """Find natural section boundaries in audio by detecting silence gaps.

    Returns list of (start_sec, end_sec) tuples for each section.
    """
    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # mono

    # Convert to dB envelope using short windows
    hop = int(sr * 0.01)  # 10ms hops
    n_frames = len(audio) // hop
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(audio))
        chunk = audio[start:end]
        rms[i] = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0

    # Convert to dB
    rms_db = np.where(rms > 0, 20 * np.log10(rms + 1e-10), -100)

    # Find silent frames
    is_silent = rms_db < silence_thresh_db

    # Find silence regions (consecutive silent frames)
    min_silence_frames = int(min_silence_sec / 0.01)
    silence_regions = []
    in_silence = False
    silence_start = 0

    for i, s in enumerate(is_silent):
        if s and not in_silence:
            silence_start = i
            in_silence = True
        elif not s and in_silence:
            duration = i - silence_start
            if duration >= min_silence_frames:
                # Split point is the middle of the silence
                mid = (silence_start + i) // 2
                silence_regions.append((silence_start, i, mid, duration))
            in_silence = False

    if not silence_regions:
        # No silence found — return the whole file as one section
        total_sec = len(audio) / sr
        return [(0, total_sec)]

    # Sort by silence duration (longest gaps = most likely section boundaries)
    silence_regions.sort(key=lambda x: x[3], reverse=True)

    # Pick the best split points — ensure sections are at least min_section_sec
    total_sec = len(audio) / sr
    split_times = [0.0]  # always start at 0

    min_section_frames = int(min_section_sec / 0.01)

    for _, _, mid, dur in silence_regions:
        split_sec = mid * 0.01
        # Check this split doesn't create a section shorter than min_section_sec
        too_close = False
        for existing in split_times:
            if abs(split_sec - existing) < min_section_sec:
                too_close = True
                break
        if too_close:
            continue
        # Check distance from end
        if total_sec - split_sec < min_section_sec:
            continue

        split_times.append(split_sec)

        # Limit to max_sections
        if len(split_times) >= max_sections + 1:
            break

    split_times.append(total_sec)
    split_times.sort()

    # Build sections
    sections = []
    for i in range(len(split_times) - 1):
        sections.append((split_times[i], split_times[i + 1]))

    return sections


def split_audio_file(
    audio_path: str,
    sections: list[tuple[float, float]],
    output_dir: str,
) -> list[str]:
    """Split an audio file into section files. Returns list of file paths."""
    import os
    from pathlib import Path

    audio, sr = sf.read(audio_path)
    stem = Path(audio_path).stem
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for i, (start, end) in enumerate(sections):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        section_audio = audio[start_sample:end_sample]

        path = os.path.join(output_dir, f"{stem}_section_{i + 1:02d}.wav")
        sf.write(path, section_audio, sr)
        paths.append(path)

    return paths


def analyze_section_pitches(section_paths: list[str], silence_thresh_db: float = -35) -> list[dict]:
    """Analyze median pitch of each section. Returns list of dicts with pitch info.

    Sections below silence_thresh_db are marked as silent and skipped for pitch analysis.
    """
    import librosa

    results = []
    for path in section_paths:
        try:
            y, sr = librosa.load(path, sr=22050, duration=120)
            # Check if section is effectively silence
            rms = np.sqrt(np.mean(y ** 2))
            rms_db = 20 * np.log10(rms + 1e-10)
            if rms_db < silence_thresh_db:
                results.append({"path": path, "median_hz": 0, "silent": True})
                continue

            f0, voiced, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"), sr=sr,
            )
            # voiced may be float probabilities or bool mask depending on librosa version
            voiced_mask = voiced > 0.5 if voiced.dtype != bool else voiced
            voiced_f0 = f0[voiced_mask & ~np.isnan(f0)]
            if len(voiced_f0) == 0:
                # Fallback: use any non-NaN f0 values regardless of voicing
                voiced_f0 = f0[~np.isnan(f0)]
                voiced_f0 = voiced_f0[voiced_f0 > 0]
            if len(voiced_f0) > 0:
                median = float(np.median(voiced_f0))
                results.append({"path": path, "median_hz": median, "silent": False})
            else:
                results.append({"path": path, "median_hz": 0, "silent": False})
        except Exception:
            results.append({"path": path, "median_hz": 0, "silent": False})

    return results


def calculate_section_transposes(
    sections: list[dict],
    model_center_hz: float,
    source_median_hz: float = 0,
) -> list[dict]:
    """Calculate the best transpose for each section.

    1. Find the base transpose (any semitone) to shift the song's overall
       median pitch to the model's center.
    2. For each section, adjust by ±12 on top of that base to keep sections
       in the model's sweet spot while staying in key.

    Returns sections with 'transpose' and 'base_transpose' fields.
    """
    import math

    if model_center_hz <= 0:
        for s in sections:
            s["transpose"] = 0
            s["base_transpose"] = 0
        return sections

    # Use provided source median, or compute from section medians
    if source_median_hz > 0:
        overall_median = source_median_hz
    else:
        voiced = [s["median_hz"] for s in sections if s["median_hz"] > 0]
        if not voiced:
            for s in sections:
                s["transpose"] = 0
                s["base_transpose"] = 0
            return sections
        overall_median = float(np.median(voiced))

    # Base transpose: shift overall median to model center (any semitone)
    base_transpose = round(12 * math.log2(model_center_hz / overall_median))

    for section in sections:
        if section["median_hz"] <= 0:
            section["transpose"] = base_transpose
            section["base_transpose"] = base_transpose
            continue

        # Only allow octave shifts from base transpose to stay in key
        candidates = [base_transpose + offset for offset in [0, -12, 12, -24, 24]]

        best_total = base_transpose
        best_distance = float("inf")

        for total in candidates:
            test_hz = section["median_hz"] * (2 ** (total / 12))
            if test_hz <= 0:
                continue
            distance = abs(12 * math.log2(test_hz / model_center_hz))
            # Pick closest to model center; on ties, prefer smaller absolute transpose
            if distance < best_distance or (distance == best_distance and abs(total) < abs(best_total)):
                best_distance = distance
                best_total = total

        section["transpose"] = best_total
        section["base_transpose"] = base_transpose

    return sections


def rejoin_sections(
    section_paths: list[str],
    output_path: str,
    crossfade_sec: float = 0.05,
):
    """Rejoin processed section files into one audio file with short crossfades."""
    segments = []
    sr = None
    for path in section_paths:
        audio, file_sr = sf.read(path)
        if sr is None:
            sr = file_sr
        segments.append(audio)

    if not segments:
        return

    # Normalize all segments to mono to avoid shape mismatches
    for i, seg in enumerate(segments):
        if seg.ndim == 2:
            segments[i] = seg.mean(axis=1)

    crossfade_samples = int(crossfade_sec * sr)
    result = segments[0]

    for seg in segments[1:]:
        fade_len = min(crossfade_samples, len(result), len(seg))
        if fade_len > 0:
            fade_out = np.linspace(1.0, 0.0, fade_len)
            fade_in = np.linspace(0.0, 1.0, fade_len)

            if result.ndim == 2:
                fade_out = fade_out[:, np.newaxis]
                fade_in = fade_in[:, np.newaxis]

            blended = result[-fade_len:] * fade_out + seg[:fade_len] * fade_in
            result = np.concatenate([result[:-fade_len], blended, seg[fade_len:]])
        else:
            result = np.concatenate([result, seg])

    sf.write(output_path, result, sr)
