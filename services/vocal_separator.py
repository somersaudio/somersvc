"""Vocal separator using Demucs (Meta) for isolating vocals from music."""

import os
import sys
import subprocess
from pathlib import Path
from typing import Callable


class VocalSeparator:
    """Separates vocals from instrumentals using Demucs."""

    @staticmethod
    def is_available() -> bool:
        try:
            import demucs
            return True
        except ImportError:
            return False

    def separate(
        self,
        input_path: str,
        output_dir: str,
        on_log: Callable[[str], None] | None = None,
    ) -> dict[str, str]:
        """Separate audio into vocals and instrumentals.

        Returns dict with 'vocals' and 'instrumentals' file paths.
        """
        log = on_log or (lambda _: None)
        os.makedirs(output_dir, exist_ok=True)

        stem = Path(input_path).stem

        log(f"Separating vocals from '{stem}'...")

        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"

        # In the bundled .app, sys.executable is SomerSVC (not python). Re-exec
        # ourselves with --demucs-mode so main.py can hand off to demucs.
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--demucs-mode"]
        else:
            cmd = [sys.executable, "-m", "demucs"]
        cmd += [
            "--two-stems", "vocals",  # only split into vocals + no_vocals
            "-n", "htdemucs",
            "-o", output_dir,
            input_path,
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        output_lines = []
        for line in iter(process.stdout.readline, ""):
            stripped = line.rstrip("\n")
            if stripped.strip():
                output_lines.append(stripped)
                log(stripped)

        process.wait()

        if process.returncode != 0:
            # Surface Demucs' own output instead of a generic message —
            # the tail almost always names the real cause (out of
            # memory, unreadable/too-short input, missing model, ...).
            tail = "\n".join(output_lines[-12:]) or "(no output captured)"
            raise RuntimeError(
                f"Demucs exited with code {process.returncode}.\n{tail}"
            )

        # Demucs outputs to: output_dir/htdemucs/{stem}/vocals.wav and no_vocals.wav
        demucs_dir = os.path.join(output_dir, "htdemucs", stem)
        vocals_path = os.path.join(demucs_dir, "vocals.wav")
        no_vocals_path = os.path.join(demucs_dir, "no_vocals.wav")

        if not os.path.exists(vocals_path):
            raise RuntimeError(f"Vocals file not found at {vocals_path}")

        log("Vocal separation complete!")

        return {
            "vocals": vocals_path,
            "instrumentals": no_vocals_path,
        }

    def separate_for_inference(
        self,
        input_path: str,
        on_log: Callable[[str], None] | None = None,
    ) -> dict[str, str]:
        """Separate vocals for inference — uses temp directory."""
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="svc_demucs_")
        return self.separate(input_path, tmp_dir, on_log)
