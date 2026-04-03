"""Local inference runner using so-vits-svc-fork CLI."""

import os
import subprocess
from pathlib import Path
from typing import Callable


class InferenceRunner:
    def run(
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
        on_log: Callable[[str], None] | None = None,
    ) -> str:
        """Run voice conversion on a source WAV file. Returns output file path."""
        log = on_log or (lambda _: None)

        os.makedirs(output_dir, exist_ok=True)

        source_name = Path(source_wav).stem
        output_path = os.path.join(output_dir, f"{source_name}.out.wav")

        cmd = [
            "svc", "infer", source_wav,
            "-m", model_path,
            "-c", config_path,
            "-o", output_path,
            "-t", str(transpose),
            "-fm", f0_method,
            "-n", str(noise_scale),
            "-db", str(db_thresh),
            "-p", str(pad_seconds),
            "-ch", str(chunk_seconds),
        ]

        if speaker:
            cmd.extend(["-s", speaker])

        if auto_predict_f0:
            cmd.append("-a")
        else:
            cmd.append("-na")

        log(f"Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):
            log(line.rstrip("\n"))

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Inference failed with exit code {process.returncode}")

        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file not created: {output_path}")

        log(f"Output saved: {output_path}")
        return output_path
