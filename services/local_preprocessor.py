"""Local preprocessing — runs svc pre-resample, pre-config, pre-hubert on the Mac.
Outputs preprocessed data ready to upload directly to the pod for training."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


class LocalPreprocessor:
    def __init__(self, on_log: Callable[[str], None] | None = None):
        self.on_log = on_log or (lambda _: None)

    def preprocess(self, dataset_tar_path: str, f0_method: str = "dio") -> str:
        """Run preprocessing locally. Returns path to tar.gz of preprocessed data.

        The preprocessed tar contains:
        - dataset_raw/ (resampled audio)
        - dataset/ (processed features)
        - configs/44k/config.json
        - filelists/44k/*.txt
        """
        log = self.on_log

        # Create a temp working directory
        work_dir = tempfile.mkdtemp(prefix="svc_preprocess_")
        log(f"Local preprocessing in: {work_dir}")

        try:
            # Extract dataset
            log("Extracting dataset...")
            subprocess.run(
                ["tar", "xzf", dataset_tar_path, "-C", work_dir],
                check=True, capture_output=True,
            )

            # Run preprocessing steps locally
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            steps = [
                ("Resampling audio...", ["svc", "pre-resample"]),
                ("Generating config...", ["svc", "pre-config"]),
                ("Extracting features...", ["svc", "pre-hubert", "-fm", f0_method]),
            ]

            for label, cmd in steps:
                log(f"  {label}")
                proc = subprocess.Popen(
                    cmd, cwd=work_dir, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in iter(proc.stdout.readline, ""):
                    log(f"    {line.rstrip()}")
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"Local preprocessing failed at: {label}")

            # Package everything into a tar for upload
            log("Packaging preprocessed data...")
            out_tar = os.path.join(work_dir, "preprocessed.tar.gz")
            subprocess.run(
                ["tar", "czf", out_tar,
                 "-C", work_dir,
                 "dataset_raw", "dataset", "configs", "filelists"],
                check=True, capture_output=True,
            )

            size_mb = os.path.getsize(out_tar) / (1024 * 1024)
            log(f"Preprocessed data packaged: {size_mb:.1f} MB")
            return out_tar

        except Exception:
            # Clean up on error
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

    @staticmethod
    def is_available() -> bool:
        """Check if svc CLI is available locally."""
        try:
            result = subprocess.run(
                ["svc", "--help"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
