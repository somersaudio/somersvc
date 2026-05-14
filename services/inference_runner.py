"""Local inference runner using so-vits-svc-fork CLI."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable


def _svc_argv() -> list[str]:
    """Resolve how to invoke the svc CLI in both dev and bundled-app modes.

    - Dev: a venv exists next to sys.executable with bin/svc → call that.
    - Bundled .app: no separate svc binary; re-exec ourselves with
      --svc-mode and main.py hands off to so-vits-svc-fork's CLI.
    """
    dev_bin = os.path.join(os.path.dirname(sys.executable), "svc")
    if os.path.exists(dev_bin):
        return [dev_bin]
    return [sys.executable, "--svc-mode"]

SUPPRESS = [
    "UserWarning", "FutureWarning", "UNEXPECTED", "HTTP Request",
    "unauthenticated", "HF_TOKEN", "weight_norm", "MPS: The constant padding",
    "Unused arguments", "Speaker None is not found", "Speaker 0 is not found",
    "Loading weights:", "HubertModel LOAD REPORT", "Key               |",
    "------------------+", "final_proj", "Notes:", "- UNEXPECTED",
    "can be ignored", "auto_predict_f0", "transpose", "WARNING",
    "huggingface.co", "HTTP/1.1", "_http.py", "faster downloads",
    "Redirect", "Decoder type:", "hifi-gan",
]


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

        cmd = _svc_argv() + [
            "infer", source_wav,
            "-m", model_path,
            "-o", output_path,
            "-s", "0",
            "-t", str(transpose),
            "-fm", f0_method,
            "-n", str(noise_scale),
            "-db", str(db_thresh),
            "-p", str(pad_seconds),
            "-ch", str(chunk_seconds),
        ]

        # Generate default config if missing
        if not config_path or not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(model_path), "config.json")
            if not os.path.exists(config_path):
                self._generate_default_config(config_path)

        cmd.insert(cmd.index("-o"), "-c")
        cmd.insert(cmd.index("-o"), config_path)

        if auto_predict_f0:
            cmd.append("-a")
        else:
            cmd.append("-na")

        log(f"Running: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        # Capture the FULL unfiltered subprocess output to a debug log so we
        # can see the actual traceback when inference fails. The on-screen
        # log still gets the SUPPRESS-filtered stream so the UI stays clean.
        full_lines = []
        for line in iter(process.stdout.readline, ""):
            stripped = line.rstrip("\n")
            full_lines.append(stripped)
            if any(s in stripped for s in SUPPRESS):
                continue
            if not stripped.strip():
                continue
            log(stripped)

        process.wait()

        if process.returncode != 0:
            # Persist the entire subprocess output so the user can send us
            # the real error. Also include the tail in the error message
            # so the dialog itself shows something useful.
            debug_path = self._write_debug_log(full_lines, "infer")
            tail = "\n".join(full_lines[-25:]) if full_lines else "(no output captured)"
            raise RuntimeError(
                f"Inference failed with exit code {process.returncode}.\n\n"
                f"Last output:\n{tail}\n\n"
                f"Full log saved to: {debug_path}"
            )

        if not os.path.exists(output_path):
            debug_path = self._write_debug_log(full_lines, "infer-no-output")
            raise RuntimeError(
                f"Output file not created: {output_path}\n\n"
                f"Full log saved to: {debug_path}"
            )

        log(f"Output saved: {output_path}")
        return output_path

    @staticmethod
    def _write_debug_log(lines: list, prefix: str) -> str:
        """Dump full subprocess output to ~/.somersvc/output/debug/ so the
        user can attach it when reporting the failure."""
        try:
            from services.paths import OUTPUT_DIR
            import datetime
            debug_dir = os.path.join(str(OUTPUT_DIR), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(debug_dir, f"{prefix}-{stamp}.log")
            with open(path, "w") as f:
                f.write("\n".join(lines))
            return path
        except Exception:
            return "(could not save debug log)"

    @staticmethod
    def _generate_default_config(config_path: str):
        """Generate a default so-vits-svc config.json for downloaded models."""
        import json
        config = {
            "train": {
                "log_interval": 100, "eval_interval": 200, "seed": 1234,
                "epochs": 10000, "learning_rate": 0.0001,
                "betas": [0.8, 0.99], "eps": 1e-09, "batch_size": 16,
                "fp16_run": False, "bf16_run": False, "lr_decay": 0.999875,
                "segment_size": 10240, "init_lr_ratio": 1, "warmup_epochs": 0,
                "c_mel": 45, "c_kl": 1.0, "use_sr": True, "max_speclen": 512,
            },
            "data": {
                "training_files": "filelists/44k/train.txt",
                "validation_files": "filelists/44k/val.txt",
                "max_wav_value": 32768.0, "sampling_rate": 44100,
                "filter_length": 2048, "hop_length": 512, "win_length": 2048,
                "n_mel_channels": 80, "mel_fmin": 0.0, "mel_fmax": 22050,
                "contentvec_final_proj": False,
            },
            "model": {
                "inter_channels": 192, "hidden_channels": 192,
                "filter_channels": 768, "n_heads": 2, "n_layers": 6,
                "kernel_size": 3, "p_dropout": 0.1, "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [8, 8, 2, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 4, 4, 4],
                "n_layers_q": 3, "use_spectral_norm": False,
                "gin_channels": 256, "ssl_dim": 768, "n_speakers": 200,
                "type_": "hifi-gan",
            },
            "spk": {"speaker": 0},
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
