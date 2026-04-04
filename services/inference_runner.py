"""Local inference runner using so-vits-svc-fork CLI."""

import os
import subprocess
from pathlib import Path
from typing import Callable

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

        cmd = [
            "svc", "infer", source_wav,
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

        for line in iter(process.stdout.readline, ""):
            stripped = line.rstrip("\n")
            if any(s in stripped for s in SUPPRESS):
                continue
            if not stripped.strip():
                continue
            log(stripped)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Inference failed with exit code {process.returncode}")

        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file not created: {output_path}")

        log(f"Output saved: {output_path}")
        return output_path

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
