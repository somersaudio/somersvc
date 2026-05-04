"""QThread worker that runs the entire training pipeline on the local
machine — no cloud GPU. A novelty path for users with very fast Macs
(M5+) or NVIDIA-equipped machines. The pod orchestrator is the
recommended path for everyone else.

Mirrors the orchestrator's high-level shape: package → preprocess →
train → save metadata. Uses subprocess for each `svc` step instead of
SSH. Honours request_stop() (SIGTERM the train child + sentinel file)
and target_epochs (baked into config.json before train).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from services.dataset_manager import DatasetManager
from services.inference_runner import _svc_argv  # bundled-app-aware


class LocalTrainingWorker(QThread):
    log_line = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)  # job_id
    error = pyqtSignal(str)

    def __init__(
        self,
        job_id: str,
        speaker_name: str,
        dataset_manager: DatasetManager,
        models_dir: str,
        f0_method: str = "dio",
        resume_from: str = "",
        target_epochs: int = 0,
    ):
        super().__init__()
        self.job_id = job_id
        self.speaker_name = speaker_name
        self.dataset_manager = dataset_manager
        self.models_dir = Path(models_dir)
        self.f0_method = f0_method
        self.resume_from = resume_from
        self.target_epochs = int(target_epochs or 0)

        self._stop_requested = False
        self._train_proc: subprocess.Popen | None = None
        self._work_dir: Path | None = None
        self._resume_offset = 0
        if resume_from:
            m = re.search(r'G_(\d+)\.pth', os.path.basename(resume_from))
            if m:
                self._resume_offset = int(m.group(1))

    def request_stop(self):
        self._stop_requested = True
        # Drop a sentinel so any inner retry loop notices, and SIGTERM the
        # current svc train child so Lightning checkpoints cleanly.
        try:
            if self._work_dir is not None:
                (self._work_dir / "_stop").touch()
        except Exception:
            pass
        if self._train_proc and self._train_proc.poll() is None:
            try:
                self._train_proc.send_signal(signal.SIGTERM)
            except Exception:
                pass

    # ---- lifecycle ----

    def run(self):
        try:
            self._run_pipeline()
            self.finished_ok.emit(self.job_id)
        except Exception as e:
            if self._stop_requested:
                self.error.emit("stopped")
            else:
                self.error.emit(str(e))

    def _log(self, msg: str):
        self.log_line.emit(msg)

    def _status(self, status: str):
        self.status_changed.emit(status)

    # ---- pipeline ----

    def _run_pipeline(self):
        # Local working dir: ~/.somersvc/cache/local_train/<speaker>/<job_id>/
        from services.paths import CACHE_DIR
        root = Path(CACHE_DIR) / "local_train" / self.speaker_name / self.job_id
        root.mkdir(parents=True, exist_ok=True)
        self._work_dir = root

        self._status("preparing")
        self.progress.emit(5)

        # Step 1: copy clips into dataset_raw/<speaker>/
        ds_raw = root / "dataset_raw" / self.speaker_name
        ds_raw.mkdir(parents=True, exist_ok=True)
        clips = self.dataset_manager.list_files(self.speaker_name)
        if not clips:
            raise RuntimeError("No clips found for this speaker.")
        self._log(f"Staging {len(clips)} clips for local preprocessing...")
        for c in clips:
            src = c.get("path") or os.path.join(c.get("dir", ""), c.get("name", ""))
            if not os.path.isfile(src):
                continue
            dst = ds_raw / os.path.basename(src)
            if not dst.exists():
                shutil.copy2(src, dst)

        # Step 2: optional resume — drop existing checkpoints into logs/44k
        logs_dir = root / "logs" / "44k"
        logs_dir.mkdir(parents=True, exist_ok=True)
        if self.resume_from and os.path.exists(self.resume_from):
            self._log(f"Using resume checkpoint: {os.path.basename(self.resume_from)}")
            shutil.copy2(
                self.resume_from, logs_dir / os.path.basename(self.resume_from)
            )
            d_path = self.resume_from.replace("G_", "D_")
            if os.path.exists(d_path):
                shutil.copy2(d_path, logs_dir / os.path.basename(d_path))
            cfg_src = os.path.join(os.path.dirname(self.resume_from), "config.json")
            if os.path.exists(cfg_src):
                cfg_dir = root / "configs" / "44k"
                cfg_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cfg_src, cfg_dir / "config.json")

        if self._stop_requested:
            raise RuntimeError("stopped")

        # Step 3: preprocess (resample → config → hubert)
        self._status("preprocessing")
        self.progress.emit(20)
        for label, args in [
            ("Resampling audio", ["pre-resample"]),
            ("Generating config", ["pre-config"]),
            ("Extracting features", ["pre-hubert", "-fm", self.f0_method]),
        ]:
            if self._stop_requested:
                raise RuntimeError("stopped")
            self._log(f"  {label}...")
            self._run_svc_step(args, cwd=root)

        # Step 4: tune config — bake target epochs and a sane batch_size
        if self._stop_requested:
            raise RuntimeError("stopped")
        self._tune_config(root)

        # Step 5: train
        self._status("training")
        self.progress.emit(55)
        self._log(
            "Starting local training. This will use your CPU/MPS/CUDA — "
            "expect it to be much slower than a cloud GPU."
        )
        self._run_train(root)

        if self._stop_requested:
            self._log("Training stopped — saving latest checkpoint.")
        else:
            self._log("Training complete!")

        # Step 6: post-process and copy result to MODELS_DIR/<speaker>/
        self._status("downloading")
        self.progress.emit(90)
        model_dir = self.models_dir / self.speaker_name
        model_dir.mkdir(parents=True, exist_ok=True)

        g_files = sorted(
            [p for p in (logs_dir).glob("G_*.pth")],
            key=lambda p: self._epoch_from_name(p.name),
        )
        if not g_files:
            raise RuntimeError("Training produced no checkpoints.")

        # If this was a resume run, rename session-written files to add
        # the offset so the on-disk filename reflects total epochs.
        if self._resume_offset > 0:
            for p in list(logs_dir.glob("G_*.pth")) + list(logs_dir.glob("D_*.pth")):
                # Skip the resume checkpoint we copied in (we'll delete it
                # at the end). Anything written by THIS session has mtime
                # newer than the work_dir creation.
                if p.stat().st_mtime > root.stat().st_ctime + 1:
                    n = self._epoch_from_name(p.name)
                    if n is None:
                        continue
                    prefix = "G" if p.name.startswith("G_") else "D"
                    new = logs_dir / f"{prefix}_{n + self._resume_offset}.pth"
                    p.rename(new)
            # Drop the original resume checkpoint after rename so we have
            # one continuous sequence.
            for prefix in ("G", "D"):
                stale = logs_dir / f"{prefix}_{self._resume_offset}.pth"
                if stale.exists():
                    # If a renamed file collides, keep the renamed one.
                    if any(p.name == stale.name and p != stale
                           for p in logs_dir.glob(f"{prefix}_*.pth")):
                        stale.unlink(missing_ok=True)

        # Re-list after renaming
        g_files = sorted(
            list(logs_dir.glob("G_*.pth")),
            key=lambda p: self._epoch_from_name(p.name) or 0,
        )
        latest = g_files[-1]
        self._log(f"Saving model: {latest.name}")
        shutil.copy2(latest, model_dir / latest.name)
        cfg_src = root / "configs" / "44k" / "config.json"
        if cfg_src.exists():
            shutil.copy2(cfg_src, model_dir / "config.json")

        self._save_metadata(model_dir, latest.name)
        self.progress.emit(100)
        self._status("completed")
        self._log(f"Local training done. Model saved to {model_dir}")

    # ---- helpers ----

    def _run_svc_step(self, svc_args: list, cwd: Path):
        cmd = _svc_argv() + svc_args
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in iter(proc.stdout.readline, ""):
            self._log(f"    {line.rstrip()}")
            if self._stop_requested:
                proc.terminate()
                break
        proc.wait()
        if proc.returncode != 0 and not self._stop_requested:
            raise RuntimeError(
                f"svc {' '.join(svc_args)} failed (exit {proc.returncode})"
            )

    def _tune_config(self, root: Path):
        """Drop a sane batch + log/eval interval and the target epoch cap
        into every config under configs/."""
        for cfg_path in root.glob("configs/*/config.json"):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
            except Exception:
                continue
            train = cfg.setdefault("train", {})
            # Conservative local batch — Mac MPS / CPU is the limiting factor.
            # Override only if the user hasn't already supplied one.
            train.setdefault("batch_size", 8)
            train["log_interval"] = 1
            target_batch = int(train.get("batch_size", 8))
            train["eval_interval"] = max(1, min(25, target_batch // 4 or 1))
            if self.target_epochs > 0:
                if self._resume_offset > 0:
                    train["epochs"] = max(1, self.target_epochs - self._resume_offset)
                else:
                    train["epochs"] = self.target_epochs
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            self._log(
                f"Local config: batch={train['batch_size']}, "
                f"eval_interval={train['eval_interval']}, "
                f"epochs={train.get('epochs', 'default')}"
            )

    def _run_train(self, root: Path):
        # so-vits-svc-fork uses np.fromstring(..., sep="") (binary mode)
        # which NumPy 2.x removed. Patch the installed file once so
        # validation doesn't crash. No-op if already patched or the file
        # isn't writable (bundled .app — main.py handles the same shim
        # at runtime via the --svc-mode dispatcher).
        self._patch_svc_fork_for_numpy2()
        cmd = _svc_argv() + [
            "train", "--model-path", str(root / "logs" / "44k"),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Prefer MPS on Apple Silicon for any speedup over CPU. svc-fork
        # may or may not honour this — best-effort.
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        self._train_proc = subprocess.Popen(
            cmd, cwd=str(root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in iter(self._train_proc.stdout.readline, ""):
            self._log(line.rstrip())
            if self._stop_requested and self._train_proc.poll() is None:
                # Already SIGTERMed in request_stop — just keep draining.
                pass
        self._train_proc.wait()
        rc = self._train_proc.returncode
        self._train_proc = None
        if rc != 0 and not self._stop_requested:
            raise RuntimeError(f"svc train failed (exit {rc})")

    def _epoch_from_name(self, name: str):
        m = re.match(r'[GD]_(\d+)\.pth', name)
        return int(m.group(1)) if m else None

    def _patch_svc_fork_for_numpy2(self) -> None:
        """One-shot textual patch of the installed so-vits-svc-fork copy:
        np.fromstring(..., sep="") → np.frombuffer(..., dtype=np.uint8).
        Idempotent. Best-effort — silent if the file isn't writable.
        """
        try:
            from so_vits_svc_fork import utils as _svc_utils
            utils_path = Path(_svc_utils.__file__)
        except Exception:
            return
        try:
            text = utils_path.read_text()
        except Exception:
            return
        # Already patched?
        if "np.frombuffer(fig.canvas.tostring_argb()" in text:
            return
        old = 'np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8, sep="")'
        new_call = 'np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)'
        if old not in text:
            return
        try:
            utils_path.write_text(text.replace(old, new_call))
            self._log(
                "Patched so-vits-svc-fork plot_spectrogram_to_numpy for "
                "NumPy 2.x compatibility."
            )
        except Exception as e:
            self._log(
                f"Could not patch so-vits-svc-fork on disk ({e}). "
                "If training crashes on validation, downgrade numpy<2 in "
                "your venv."
            )

    def _save_metadata(self, model_dir: Path, checkpoint: str):
        epoch_str = checkpoint.replace("G_", "").replace(".pth", "")
        epoch_num = int(epoch_str) if epoch_str.isdigit() else 0
        # Filename is already absolute total (post-rename), so use directly.
        total_epochs = epoch_num
        batch_size = 8
        try:
            cfg_data = json.loads((model_dir / "config.json").read_text())
            batch_size = int(cfg_data.get("train", {}).get("batch_size", 8))
        except Exception:
            pass
        # Live duration / clip count from the dataset manager
        try:
            files = self.dataset_manager.list_files(self.speaker_name)
            total_duration = sum(f.get("duration", 0) or 0 for f in files)
            total_clips = len(files)
        except Exception:
            total_duration, total_clips = 0.0, 0
        metadata = {
            "epochs": total_epochs,
            "batch_size": batch_size,
            "dataset_duration_s": round(total_duration, 1),
            "dataset_clips": total_clips,
            "checkpoint": checkpoint,
            "trained_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "trained_locally": True,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
