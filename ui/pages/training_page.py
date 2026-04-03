"""Training page for launching and monitoring RunPod training jobs."""

import os
import re
from datetime import datetime, timezone

from PyQt6.QtCore import Qt, QTimer, QElapsedTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from services.job_store import create_job, get_active_jobs, list_jobs, load_config
from ui.widgets.log_viewer import LogViewer
from workers.resume_worker import ResumeWorker
from workers.training_worker import TrainingWorker

APP_DIR = None  # Set by main_window


class TrainingPage(QWidget):
    def __init__(self, get_speaker_name, get_dataset_manager, get_api_key, get_ssh_key_path, models_dir, parent=None):
        super().__init__(parent)
        self.get_speaker_name = get_speaker_name
        self.get_dataset_manager = get_dataset_manager
        self.get_api_key = get_api_key
        self.get_ssh_key_path = get_ssh_key_path
        self.models_dir = models_dir
        self._worker: TrainingWorker | None = None
        self._resume_worker: ResumeWorker | None = None
        self._elapsed_offset_s = 0  # for resumed jobs
        self._init_ui()
        self._refresh_job_history()
        self._try_resume_active_jobs()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Training")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Train a voice model on RunPod GPU")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        # Selected voice display
        self.lbl_voice = QLabel("No voice selected")
        self.lbl_voice.setStyleSheet("font-size: 16px; font-weight: bold; color: #5599ff; padding: 4px 0;")
        layout.addWidget(self.lbl_voice)

        # Options row
        opts_row = QHBoxLayout()
        opts_row.addWidget(QLabel("Voice Type:"))
        self.cmb_f0 = QComboBox()
        self.cmb_f0.addItems(["Singing", "Speech"])
        opts_row.addWidget(self.cmb_f0)

        opts_row.addSpacing(20)
        opts_row.addWidget(QLabel("Max Epochs:"))
        self.txt_max_epochs = QLineEdit()
        self.txt_max_epochs.setPlaceholderText("auto")
        self.txt_max_epochs.setFixedWidth(80)
        self.txt_max_epochs.setToolTip("Leave blank for auto-recommendation based on dataset size")
        opts_row.addWidget(self.txt_max_epochs)

        opts_row.addStretch()
        layout.addLayout(opts_row)

        # Resume option row
        resume_row = QHBoxLayout()
        self.chk_resume = QCheckBox("Resume from existing model")
        self.chk_resume.setToolTip("Continue training from the latest checkpoint instead of starting fresh")
        resume_row.addWidget(self.chk_resume)
        resume_row.addStretch()
        layout.addLayout(resume_row)

        # Start button
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self._start_training)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._cancel_training)
        btn_row.addWidget(self.btn_stop)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Status
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 14px;")
        status_row.addWidget(self.lbl_status)
        status_row.addStretch()
        layout.addLayout(status_row)

        # Epoch counter + elapsed time
        progress_header = QHBoxLayout()
        self.lbl_epochs = QLabel("")
        self.lbl_epochs.setStyleSheet("color: #5599ff; font-size: 14px; font-weight: bold;")
        progress_header.addWidget(self.lbl_epochs)
        progress_header.addStretch()
        self.lbl_elapsed = QLabel("")
        self.lbl_elapsed.setStyleSheet("color: #888; font-size: 13px;")
        progress_header.addWidget(self.lbl_elapsed)
        layout.addLayout(progress_header)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self._current_epoch = 0
        self._recommended_epochs = 2000  # default, updated based on dataset duration

        # Timer for elapsed time
        self._elapsed_timer = QElapsedTimer()
        self._tick_timer = QTimer()
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._update_elapsed)

        # Log viewer
        layout.addWidget(QLabel("Training Log:"))
        self.log_viewer = LogViewer()
        layout.addWidget(self.log_viewer, 1)

        # Job history
        layout.addWidget(QLabel("Job History:"))
        self.job_list = QListWidget()
        self.job_list.setMaximumHeight(120)
        layout.addWidget(self.job_list)

    def _start_training(self):
        speaker = self.get_speaker_name()
        self.lbl_voice.setText(f"Voice: {speaker}" if speaker else "No voice selected")
        if not speaker:
            QMessageBox.warning(self, "Missing", "Set a speaker name on the Dataset page first.")
            return

        api_key = self.get_api_key()
        if not api_key:
            QMessageBox.warning(self, "Missing", "Set your RunPod API key on the Settings page first.")
            return

        ssh_key = self.get_ssh_key_path()
        dataset_mgr = self.get_dataset_manager()

        # Validate dataset
        warnings = dataset_mgr.validate(speaker)
        no_files = any("No audio files" in w for w in warnings)
        if no_files:
            QMessageBox.warning(self, "No Data", "Add audio files on the Dataset page first.")
            return

        # Create job
        job = create_job(speaker)
        self.log_viewer.clear_log()
        self.log_viewer.append_line(f"Job created: {job['job_id'][:8]}...")

        # Find resume checkpoint if requested
        resume_from = ""
        if self.chk_resume.isChecked():
            model_dir = os.path.join(self.models_dir, speaker)
            if os.path.isdir(model_dir):
                g_files = sorted([
                    f for f in os.listdir(model_dir)
                    if f.startswith("G_") and f.endswith(".pth")
                ])
                if g_files:
                    resume_from = os.path.join(model_dir, g_files[-1])
                    self.log_viewer.append_line(f"Resuming from: {g_files[-1]}")
                else:
                    self.log_viewer.append_line("No existing checkpoint found — training from scratch")
            else:
                self.log_viewer.append_line("No existing model found — training from scratch")

        # Start worker
        self._worker = TrainingWorker(
            job_id=job["job_id"],
            speaker_name=speaker,
            api_key=api_key,
            ssh_key_path=ssh_key,
            dataset_manager=dataset_mgr,
            models_dir=self.models_dir,
            f0_method="crepe" if self.cmb_f0.currentText() == "Singing" else "dio",
            resume_from=resume_from,
        )
        self._worker.log_line.connect(self._on_log_line)
        self._worker.status_changed.connect(self._on_status_changed)
        self._worker.progress.connect(self.progress.setValue)

        # Calculate recommended epochs based on dataset duration
        self._update_recommended_epochs(speaker, dataset_mgr)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Starting...")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: #f59e0b;")
        self._elapsed_offset_s = 0
        self._elapsed_timer.start()
        self._tick_timer.start()

    def _cancel_training(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self.lbl_status.setText("Stopping training & downloading model...")
            self.lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: #f59e0b;")
            self.btn_stop.setEnabled(False)

    def _on_status_changed(self, status: str):
        labels = {
            "packaging": "Packaging dataset...",
            "creating_pod": "Creating GPU instance...",
            "waiting_for_pod": "Waiting for pod to start...",
            "connecting": "Connecting via SSH...",
            "installing": "Installing dependencies...",
            "uploading": "Uploading dataset...",
            "preprocessing": "Preprocessing audio...",
            "training": "Training model...",
            "downloading": "Downloading model...",
            "completed": "Completed!",
        }
        self.lbl_status.setText(labels.get(status, status))
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: #f59e0b;")

    def _on_finished(self, job_id: str):
        self._tick_timer.stop()
        self.lbl_status.setText("Training Complete!")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: #22c55e;")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(100)
        self._refresh_job_history()
        QMessageBox.information(self, "Done", "Training complete! Your model is ready for inference.")

    def _on_error(self, error: str):
        self._tick_timer.stop()
        self.lbl_status.setText(f"Failed: {error[:60]}")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: #ef4444;")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._refresh_job_history()

    def _on_log_line(self, line: str):
        """Process each log line — display it and parse epoch info."""
        self.log_viewer.append_line(line)

        # Parse epoch from training output: "Epoch 1234/9999"
        match = re.search(r'Epoch (\d+)/(\d+)', line)
        if match:
            self._current_epoch = int(match.group(1))
            self._update_epoch_display()

            # Auto-stop when we hit the target
            if self._current_epoch >= self._recommended_epochs:
                if self._worker and self._worker.isRunning():
                    self.log_viewer.append_line(
                        f"Reached target of {self._recommended_epochs} epochs — stopping & downloading model..."
                    )
                    self._worker.request_stop()
                    self.lbl_status.setText("Target reached — downloading model...")
                    self.lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: #22c55e;")
                    self.btn_stop.setEnabled(False)

    def _update_recommended_epochs(self, speaker: str, dataset_mgr):
        """Set recommended epochs based on user input or dataset duration."""
        # Check if user specified a max
        user_input = self.txt_max_epochs.text().strip()
        if user_input.isdigit() and int(user_input) > 0:
            self._recommended_epochs = int(user_input)
        else:
            # Auto-calculate from dataset duration
            files = dataset_mgr.list_files(speaker)
            total_duration = sum(f.get("duration", 0) or 0 for f in files)

            if total_duration <= 0:
                self._recommended_epochs = 2000
            elif total_duration < 180:  # < 3 min
                self._recommended_epochs = 3000
            elif total_duration < 300:  # < 5 min
                self._recommended_epochs = 2500
            elif total_duration < 600:  # < 10 min
                self._recommended_epochs = 1500
            elif total_duration < 1800:  # < 30 min
                self._recommended_epochs = 500
            else:  # 30+ min
                self._recommended_epochs = 300

            # Show the auto value in the placeholder
            self.txt_max_epochs.setPlaceholderText(str(self._recommended_epochs))

        self._current_epoch = 0
        self._update_epoch_display()

    def _update_epoch_display(self):
        rec = self._recommended_epochs
        cur = self._current_epoch
        self.lbl_epochs.setText(f"{cur}/{rec}E")

        # Update progress bar based on epoch progress
        if rec > 0 and cur > 0:
            pct = min(int((cur / rec) * 100), 100)
            self.progress.setValue(pct)

        # Color: green when past recommended, blue when in progress
        if cur >= rec:
            self.lbl_epochs.setStyleSheet("color: #22c55e; font-size: 14px; font-weight: bold;")
        else:
            self.lbl_epochs.setStyleSheet("color: #5599ff; font-size: 14px; font-weight: bold;")

    def _update_elapsed(self):
        ms = self._elapsed_timer.elapsed()
        total_s = (ms // 1000) + self._elapsed_offset_s
        h = total_s // 3600
        m = (total_s % 3600) // 60
        sec = total_s % 60
        if h > 0:
            self.lbl_elapsed.setText(f"Elapsed: {h}:{m:02d}:{sec:02d}")
        else:
            self.lbl_elapsed.setText(f"Elapsed: {m}:{sec:02d}")

    def _try_resume_active_jobs(self):
        """On app startup, check for active jobs and resume monitoring."""
        active_jobs = get_active_jobs()
        if not active_jobs:
            return

        config = load_config()
        api_key = config.get("runpod_api_key", "")
        ssh_key = config.get("ssh_key_path", "~/.ssh/id_rsa")
        ssh_key = os.path.expanduser(ssh_key)

        if not api_key:
            return

        job = active_jobs[0]  # Resume the most recent active job
        self.log_viewer.append_line(f"Found active job: {job['job_id'][:8]} ({job['status']})")
        self.log_viewer.append_line(f"Speaker: {job['speaker_name']}")
        self.log_viewer.append_line("Attempting to reconnect...")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Calculate elapsed time from job creation
        try:
            created = datetime.fromisoformat(job["created_at"])
            now = datetime.now(timezone.utc)
            self._elapsed_offset_s = int((now - created).total_seconds())
        except Exception:
            self._elapsed_offset_s = 0

        self._elapsed_timer.start()
        self._tick_timer.start()

        # Show current status
        self._on_status_changed(job["status"])

        # Start resume worker
        self._resume_worker = ResumeWorker(
            api_key=api_key,
            ssh_key_path=ssh_key,
            models_dir=self.models_dir,
        )
        self._resume_worker.log_line.connect(self.log_viewer.append_line)
        self._resume_worker.status_changed.connect(
            lambda jid, st: self._on_status_changed(st)
        )
        self._resume_worker.progress.connect(self.progress.setValue)
        self._resume_worker.job_finished.connect(self._on_finished)
        self._resume_worker.job_failed.connect(
            lambda jid, err: self._on_error(err)
        )
        self._resume_worker.elapsed_seconds.connect(self._on_elapsed_update)
        self._resume_worker.start()

    def _on_elapsed_update(self, seconds: int):
        """Update elapsed offset when resume worker reports job age."""
        self._elapsed_offset_s = seconds
        self._elapsed_timer.restart()

    def _refresh_job_history(self):
        self.job_list.clear()
        for job in reversed(list_jobs()):
            status_icon = {
                "completed": "[OK]",
                "failed": "[FAIL]",
                "pending": "[...]",
            }.get(job["status"], "[...]")
            text = f"{status_icon}  {job['speaker_name']}  |  {job['status']}  |  {job['created_at'][:16]}"
            item = QListWidgetItem(text)
            self.job_list.addItem(item)

    def cleanup(self):
        if self._resume_worker:
            self._resume_worker.stop()
            self._resume_worker.wait(3000)
