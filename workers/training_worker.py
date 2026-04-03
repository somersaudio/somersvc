"""QThread worker for async training orchestration."""

from PyQt6.QtCore import QThread, pyqtSignal

from services.dataset_manager import DatasetManager
from services.training_orchestrator import TrainingOrchestrator


class TrainingWorker(QThread):
    log_line = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)  # job_id
    error = pyqtSignal(str)

    def __init__(
        self,
        job_id: str,
        speaker_name: str,
        api_key: str,
        ssh_key_path: str,
        dataset_manager: DatasetManager,
        models_dir: str,
        f0_method: str = "dio",
        resume_from: str = "",
    ):
        super().__init__()
        self.job_id = job_id
        self.speaker_name = speaker_name
        self.api_key = api_key
        self.ssh_key_path = ssh_key_path
        self.dataset_manager = dataset_manager
        self.models_dir = models_dir
        self.f0_method = f0_method
        self.resume_from = resume_from
        self._orchestrator: TrainingOrchestrator | None = None

    def run(self):
        try:
            self._orchestrator = TrainingOrchestrator(
                api_key=self.api_key,
                ssh_key_path=self.ssh_key_path,
                dataset_manager=self.dataset_manager,
                models_dir=self.models_dir,
                on_log=self.log_line.emit,
                on_status=self.status_changed.emit,
                on_progress=self.progress.emit,
                resume_from=self.resume_from,
            )
            self._orchestrator.run(self.job_id, self.speaker_name, self.f0_method)
            self.finished_ok.emit(self.job_id)
        except Exception as e:
            self.error.emit(str(e))

    def request_stop(self):
        """Gracefully stop training — kills svc train, then downloads model."""
        if self._orchestrator:
            self._orchestrator.request_stop()
