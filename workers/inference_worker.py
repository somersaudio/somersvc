"""QThread worker for async local inference."""

from PyQt6.QtCore import QThread, pyqtSignal

from services.inference_runner import InferenceRunner


class InferenceWorker(QThread):
    log_line = pyqtSignal(str)
    finished_ok = pyqtSignal(str)  # output file path
    error = pyqtSignal(str)

    def __init__(
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
    ):
        super().__init__()
        self.source_wav = source_wav
        self.model_path = model_path
        self.config_path = config_path
        self.output_dir = output_dir
        self.speaker = speaker
        self.transpose = transpose
        self.f0_method = f0_method
        self.auto_predict_f0 = auto_predict_f0
        self.noise_scale = noise_scale
        self.db_thresh = db_thresh
        self.pad_seconds = pad_seconds
        self.chunk_seconds = chunk_seconds

    def run(self):
        try:
            runner = InferenceRunner()
            output = runner.run(
                source_wav=self.source_wav,
                model_path=self.model_path,
                config_path=self.config_path,
                output_dir=self.output_dir,
                speaker=self.speaker,
                transpose=self.transpose,
                f0_method=self.f0_method,
                auto_predict_f0=self.auto_predict_f0,
                noise_scale=self.noise_scale,
                db_thresh=self.db_thresh,
                pad_seconds=self.pad_seconds,
                chunk_seconds=self.chunk_seconds,
                on_log=self.log_line.emit,
            )
            self.finished_ok.emit(output)
        except Exception as e:
            self.error.emit(str(e))
