"""Main application window with sidebar navigation."""

import os

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.pages.dataset_page import DatasetPage
from ui.pages.inference_page import InferencePage
from ui.pages.models_page import ModelsPage
from ui.pages.realtime_page import RealtimePage
from ui.pages.settings_page import SettingsPage
from ui.pages.simple_page import SimplePage
from ui.pages.training_page import TrainingPage
from services.job_store import load_config, get_active_jobs, update_job

from services.paths import MODELS_DIR


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setMinimumSize(900, 650)
        self.resize(1050, 820)
        self._drag_pos = None

        # Window control buttons (macOS traffic light style)
        btn_style = """
            QPushButton {{
                background: {bg};
                border: none;
                border-radius: 6px;
                padding: 0px;
            }}
            QPushButton:hover {{ background: {hover}; }}
        """
        self._btn_close = QPushButton(self)
        self._btn_close.setFixedSize(12, 12)
        self._btn_close.setStyleSheet(btn_style.format(bg="#ff5f57", hover="#ff3b30"))
        self._btn_close.clicked.connect(self.close)
        self._btn_close.setCursor(Qt.CursorShape.PointingHandCursor)

        self._btn_minimize = QPushButton(self)
        self._btn_minimize.setFixedSize(12, 12)
        self._btn_minimize.setStyleSheet(btn_style.format(bg="#febc2e", hover="#f5a623"))
        self._btn_minimize.clicked.connect(self.showMinimized)
        self._btn_minimize.setCursor(Qt.CursorShape.PointingHandCursor)

        # Top-level stack: simple mode vs expert mode
        self._mode_stack = QStackedWidget()
        self.setCentralWidget(self._mode_stack)
        self._is_expert = False

        # ===== SIMPLE MODE =====
        self._simple_container = QWidget()
        simple_layout = QVBoxLayout(self._simple_container)
        simple_layout.setContentsMargins(0, 0, 0, 0)
        simple_layout.setSpacing(0)

        self.simple_page = SimplePage()
        simple_layout.addWidget(self.simple_page, 1)

        # Gear icon (bottom-right corner)
        self._gear_btn_simple = QPushButton("⚙")
        self._gear_btn_simple.setFixedSize(36, 36)
        self._gear_btn_simple.setCursor(Qt.CursorShape.PointingHandCursor)
        self._gear_btn_simple.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,5);
                border: 1px solid rgba(255,255,255,10);
                border-radius: 18px;
                color: #666;
                font-size: 18px;
            }
            QPushButton:hover { background: rgba(255,255,255,12); color: #aaa; }
        """)
        self._gear_btn_simple.setToolTip("Switch to Expert Mode")
        self._gear_btn_simple.clicked.connect(self._toggle_mode)
        # Position it in bottom-right — we'll update in resizeEvent
        self._gear_btn_simple.setParent(self._simple_container)
        self._gear_btn_simple.raise_()

        self._mode_stack.addWidget(self._simple_container)

        # ===== EXPERT MODE =====
        self._expert_container = QWidget()
        main_layout = QHBoxLayout(self._expert_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(180)
        self.sidebar.setStyleSheet(
            """
            QListWidget {
                background-color: #161616;
                border: none;
                border-right: 1px solid #333;
                padding: 8px 0;
            }
            QListWidget::item {
                padding: 14px 20px;
                font-size: 14px;
                border-radius: 0;
            }
            QListWidget::item:selected {
                background-color: #2563eb;
                color: white;
                font-weight: bold;
            }
            QListWidget::item:hover:!selected {
                background-color: #222;
            }
            """
        )

        pages = [
            ("Models", "Your trained models"),
            ("Create", "Create a new model"),
            ("Training", "Train on RunPod"),
            ("Inference", "Convert audio files"),
            ("Realtime", "Live mic conversion"),
            ("Settings", "Configure API keys"),
        ]

        for name, tooltip in pages:
            item = QListWidgetItem(name)
            item.setToolTip(tooltip)
            item.setSizeHint(QSize(180, 48))
            self.sidebar.addItem(item)

        main_layout.addWidget(self.sidebar)

        # Page stack
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 1)

        # Create pages
        self.models_page = ModelsPage()
        self.settings_page = SettingsPage()
        self.dataset_page = DatasetPage()
        self.training_page = TrainingPage(
            get_speaker_name=self.dataset_page.get_speaker_name,
            get_dataset_manager=self.dataset_page.get_dataset_manager,
            get_api_key=self.settings_page.get_api_key,
            get_ssh_key_path=self.settings_page.get_ssh_key_path,
            models_dir=MODELS_DIR,
        )
        self.inference_page = InferencePage()
        self.realtime_page = RealtimePage()

        self.stack.addWidget(self.models_page)
        self.stack.addWidget(self.dataset_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.inference_page)
        self.stack.addWidget(self.realtime_page)
        self.stack.addWidget(self.settings_page)

        self._mode_stack.addWidget(self._expert_container)

        # Gear icon for expert mode (bottom-right)
        self._gear_btn_expert = QPushButton("⚙")
        self._gear_btn_expert.setFixedSize(36, 36)
        self._gear_btn_expert.setCursor(Qt.CursorShape.PointingHandCursor)
        self._gear_btn_expert.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,5);
                border: 1px solid rgba(255,255,255,10);
                border-radius: 18px;
                color: #666;
                font-size: 18px;
            }
            QPushButton:hover { background: rgba(255,255,255,12); color: #aaa; }
        """)
        self._gear_btn_expert.setToolTip("Switch to Simple Mode")
        self._gear_btn_expert.clicked.connect(self._toggle_mode)
        self._gear_btn_expert.setParent(self._expert_container)
        self._gear_btn_expert.raise_()

        # Navigation
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(0)

        # Sync "Match Artist's Range" (simple) ↔ "Smart Transpose" (inference)
        self.simple_page._btn_range_match.toggled.connect(
            self.inference_page.chk_smart_transpose.setChecked
        )
        self.inference_page.chk_smart_transpose.toggled.connect(
            self.simple_page._btn_range_match.setChecked
        )

        # When a model finishes downloading, switch to simple mode and select it
        self.models_page.model_downloaded.connect(self._on_model_downloaded)

        # Start in simple mode
        self._mode_stack.setCurrentWidget(self._simple_container)

        # Training status indicator in sidebar
        self._training_anim_timer = None
        self._training_anim_dots = 0
        self.training_page._worker_started = self._on_training_started
        self.training_page._worker_stopped = self._on_training_stopped

        # Connect training signals
        original_start = self.training_page._start_training
        original_finished = self.training_page._on_finished
        original_error = self.training_page._on_error

        def wrapped_start():
            original_start()
            if self.training_page._worker and self.training_page._worker.isRunning():
                self._on_training_started()

        def wrapped_finished(job_id):
            original_finished(job_id)
            self._on_training_stopped()

        def wrapped_error(error):
            original_error(error)
            self._on_training_stopped()

        self.training_page._start_training = wrapped_start
        self.training_page._on_finished = wrapped_finished
        self.training_page._on_error = wrapped_error

        # Clean up orphaned pods on startup
        self._cleanup_orphaned_pods()

        # Check for models that finished training while app was closed
        self._check_pending_downloads()

        # Position gear buttons after layout
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._position_gear_buttons)
        QTimer.singleShot(100, self._position_window_controls)

    def _on_training_started(self):
        from PyQt6.QtCore import QTimer
        self._training_anim_dots = 0
        self._training_anim_timer = QTimer()
        self._training_anim_timer.setInterval(500)
        self._training_anim_timer.timeout.connect(self._animate_training)
        self._training_anim_timer.start()

    def _on_training_stopped(self):
        if self._training_anim_timer:
            self._training_anim_timer.stop()
            self._training_anim_timer = None
        # Reset sidebar text
        self.sidebar.item(2).setText("Training")

    def _animate_training(self):
        self._training_anim_dots = (self._training_anim_dots + 1) % 4
        dots = "." * self._training_anim_dots
        self.sidebar.item(2).setText(f"Training{dots}")

    def _check_pending_downloads(self):
        """Check R2 for models that finished training while the app was closed."""
        try:
            from services.training_orchestrator import TrainingOrchestrator
            recovered = TrainingOrchestrator.check_pending_downloads(
                MODELS_DIR,
                on_log=lambda msg: print(f"[Recovery] {msg}"),
            )
            if recovered:
                from PyQt6.QtWidgets import QMessageBox
                names = ", ".join(recovered)
                QMessageBox.information(
                    self, "Models Recovered",
                    f"The following models finished training while the app was closed "
                    f"and have been downloaded:\n\n{names}",
                )
                self.models_page._refresh_models()
        except Exception as e:
            print(f"Pending download check error (non-fatal): {e}")

    def _cleanup_orphaned_pods(self):
        """Terminate any RunPod pods from failed/stuck jobs on startup."""
        config = load_config()
        api_key = config.get("runpod_api_key", "")
        if not api_key:
            return

        try:
            import runpod
            runpod.api_key = api_key

            # Get all our pods
            pods = runpod.get_pods()
            if not pods:
                return

            # Get active job pod IDs (ones we're actually tracking)
            active_jobs = get_active_jobs()
            active_pod_ids = {j.get("pod_id") for j in active_jobs if j.get("pod_id")}

            # Find orphaned svc-gui pods
            for pod in pods:
                pod_name = pod.get("name", "")
                pod_id = pod.get("id", "")
                if pod_name in ("somersvc-training", "svc-gui-training") and pod_id not in active_pod_ids:
                    print(f"Terminating orphaned pod: {pod_id}")
                    runpod.terminate_pod(pod_id)

            # Also mark any stuck active jobs as failed if their pods are gone
            for job in active_jobs:
                pod_id = job.get("pod_id")
                if pod_id:
                    pod_exists = any(p.get("id") == pod_id for p in pods)
                    if not pod_exists:
                        update_job(job["job_id"], status="failed", error="Pod no longer exists")

        except Exception as e:
            print(f"Orphan cleanup error (non-fatal): {e}")

    def _on_model_downloaded(self, artist):
        """Switch to simple mode and select the newly downloaded model."""
        self._is_expert = False
        self._mode_stack.setCurrentWidget(self._simple_container)
        self._position_gear_buttons()
        self.simple_page.select_model_by_name(artist)

    def _toggle_mode(self):
        """Switch between simple and expert mode."""
        self._is_expert = not self._is_expert
        if self._is_expert:
            self._mode_stack.setCurrentWidget(self._expert_container)
            # Refresh models in expert view
            self.models_page._refresh_models()
            self.inference_page._refresh_models()
        else:
            self._mode_stack.setCurrentWidget(self._simple_container)
            self.simple_page._refresh_models()
        self._position_gear_buttons()

    def _position_gear_buttons(self):
        """Position gear icons in bottom-right corner."""
        margin = 16
        for container, btn in [
            (self._simple_container, self._gear_btn_simple),
            (self._expert_container, self._gear_btn_expert),
        ]:
            w = container.width()
            h = container.height()
            if w > 0 and h > 0:
                btn.move(w - btn.width() - margin, h - btn.height() - margin)
                btn.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Delay positioning to ensure layout is complete
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._position_gear_buttons)
        self._position_window_controls()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def _position_window_controls(self):
        if hasattr(self, '_btn_close'):
            self._btn_close.move(8, 6)
            self._btn_minimize.move(26, 6)
            self._btn_close.raise_()
            self._btn_minimize.raise_()

    def closeEvent(self, event):
        self.simple_page.save_session()
        self.training_page.cleanup()
        super().closeEvent(event)
