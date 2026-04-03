"""Main application window with sidebar navigation."""

import os

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from ui.pages.dataset_page import DatasetPage
from ui.pages.inference_page import InferencePage
from ui.pages.settings_page import SettingsPage
from ui.pages.training_page import TrainingPage
from services.job_store import load_config, get_active_jobs, update_job

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(APP_DIR, "data", "models")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SVC Voice Converter")
        self.setMinimumSize(900, 650)
        self.resize(1050, 720)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
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
            ("Settings", "Configure API keys"),
            ("Dataset", "Add voice samples"),
            ("Training", "Train on RunPod"),
            ("Inference", "Convert voices"),
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

        self.stack.addWidget(self.settings_page)
        self.stack.addWidget(self.dataset_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.inference_page)

        # Navigation
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(0)

        # Clean up orphaned pods on startup
        self._cleanup_orphaned_pods()

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
                if pod_name == "svc-gui-training" and pod_id not in active_pod_ids:
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

    def closeEvent(self, event):
        self.training_page.cleanup()
        super().closeEvent(event)
