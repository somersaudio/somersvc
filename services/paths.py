"""Centralized path definitions for SomerSVC."""

import os
from pathlib import Path

# All user data lives under ~/.somersvc
USER_DIR = Path.home() / ".somersvc"

# Subdirectories
MODELS_DIR = str(USER_DIR / "models")
DATASETS_DIR = str(USER_DIR / "datasets")
OUTPUT_DIR = str(USER_DIR / "output")
CONFIG_FILE = USER_DIR / "config.json"
JOBS_FILE = USER_DIR / "jobs.json"
CACHE_DIR = str(USER_DIR / "cache")

# App directory (where the code lives — for bundled assets only)
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dirs():
    """Create all required directories."""
    for d in [USER_DIR, MODELS_DIR, DATASETS_DIR, OUTPUT_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)
