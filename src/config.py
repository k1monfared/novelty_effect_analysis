"""Configuration loading.

The whole framework is driven from a single JSON file (configs/config.json)
so that one fixed seed and one set of thresholds reproduce every number and
figure in the repository.
"""
from __future__ import annotations

import json
import os

# Repository root is the parent of this file's directory (src/..).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG_PATH = os.path.join(ROOT, "configs", "config.json")

DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
IMAGES_DIR = os.path.join(ROOT, "docs", "images")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load the JSON configuration into a plain dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dirs() -> None:
    """Create output directories if they do not yet exist."""
    for d in (DATA_DIR, OUTPUTS_DIR, IMAGES_DIR):
        os.makedirs(d, exist_ok=True)
