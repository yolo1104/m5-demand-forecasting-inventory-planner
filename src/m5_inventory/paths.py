"""Shared project paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
