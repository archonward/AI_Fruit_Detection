"""Reusable project paths for Fruit Quality AI."""

from pathlib import Path


# src/config.py lives inside the src folder, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
SPLITS_DATA_DIR = DATASET_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
