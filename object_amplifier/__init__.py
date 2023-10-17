"""
Top-level package global definitions
"""
import os

from pathlib import Path


BASE_PATH = Path(os.path.dirname(__file__))
CONFIG_PATH = BASE_PATH / "config"
DATA_PATH = BASE_PATH / "data"
IMAGE_PATH = DATA_PATH / "images"
MODELS_PATH = DATA_PATH / "saved_models"
TRAIN_IMAGES = DATA_PATH / "background/images"
TRAIN_LABELS = DATA_PATH / "background/labels"
