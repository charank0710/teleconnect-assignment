"""Shared helper utilities for the TeleConnect ML assignment."""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_random_seed(random_state: int = 42) -> None:
    """Seed Python, NumPy, and hash-based randomness."""

    random.seed(random_state)
    np.random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not exist and return it."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_pickle(obj, path: Path | str) -> None:
    """Serialize an object to disk using pickle."""

    file_path = Path(path)
    ensure_directory(file_path.parent)
    with file_path.open("wb") as file_handle:
        pickle.dump(obj, file_handle)


def load_pickle(path: Path | str):
    """Load a pickled object from disk."""

    file_path = Path(path)
    with file_path.open("rb") as file_handle:
        return pickle.load(file_handle)


def adjusted_r2_score(r2: float, n_samples: int, n_features: int) -> float:
    """Compute adjusted R-squared for regression models."""

    if n_samples <= n_features + 1:
        return float("nan")
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

