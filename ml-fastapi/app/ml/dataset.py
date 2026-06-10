"""
Dataset loader.

Encapsulates dataset acquisition so the rest of the application is
agnostic to its source. Today: Iris from sklearn. Tomorrow: anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris


@dataclass(frozen=True)
class Dataset:
    """Container for a tabular dataset."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def load_dataset() -> Dataset:
    """Load the Iris dataset bundled with scikit-learn."""
    raw = load_iris()
    return Dataset(
        X=raw.data,
        y=raw.target,
        feature_names=list(raw.feature_names),
        target_names=list(raw.target_names),
    )
