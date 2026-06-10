"""
Training orchestrator.

Pure functions that take an algorithm key and produce a fitted pipeline
plus an evaluation report. No persistence concerns here.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.ml.algorithms import AlgorithmSpec, get_algorithm
from app.ml.dataset import Dataset, load_dataset
from app.ml.evaluator import EvaluationReport, evaluate


@dataclass(frozen=True)
class TrainingResult:
    """Output of a training run."""

    spec: AlgorithmSpec
    pipeline: Pipeline
    report: EvaluationReport
    feature_names: list[str]
    target_names: list[str]
    n_train: int
    n_test: int


def train(
    algorithm_key: str,
    test_size: float | None = None,
    random_state: int | None = None,
    dataset: Dataset | None = None,
) -> TrainingResult:
    """Train `algorithm_key` on the configured dataset and evaluate it."""
    spec = get_algorithm(algorithm_key)
    ds = dataset or load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        ds.X,
        ds.y,
        test_size=test_size or settings.default_test_size,
        random_state=random_state or settings.default_random_state,
        stratify=ds.y,
    )

    pipeline = spec.factory()
    pipeline.fit(X_train, y_train)

    report = evaluate(
        pipeline,
        X_test,
        y_test,
        target_names=ds.target_names,
        X_full=ds.X,
        y_full=ds.y,
    )

    return TrainingResult(
        spec=spec,
        pipeline=pipeline,
        report=report,
        feature_names=ds.feature_names,
        target_names=ds.target_names,
        n_train=len(X_train),
        n_test=len(X_test),
    )
