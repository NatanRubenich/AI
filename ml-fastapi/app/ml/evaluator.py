"""
Model evaluation utilities.

Returns a structured evaluation report so the API can serialize it
without coupling to sklearn types.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated metrics for a trained classifier."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    cv_mean_accuracy: float
    cv_std_accuracy: float
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float]]


def evaluate(
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
    X_full: np.ndarray | None = None,
    y_full: np.ndarray | None = None,
    cv_folds: int = 5,
) -> EvaluationReport:
    """Compute metrics on the test set plus k-fold CV on the full dataset."""
    y_pred = estimator.predict(X_test)

    cv_mean = cv_std = 0.0
    if X_full is not None and y_full is not None:
        scores = cross_val_score(
            estimator, X_full, y_full, cv=cv_folds, scoring="accuracy"
        )
        cv_mean = float(scores.mean())
        cv_std = float(scores.std())

    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(report_dict[name]["precision"]),
            "recall": float(report_dict[name]["recall"]),
            "f1": float(report_dict[name]["f1-score"]),
            "support": float(report_dict[name]["support"]),
        }
        for name in target_names
    }

    return EvaluationReport(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision_macro=float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        recall_macro=float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        cv_mean_accuracy=cv_mean,
        cv_std_accuracy=cv_std,
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
        per_class=per_class,
    )
