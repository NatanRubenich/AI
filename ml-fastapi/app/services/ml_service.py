"""
ML application service.

Orchestrates the ML domain (`app.ml`) and persistence (`repositories`)
on behalf of controllers. Controllers only see this service.
"""

from __future__ import annotations

import numpy as np

from app.core.exceptions import InvalidInputError
from app.core.logging import get_logger
from app.ml.algorithms import AlgorithmSpec, get_algorithm, list_algorithms
from app.ml.trainer import TrainingResult, train
from app.repositories.model_repository import ModelRepository, model_repository

logger = get_logger(__name__)


class MLService:
    """Application service exposing ML use-cases."""

    def __init__(self, repository: ModelRepository | None = None) -> None:
        self._repo = repository or model_repository

    # --- Catalogue ------------------------------------------------------

    @staticmethod
    def list_algorithms() -> list[AlgorithmSpec]:
        return list_algorithms()

    # --- Training -------------------------------------------------------

    def train_algorithm(
        self,
        algorithm_key: str,
        test_size: float,
        random_state: int,
    ) -> tuple[TrainingResult, str]:
        """Train and persist a model. Returns (result, model_id)."""
        logger.info("Training algorithm=%s", algorithm_key)
        result = train(
            algorithm_key=algorithm_key,
            test_size=test_size,
            random_state=random_state,
        )
        model_id = self._repo.save(
            model_id=algorithm_key,
            pipeline=result.pipeline,
            feature_names=result.feature_names,
            target_names=result.target_names,
        )
        logger.info(
            "Trained algorithm=%s accuracy=%.4f cv=%.4f±%.4f",
            algorithm_key,
            result.report.accuracy,
            result.report.cv_mean_accuracy,
            result.report.cv_std_accuracy,
        )
        return result, model_id

    # --- Prediction -----------------------------------------------------

    def predict(
        self, algorithm_key: str, instances: list[list[float]]
    ) -> list[dict]:
        """Run inference on a batch of instances."""
        # Validate algorithm is known.
        get_algorithm(algorithm_key)
        stored = self._repo.load(algorithm_key)

        X = np.asarray(instances, dtype=float)
        expected = len(stored.feature_names)
        if X.shape[1] != expected:
            raise InvalidInputError(
                f"Each instance must have {expected} features "
                f"({stored.feature_names}); got {X.shape[1]}."
            )

        y_pred = stored.pipeline.predict(X)

        probas: np.ndarray | None = None
        if hasattr(stored.pipeline, "predict_proba"):
            try:
                probas = stored.pipeline.predict_proba(X)
            except (AttributeError, NotImplementedError):
                probas = None

        results: list[dict] = []
        for i, idx in enumerate(y_pred):
            idx_int = int(idx)
            entry: dict = {
                "predicted_index": idx_int,
                "predicted_class": stored.target_names[idx_int],
                "probabilities": None,
            }
            if probas is not None:
                entry["probabilities"] = {
                    stored.target_names[j]: float(probas[i, j])
                    for j in range(probas.shape[1])
                }
            results.append(entry)
        return results
