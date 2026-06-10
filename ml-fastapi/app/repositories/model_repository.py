"""
Model persistence repository.

Encapsulates how trained pipelines are stored and retrieved. Currently
uses joblib on the local filesystem; could be swapped for S3, MLflow,
etc. without touching the rest of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.core.exceptions import ModelNotTrainedError


@dataclass(frozen=True)
class StoredModel:
    """Container for a persisted model and its metadata."""

    pipeline: Pipeline
    feature_names: list[str]
    target_names: list[str]


class ModelRepository:
    """Filesystem-backed model repository."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or settings.artifacts_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # --- Paths -----------------------------------------------------------

    def _path(self, model_id: str) -> Path:
        # `model_id` is constrained by the algorithm registry, so no
        # path-traversal risk, but we still strip separators defensively.
        safe = model_id.replace("/", "_").replace("\\", "_")
        return self._base_dir / f"{safe}.joblib"

    # --- Public API ------------------------------------------------------

    def save(
        self,
        model_id: str,
        pipeline: Pipeline,
        feature_names: list[str],
        target_names: list[str],
    ) -> str:
        """Persist `pipeline` under `model_id` and return its identifier."""
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_names": feature_names,
                "target_names": target_names,
            },
            self._path(model_id),
        )
        return model_id

    def load(self, model_id: str) -> StoredModel:
        """Load a previously saved model or raise `ModelNotTrainedError`."""
        path = self._path(model_id)
        if not path.exists():
            raise ModelNotTrainedError(
                f"No trained model found for '{model_id}'. "
                "Call POST /train first."
            )
        blob = joblib.load(path)
        return StoredModel(
            pipeline=blob["pipeline"],
            feature_names=blob["feature_names"],
            target_names=blob["target_names"],
        )

    def exists(self, model_id: str) -> bool:
        return self._path(model_id).exists()

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self._base_dir.glob("*.joblib"))


# Singleton used by the service layer.
model_repository = ModelRepository()
