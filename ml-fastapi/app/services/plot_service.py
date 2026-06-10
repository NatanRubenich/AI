"""
Plot application service.

Bridges the plot primitives (`app.ml.plots`) with the rest of the
application: loads trained models, re-creates reproducible test splits,
extracts importances/probabilities, and returns PNG bytes ready to be
served by FastAPI.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.core.exceptions import InvalidInputError
from app.ml import plots
from app.ml.algorithms import get_algorithm
from app.ml.dataset import load_dataset
from app.ml.trainer import train
from app.repositories.model_repository import ModelRepository, model_repository
from app.services.ml_service import MLService


class PlotService:
    """Generates PNG visualizations for datasets and trained models."""

    def __init__(
        self,
        repository: ModelRepository | None = None,
        ml_service: MLService | None = None,
    ) -> None:
        self._repo = repository or model_repository
        self._ml = ml_service or MLService(repository=self._repo)

    # --- Internal helpers ----------------------------------------------

    def _reproduce_test_split(
        self, test_size: float, random_state: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the same X_test/y_test used during training (same seed)."""
        ds = load_dataset()
        _, X_test, _, y_test = train_test_split(
            ds.X,
            ds.y,
            test_size=test_size,
            random_state=random_state,
            stratify=ds.y,
        )
        return X_test, y_test

    # --- Dataset plot --------------------------------------------------

    def scatter_matrix_png(self) -> bytes:
        ds = load_dataset()
        return plots.plot_scatter_matrix(
            ds.X, ds.y, ds.feature_names, ds.target_names,
            title="Dataset Iris — matriz de dispersão",
        )

    # --- Model plots ---------------------------------------------------

    def confusion_matrix_png(
        self,
        algorithm_key: str,
        test_size: float = settings.default_test_size,
        random_state: int = settings.default_random_state,
    ) -> bytes:
        spec = get_algorithm(algorithm_key)
        stored = self._repo.load(algorithm_key)
        X_test, y_test = self._reproduce_test_split(test_size, random_state)
        y_pred = stored.pipeline.predict(X_test)
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred).tolist()
        return plots.plot_confusion_matrix(
            cm, stored.target_names,
            title=f"Matriz de confusão — {spec.display_name}",
        )

    def feature_importance_png(self, algorithm_key: str) -> bytes:
        spec = get_algorithm(algorithm_key)
        stored = self._repo.load(algorithm_key)
        estimator = stored.pipeline.named_steps["estimator"]
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            raise InvalidInputError(
                f"Algorithm '{algorithm_key}' does not expose "
                "feature_importances_. Try a tree-based model "
                "(decision_tree, random_forest, gradient_boosting)."
            )
        return plots.plot_feature_importance(
            importances,
            stored.feature_names,
            title=f"Importância das features — {spec.display_name}",
        )

    def roc_curve_png(
        self,
        algorithm_key: str,
        test_size: float = settings.default_test_size,
        random_state: int = settings.default_random_state,
    ) -> bytes:
        spec = get_algorithm(algorithm_key)
        stored = self._repo.load(algorithm_key)
        if not hasattr(stored.pipeline, "predict_proba"):
            raise InvalidInputError(
                f"Algorithm '{algorithm_key}' does not support "
                "predict_proba; ROC curves are unavailable."
            )
        X_test, y_test = self._reproduce_test_split(test_size, random_state)
        y_score = stored.pipeline.predict_proba(X_test)
        return plots.plot_roc_curves(
            y_test, y_score, stored.target_names,
            title=f"Curvas ROC (Um-vs-Resto) — {spec.display_name}",
        )

    def comparison_png(self) -> bytes:
        """Train every registered algorithm and chart their metrics."""
        results: dict[str, dict[str, float]] = {}
        for spec in self._ml.list_algorithms():
            result = train(spec.key)
            results[spec.key] = {
                "accuracy": result.report.accuracy,
                "f1_macro": result.report.f1_macro,
                "cv_mean_accuracy": result.report.cv_mean_accuracy,
            }
        return plots.plot_algorithm_comparison(
            results,
            title="Comparação de algoritmos no Iris (teste + validação cruzada 5-fold)",
        )
