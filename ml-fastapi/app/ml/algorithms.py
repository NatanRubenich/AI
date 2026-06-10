"""
Algorithm registry.

Centralizes the catalogue of supported scikit-learn estimators. Each entry
exposes a factory that returns a fresh `Pipeline` (scaler + estimator),
keeping the rest of the codebase decoupled from sklearn internals.

Why a registry?
- Adding a new algorithm is a single-line change.
- Controllers/services never import sklearn directly.
- The same identifiers are used in API requests and persistence layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.core.config import settings
from app.core.exceptions import AlgorithmNotFoundError


@dataclass(frozen=True)
class AlgorithmSpec:
    """Metadata describing a supported algorithm."""

    key: str
    display_name: str
    category: str  # "classic" or "robust"
    description: str
    factory: Callable[[], Pipeline]


def _pipeline(estimator) -> Pipeline:
    """Wrap an estimator in a standard scaling pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ]
    )


_RANDOM_STATE = settings.default_random_state

# --- Registry ----------------------------------------------------------------
_REGISTRY: dict[str, AlgorithmSpec] = {
    # Classic algorithms
    "logistic_regression": AlgorithmSpec(
        key="logistic_regression",
        display_name="Logistic Regression",
        category="classic",
        description="Linear probabilistic classifier; strong baseline.",
        factory=lambda: _pipeline(
            LogisticRegression(max_iter=1000, random_state=_RANDOM_STATE)
        ),
    ),
    "knn": AlgorithmSpec(
        key="knn",
        display_name="K-Nearest Neighbors",
        category="classic",
        description="Instance-based learner using distance to neighbors.",
        factory=lambda: _pipeline(KNeighborsClassifier(n_neighbors=5)),
    ),
    "decision_tree": AlgorithmSpec(
        key="decision_tree",
        display_name="Decision Tree",
        category="classic",
        description="Interpretable tree of axis-aligned splits.",
        factory=lambda: _pipeline(
            DecisionTreeClassifier(random_state=_RANDOM_STATE)
        ),
    ),
    "naive_bayes": AlgorithmSpec(
        key="naive_bayes",
        display_name="Gaussian Naive Bayes",
        category="classic",
        description="Probabilistic classifier with Gaussian feature assumption.",
        factory=lambda: _pipeline(GaussianNB()),
    ),
    # Robust algorithms
    "random_forest": AlgorithmSpec(
        key="random_forest",
        display_name="Random Forest",
        category="robust",
        description="Ensemble of decorrelated decision trees (bagging).",
        factory=lambda: _pipeline(
            RandomForestClassifier(
                n_estimators=200, random_state=_RANDOM_STATE, n_jobs=-1
            )
        ),
    ),
    "gradient_boosting": AlgorithmSpec(
        key="gradient_boosting",
        display_name="Gradient Boosting",
        category="robust",
        description="Sequential boosting of weak learners minimizing residuals.",
        factory=lambda: _pipeline(
            GradientBoostingClassifier(random_state=_RANDOM_STATE)
        ),
    ),
    "svm": AlgorithmSpec(
        key="svm",
        display_name="Support Vector Machine (RBF)",
        category="robust",
        description="Maximum-margin classifier with RBF kernel; probabilities "
        "via Platt-style calibration (CalibratedClassifierCV).",
        factory=lambda: _pipeline(
            CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=_RANDOM_STATE),
                ensemble=False,
            )
        ),
    ),
}


def list_algorithms() -> list[AlgorithmSpec]:
    """Return all registered algorithm specs."""
    return list(_REGISTRY.values())


def get_algorithm(key: str) -> AlgorithmSpec:
    """Return the spec for `key` or raise `AlgorithmNotFoundError`."""
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise AlgorithmNotFoundError(
            f"Unknown algorithm '{key}'. "
            f"Available: {sorted(_REGISTRY)}"
        ) from exc


def algorithm_keys() -> list[str]:
    """Return the list of valid algorithm keys."""
    return list(_REGISTRY.keys())
