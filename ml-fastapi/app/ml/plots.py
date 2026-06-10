"""
Plot primitives.

Pure functions that render matplotlib figures to PNG bytes. They have
**no knowledge** of FastAPI, services, or persistence — they only know
how to draw given numeric inputs.

The `Agg` backend is selected explicitly so the API works on headless
servers (no display required).
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must come after backend select)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import auc, roc_curve  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402


# --- Helpers -----------------------------------------------------------------


def _fig_to_png(fig) -> bytes:
    """Serialize a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# --- Plots -------------------------------------------------------------------


def plot_confusion_matrix(
    cm: list[list[int]], target_names: list[str], title: str
) -> bytes:
    """Heatmap of a confusion matrix with numeric annotations."""
    matrix = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=30, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(title)

    threshold = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    return _fig_to_png(fig)


def plot_feature_importance(
    importances: np.ndarray, feature_names: list[str], title: str
) -> bytes:
    """Horizontal bar chart of feature importances, sorted ascending."""
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(
        np.asarray(feature_names)[order],
        np.asarray(importances)[order],
        color="#4C72B0",
    )
    ax.set_xlabel("Importância")
    ax.set_title(title)
    return _fig_to_png(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_names: list[str],
    title: str,
) -> bytes:
    """One-vs-rest ROC curves for a multiclass problem."""
    classes = list(range(len(target_names)))
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _fig_to_png(fig)


def plot_algorithm_comparison(
    results: dict[str, dict[str, float]], title: str
) -> bytes:
    """Grouped bar chart comparing algorithms by accuracy / F1 / CV mean."""
    algos = list(results.keys())
    metrics = [
        ("accuracy", "Acurácia"),
        ("f1_macro", "F1 (macro)"),
        ("cv_mean_accuracy", "Acurácia média (CV)"),
    ]
    x = np.arange(len(algos))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(algos) * 1.1), 5))
    for i, (key, label) in enumerate(metrics):
        values = [results[a][key] for a in algos]
        ax.bar(x + (i - 1) * width, values, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Pontuação")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    return _fig_to_png(fig)


def plot_scatter_matrix(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    title: str,
) -> bytes:
    """Pandas scatter-matrix colored by class — quick dataset overview."""
    df = pd.DataFrame(X, columns=feature_names)
    colors = np.array(["#E24A33", "#348ABD", "#988ED5"])
    point_colors = colors[y % len(colors)]

    axes = pd.plotting.scatter_matrix(
        df,
        figsize=(8, 8),
        diagonal="hist",
        color=point_colors,
        alpha=0.7,
        s=18,
    )
    fig = axes[0, 0].get_figure()
    fig.suptitle(title, y=0.92)

    # Build a manual legend (scatter_matrix does not provide one).
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=colors[i], label=name
        )
        for i, name in enumerate(target_names)
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    return _fig_to_png(fig)
