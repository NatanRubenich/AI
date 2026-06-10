"""Tests for the visualization endpoints."""

from __future__ import annotations

import pytest

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_png(response) -> None:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == PNG_MAGIC
    assert len(response.content) > 500  # something meaningful was rendered


def test_dataset_scatter_matrix(client):
    _assert_png(client.get("/plots/dataset/scatter-matrix.png"))


def test_comparison(client):
    _assert_png(client.get("/plots/comparison.png"))


@pytest.mark.parametrize(
    "algo",
    ["logistic_regression", "random_forest", "svm"],
)
def test_confusion_matrix(client, algo):
    client.post("/train", json={"algorithm": algo})
    _assert_png(client.get(f"/plots/{algo}/confusion-matrix.png"))


def test_confusion_matrix_without_training(client):
    r = client.get("/plots/random_forest/confusion-matrix.png")
    assert r.status_code == 409


@pytest.mark.parametrize(
    "algo", ["decision_tree", "random_forest", "gradient_boosting"]
)
def test_feature_importance_tree_based(client, algo):
    client.post("/train", json={"algorithm": algo})
    _assert_png(client.get(f"/plots/{algo}/feature-importance.png"))


def test_feature_importance_unsupported(client):
    client.post("/train", json={"algorithm": "knn"})
    r = client.get("/plots/knn/feature-importance.png")
    assert r.status_code == 422


@pytest.mark.parametrize("algo", ["logistic_regression", "random_forest", "svm"])
def test_roc_curve(client, algo):
    client.post("/train", json={"algorithm": algo})
    _assert_png(client.get(f"/plots/{algo}/roc-curve.png"))


def test_plot_unknown_algorithm(client):
    r = client.get("/plots/does_not_exist/confusion-matrix.png")
    assert r.status_code == 404
