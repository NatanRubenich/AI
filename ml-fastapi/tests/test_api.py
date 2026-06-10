"""End-to-end API tests."""

from __future__ import annotations

import pytest

ALGORITHMS = [
    "logistic_regression",
    "knn",
    "decision_tree",
    "naive_bayes",
    "random_forest",
    "gradient_boosting",
    "svm",
]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_list_algorithms(client):
    response = client.get("/algorithms")
    assert response.status_code == 200
    keys = [a["key"] for a in response.json()]
    assert set(keys) == set(ALGORITHMS)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_train_and_predict(client, algo):
    # Train
    r = client.post("/train", json={"algorithm": algo})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["algorithm"]["key"] == algo
    assert body["evaluation"]["accuracy"] > 0.6  # all sklearn algos > 0.6 on iris

    # Predict (Iris setosa-ish sample)
    r = client.post(
        "/predict",
        json={"algorithm": algo, "instances": [[5.1, 3.5, 1.4, 0.2]]},
    )
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert len(preds) == 1
    assert preds[0]["predicted_class"] in {"setosa", "versicolor", "virginica"}


def test_unknown_algorithm(client):
    r = client.post("/train", json={"algorithm": "does_not_exist"})
    assert r.status_code == 404


def test_predict_without_training(client):
    r = client.post(
        "/predict",
        json={"algorithm": "svm", "instances": [[5.1, 3.5, 1.4, 0.2]]},
    )
    assert r.status_code == 409


def test_invalid_feature_count(client):
    client.post("/train", json={"algorithm": "logistic_regression"})
    r = client.post(
        "/predict",
        json={"algorithm": "logistic_regression", "instances": [[1.0, 2.0]]},
    )
    assert r.status_code == 422
