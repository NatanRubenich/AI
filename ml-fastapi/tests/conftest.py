"""Test fixtures."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def tmp_artifacts(tmp_path_factory) -> Iterator[Path]:
    """Override the artifacts directory so tests do not pollute the repo.

    Function-scoped: each test gets a fresh empty directory, ensuring
    full isolation (no leakage of previously trained models).
    """
    path = tmp_path_factory.mktemp("artifacts")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def client(tmp_artifacts, monkeypatch) -> Iterator[TestClient]:
    """Provide a TestClient with isolated artifacts dir."""
    # Patch settings and repository singleton before importing app.
    from app.core import config as config_module

    monkeypatch.setattr(config_module.settings, "artifacts_dir", tmp_artifacts)

    from app.repositories import model_repository as mr_module

    mr_module.model_repository = mr_module.ModelRepository(base_dir=tmp_artifacts)

    # Reset cached services so they pick up the new repository.
    from app.controllers import dependencies as deps
    from app.controllers import plot_controller

    deps.get_ml_service.cache_clear()
    plot_controller.get_plot_service.cache_clear()

    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
