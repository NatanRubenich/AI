"""FastAPI dependency providers."""

from __future__ import annotations

from functools import lru_cache

from app.services.ml_service import MLService


@lru_cache(maxsize=1)
def get_ml_service() -> MLService:
    """Singleton MLService used by route handlers."""
    return MLService()
