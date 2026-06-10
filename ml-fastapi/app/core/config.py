"""
Application configuration.

Uses pydantic-settings to load configuration from environment variables
with sensible defaults. Centralizing config here keeps the rest of the
application free of `os.environ` lookups.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API metadata
    app_name: str = "ML FastAPI - Scikit-learn Showcase"
    app_version: str = "1.0.0"
    app_description: str = (
        "Clean-architecture FastAPI service demonstrating classic and "
        "robust scikit-learn algorithms on the Iris dataset."
    )

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    artifacts_dir: Path = base_dir / "artifacts"

    # ML defaults
    default_test_size: float = 0.2
    default_random_state: int = 42

    # Logging
    log_level: str = "INFO"


settings = Settings()
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
