"""Domain-level exceptions used across the application."""

from __future__ import annotations


class MLAppError(Exception):
    """Base class for all application-level errors."""


class AlgorithmNotFoundError(MLAppError):
    """Raised when a requested algorithm key is not registered."""


class ModelNotTrainedError(MLAppError):
    """Raised when prediction is attempted before training a model."""


class InvalidInputError(MLAppError):
    """Raised when prediction input does not match the expected schema."""
