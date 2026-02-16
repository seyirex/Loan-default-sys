"""Application state management.

This module manages global application state including the model service
and model loading status. Centralizing state here provides a single source
of truth and cleaner separation of concerns.
"""

from typing import Optional

# Global application state
_model_service = None
_model_loaded = False


def set_model_service(service) -> None:
    """Set the global model service instance.

    Args:
        service: The ModelService instance to set
    """
    global _model_service
    _model_service = service


def get_model_service():
    """Get the global model service instance.

    Returns:
        ModelService instance

    Raises:
        RuntimeError: If model service is not initialized
    """
    if _model_service is None:
        raise RuntimeError("Model service not initialized")
    return _model_service


def set_model_loaded(status: bool) -> None:
    """Set the model loaded status.

    Args:
        status: True if model is loaded, False otherwise
    """
    global _model_loaded
    _model_loaded = status


def is_model_loaded() -> bool:
    """Check if the ML model is loaded.

    Returns:
        True if model is loaded, False otherwise
    """
    return _model_loaded
