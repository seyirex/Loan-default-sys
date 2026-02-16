"""Core application components.

This package contains core application logic including state management.
"""

from src.core.app_state import (
    get_model_service,
    set_model_service,
    is_model_loaded,
    set_model_loaded,
)

__all__ = [
    "get_model_service",
    "set_model_service",
    "is_model_loaded",
    "set_model_loaded",
]
