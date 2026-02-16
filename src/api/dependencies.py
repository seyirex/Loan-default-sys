"""API dependencies for authentication and rate limiting.

This module provides dependency injection functions for FastAPI endpoints.
"""

from fastapi import Header, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import settings


def verify_api_key(x_api_key: str = Header(..., description="API Key for authentication")) -> str:
    """Verify API key authentication.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The verified API key

    Raises:
        HTTPException: If API key is invalid
    """
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return x_api_key


# Create rate limiter instance
limiter = Limiter(key_func=get_remote_address)
