"""
Response utilities for consistent API responses.
"""

from typing import Any, Dict, Optional
from fastapi.responses import JSONResponse
from src.schemas.prediction import StandardResponse





def generate_response(
    success: bool = True,
    data: Optional[Any] = None,
    error: Optional[str] = None,
    exception_error: Optional[str] = None,
    message: Optional[str] = None,
    status_code: int = 200
) -> JSONResponse:
    """
    Generate a consistent JSON response for all API endpoints.

    Args:
        success: Whether the operation was successful
        data: The data payload to return
        error: General error message (e.g., "bad_request", "unauthorized")
        exception_error: Specific exception error message (sanitized)
        message: Additional message or success message
        status_code: HTTP status code

    Returns:
        JSONResponse with standardized envelope
    """
    response_data = StandardResponse(
        success=success,
        data=data,
        error=error,
        exceptionError=exception_error,
        message=message
    )

    return JSONResponse(
        content=response_data.model_dump(exclude_none=True),
        status_code=status_code
    )