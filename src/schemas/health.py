"""Health check schemas.

This module defines Pydantic models for health check endpoints.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for health check endpoints.

    Attributes:
        status: Health status (healthy/unhealthy)
    """

    status: str = Field(..., description="Health status", examples=["healthy"])


class ReadinessResponse(BaseModel):
    """Response model for readiness check endpoint.

    Attributes:
        ready: Whether the service is ready to accept requests
        model_loaded: Whether the ML model is loaded
    """

    ready: bool = Field(..., description="Service readiness status")
    model_loaded: bool = Field(..., description="ML model loaded status")
