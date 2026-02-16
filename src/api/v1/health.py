"""Health check endpoints.

This module provides liveness and readiness probe endpoints for Kubernetes.
"""

from fastapi import APIRouter
from loguru import logger

from src.core.app_state import is_model_loaded
from src.schemas.health import HealthResponse, ReadinessResponse
from src.utils.response import generate_response

router = APIRouter()


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Check if the service is alive and running",
)
async def liveness_check():
    """Liveness probe for Kubernetes.

    Returns:
        Standardized response with health status
    """
    logger.debug("Liveness check called")
    health_data = HealthResponse(status="healthy")
    return generate_response(
        success=True,
        data=health_data.model_dump(),
        message="Service is alive",
        status_code=200
    )


@router.get(
    "/readyz",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Check if the service is ready to accept requests",
)
async def readiness_check():
    """Readiness probe for Kubernetes.

    Checks if the ML model is loaded and service is ready.

    Returns:
        Standardized response with readiness status and model load state
    """
    ready = is_model_loaded()
    message = "Service is ready" if ready else "Service is not ready - model not loaded"

    logger.debug(f"Readiness check called - Model loaded: {ready}")

    readiness_data = ReadinessResponse(ready=ready, model_loaded=ready)
    status_code = 200 if ready else 503

    return generate_response(
        success=ready,
        data=readiness_data.model_dump(),
        message=message,
        status_code=status_code
    )
