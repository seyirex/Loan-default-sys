"""Main FastAPI application entry point.

This module initializes the FastAPI application with all middleware, routers, and lifecycle events.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.dependencies import limiter
from src.api.v1 import health
from src.config import settings
from src.core.app_state import set_model_loaded, set_model_service
from src.utils.response import generate_response

from src.api.v1 import predict, model as model_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events.

    Handles startup (model loading) and shutdown (cleanup) events.

    Args:
        app: FastAPI application instance

    Yields:
        Control to the application
    """
    # Startup
    logger.info("=" * 80)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info("=" * 80)

    # Import here to avoid circular imports
    from src.services.model_service import ModelService

    try:
        logger.info("Loading ML model from MLflow...")
        model_service = ModelService(
            mlflow_uri=settings.mlflow_tracking_uri,
            model_name=settings.model_name,
            model_stage=settings.model_stage,
        )
        model_service.load_model()
        set_model_service(model_service)
        set_model_loaded(True)
        logger.info("Model loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        set_model_loaded(False)
        logger.warning("Service starting without model - /readyz will report not ready")

    logger.info(f"API server ready at http://{settings.api_host}:{settings.api_port}")
    logger.info(f"API documentation available at http://{settings.api_host}:{settings.api_port}/docs")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    logger.info("Cleanup complete!")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Production-ready ML inference API for loan default prediction with MLOps best practices",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors.

    Args:
        request: FastAPI request
        exc: Exception that was raised

    Returns:
        JSON error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return generate_response(
        success=False,
        error="internal_server_error",
        exception_error=type(exc).__name__,
        message="Internal server error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )


# Configure Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)

# Include routers
app.include_router(health.router, prefix="", tags=["Health"])

try:
    app.include_router(predict.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(model_router.router, prefix="/api/v1", tags=["Model"])
except ImportError:
    logger.warning("Prediction and model routers not yet available")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs.

    Returns:
        Redirect response to API documentation
    """
    return generate_response(
        success=True,
        data={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/healthz",
            "metrics": "/metrics",
        },
        message=f"Welcome to {settings.app_name}",
        status_code=200
    )


