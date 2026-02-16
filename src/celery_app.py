"""Celery application configuration.

This module initializes and configures the Celery app for asynchronous task processing.
This is a shared instance used across the application.
"""

from celery import Celery
from loguru import logger

from src.config import settings

# Initialize Celery app
celery_app = Celery(
    "loan_default_batch",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,  # 50 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)

logger.info("Celery app initialized")
