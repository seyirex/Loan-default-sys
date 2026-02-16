"""Prediction endpoints for realtime and batch predictions.

This module provides API endpoints for making loan default predictions.
"""

import time
from typing import Dict

from celery.result import AsyncResult
from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from loguru import logger

from src.api.dependencies import limiter, verify_api_key
from src.celery_app import celery_app
from src.core.app_state import get_model_service
from src.schemas.prediction import (
    BatchJobResponse,
    BatchJobStatus,
    JobStatusEnum,
    BatchPredictionRequest,
    PredictionRequest,
    PredictionResponse,
)
from src.services.batch_service import batch_predict_task
from src.services.drift_detector import DriftDetector
from src.services.metrics_service import metrics_service
from src.utils.monitoring import get_cached_model_version, record_monitoring_data
from src.utils.response import generate_response

router = APIRouter()

# Global drift detector instance
drift_detector = DriftDetector()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Make a Prediction",
    description="Make a realtime loan default prediction for a single applicant",
    status_code=status.HTTP_200_OK,
)
@limiter.limit("100/minute")
async def predict(
    request: Request,
    prediction_request: PredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Make a realtime loan default prediction.

    Args:
        request: FastAPI request object
        prediction_request: Prediction request with applicant features
        api_key: Validated API key

    Returns:
        PredictionResponse with prediction and probability

    Raises:
        HTTPException: If prediction fails
    """
    start_time = time.time()

    try:
        model_service = get_model_service()

        # Extract features from validated request
        features = {
            "employed": prediction_request.employed,
            "bank_balance": prediction_request.bank_balance,
            "annual_salary": prediction_request.annual_salary,
        }

        # Make prediction
        result = model_service.predict(features)

        model_version = get_cached_model_version()

        # Calculate duration before background tasks
        duration = time.time() - start_time

        # Schedule monitoring in background (runs after response is sent)
        background_tasks.add_task(
            record_monitoring_data,
            drift_detector,
            features,
            model_version,
            result["prediction"],
            duration,
        )

        # Build response
        response_data = PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            default_risk=result["default_risk"],
            model_version=model_version,
            features_used=features,
        )

        logger.info(
            f"Prediction completed - Result: {result['prediction']}, Probability: {result['probability']:.4f}, Duration: {duration:.3f}s"
        )

        return generate_response(
            success=True,
            data=response_data.model_dump(),
            message="Prediction completed successfully",
            status_code=200
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Prediction failed: {e}", exc_info=True)

        # Record failure metric in background
        try:
            model_version = get_cached_model_version()
            background_tasks.add_task(
                metrics_service.record_prediction,
                model_version=model_version,
                prediction_type="realtime",
                status="failure",
                duration=duration,
            )
        except Exception as metric_error:
            logger.warning(f"Failed to record failure metric: {metric_error}")

        return generate_response(
            success=False,
            error="prediction_error",
            exception_error=str(e),
            message="Prediction failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post(
    "/predict/batch",
    response_model=BatchJobResponse,
    summary="Submit Batch Prediction Job",
    description="Submit a batch of predictions to be processed asynchronously",
    status_code=status.HTTP_202_ACCEPTED,
)
async def batch_predict(
    batch_request: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key),
):
    """Submit a batch prediction job for asynchronous processing.

    Args:
        batch_request: Batch prediction request with list of predictions
        api_key: Validated API key

    Returns:
        BatchJobResponse with job ID and status

    Raises:
        HTTPException: If job submission fails
    """
    try:
        # Convert requests to list of dicts
        predictions_data = [pred.model_dump() for pred in batch_request.predictions]

        # Submit to Celery (Celery auto-generates task ID)
        task = batch_predict_task.apply_async(args=[predictions_data])
        job_id = task.id 

        logger.info(
            f"Batch job submitted - ID: {job_id}, Total predictions: {len(predictions_data)}"
        )

        response_data = BatchJobResponse(
            job_id=job_id,
            status=JobStatusEnum.PENDING,
            message="Batch job submitted successfully",
            total_predictions=len(predictions_data),
        )

        return generate_response(
            success=True,
            data=response_data.model_dump(),
            message="Batch job submitted successfully",
            status_code=status.HTTP_202_ACCEPTED
        )

    except Exception as e:
        logger.error(f"Failed to submit batch job: {e}", exc_info=True)
        return generate_response(
            success=False,
            error="batch_submission_error",
            exception_error=str(e),
            message="Failed to submit batch job",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get(
    "/predict/batch/{job_id}",
    response_model=BatchJobStatus,
    summary="Get Batch Job Status",
    description="Check the status and results of a batch prediction job",
)
async def get_batch_status(job_id: str, api_key: str = Depends(verify_api_key)):
    """Get the status of a batch prediction job.

    Args:
        job_id: Unique job identifier
        api_key: Validated API key

    Returns:
        BatchJobStatus with job progress and results

    Raises:
        HTTPException: If job not found
    """
    try:
        task_result = AsyncResult(job_id, app=celery_app)
        state = task_result.state

        # Status response mapping
        status_handlers = {
            JobStatusEnum.PENDING: lambda: BatchJobStatus(
                job_id=job_id, status=JobStatusEnum.PENDING, progress=0.0
            ),
            JobStatusEnum.PROGRESS: lambda: BatchJobStatus(
                job_id=job_id,
                status=JobStatusEnum.PROGRESS,
                progress=(task_result.info or {}).get("progress", 0),
                total=(task_result.info or {}).get("total", 0),
                completed=(task_result.info or {}).get("completed", 0),
            ),
            JobStatusEnum.SUCCESS: lambda: BatchJobStatus(
                job_id=job_id,
                status=JobStatusEnum.SUCCESS,
                progress=100.0,
                total=len(task_result.result),
                completed=len(task_result.result),
                results=task_result.result,
            ),
            JobStatusEnum.FAILURE: lambda: BatchJobStatus(
                job_id=job_id,
                status=JobStatusEnum.FAILURE,
                progress=0.0,
                error=str(task_result.info) if task_result.info else "Unknown error",
            ),
        }

        # Get status data or default to PENDING for unknown states
        if state in status_handlers:
            status_data = status_handlers[state]()
        else:
            logger.warning(f"Unknown Celery state '{state}' for job {job_id}, defaulting to PENDING")
            status_data = BatchJobStatus(job_id=job_id, status=JobStatusEnum.PENDING)

        # Determine success and message
        is_success = state != JobStatusEnum.FAILURE
        messages = {
            JobStatusEnum.PENDING: "Job is pending",
            JobStatusEnum.PROGRESS: "Job is in progress",
            JobStatusEnum.SUCCESS: "Job completed successfully",
            JobStatusEnum.FAILURE: "Job failed",
        }
        message = messages.get(state, f"Job status: {state}")

        return generate_response(
            success=is_success,
            data=status_data.model_dump(),
            error="job_failed" if state == JobStatusEnum.FAILURE else None,
            message=message,
            status_code=200,
        )

    except Exception as e:
        logger.error(f"Failed to get batch status for job {job_id}: {e}", exc_info=True)
        return generate_response(
            success=False,
            error="status_retrieval_error",
            exception_error=str(e),
            message="Failed to retrieve job status",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
