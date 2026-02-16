"""Celery service for batch prediction processing.

This module defines Celery tasks for asynchronous batch prediction processing.
"""

# TODO: Implement CSV/Excel upload feature
# - Allow users to upload CSV or Excel files for batch predictions
# - Process the uploaded file and generate predictions
# - Export results as CSV file
# - Provide a downloadable URL for the user to retrieve the results CSV

import time
from typing import Any, Dict, List

from celery import Task
from loguru import logger

from src.celery_app import celery_app
from src.config import settings
from src.services.drift_detector import DriftDetector
from src.utils.monitoring import record_monitoring_data

# Global drift detector instance for batch predictions
drift_detector = DriftDetector()


@celery_app.task(bind=True, name="batch_predict")
def batch_predict_task(self: Task, predictions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of predictions asynchronously.

    This task:
    1. Loads the model
    2. Processes predictions in batches
    3. Updates progress
    4. Returns results

    Args:
        self: Celery task instance (bound)
        predictions_data: List of prediction request dictionaries

    Returns:
        List of prediction results

    Raises:
        Exception: If batch processing fails
    """
    logger.info(f"Starting batch prediction task - Total: {len(predictions_data)}")

    try:
        # Import model service here to avoid circular imports
        from src.services.model_service import ModelService

        # Initialize model service
        model_service = ModelService(
            mlflow_uri=settings.mlflow_tracking_uri,
            model_name=settings.model_name,
            model_stage=settings.model_stage,
        )

        # Load model if not already loaded
        if model_service.model is None:
            logger.info("Loading model for batch processing...")
            model_service.load_model()

        model_version = model_service.model_version or "unknown"

        # Process predictions in batches of 100
        batch_size = 100
        total = len(predictions_data)
        results = []

        for i in range(0, total, batch_size):
            batch = predictions_data[i : i + batch_size]
            batch_results = []

            # Extract features for vectorized batch processing
            batch_features = [
                {
                    "employed": pred_data["employed"],
                    "bank_balance": pred_data["bank_balance"],
                    "annual_salary": pred_data["annual_salary"],
                }
                for pred_data in batch
            ]

            # Vectorized batch prediction (10-100x faster!)
            batch_start_time = time.time()
            predictions = model_service.predict_batch(batch_features)
            batch_duration = time.time() - batch_start_time

            # Calculate average duration per prediction for metrics
            avg_duration = batch_duration / len(batch_features) if batch_features else 0

            # Process results and record monitoring for each prediction
            for idx, (features, result) in enumerate(zip(batch_features, predictions)):
                try:
                    # Record monitoring data (drift detection + metrics)
                    record_monitoring_data(
                        drift_detector=drift_detector,
                        features=features,
                        model_version=model_version,
                        prediction_result=result["prediction"],
                        duration=avg_duration,
                        prediction_type="batch",
                    )

                    batch_results.append(
                        {
                            "status": "success",
                            "prediction": result["prediction"],
                            "probability": result["probability"],
                            "default_risk": result["default_risk"],
                            "features": features,
                        }
                    )

                except Exception as e:
                    logger.error(f"Monitoring failed for item {i + idx}: {e}")
                    # Still include the prediction result even if monitoring fails
                    batch_results.append(
                        {
                            "status": "success",
                            "prediction": result["prediction"],
                            "probability": result["probability"],
                            "default_risk": result["default_risk"],
                            "features": features,
                        }
                    )

            results.extend(batch_results)

            # Update progress
            completed = len(results)
            progress = (completed / total) * 100

            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": round(progress, 2),
                    "total": total,
                    "completed": completed,
                },
            )

            logger.info(f"Batch progress: {completed}/{total} ({progress:.1f}%)")

        logger.info(f"Batch prediction completed - Total: {total}, Success: {len([r for r in results if r['status'] == 'success'])}")

        return results

    except Exception as e:
        logger.error(f"Batch prediction task failed: {e}", exc_info=True)
        raise
