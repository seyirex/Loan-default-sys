"""Monitoring utilities for drift detection and metrics.

This module provides background monitoring functions to prevent
latency impact on prediction endpoints.
"""

import random
from typing import Dict, Optional

from loguru import logger

from src.config import settings
from src.core.app_state import get_model_service
from src.services.drift_detector import DriftDetector
from src.services.metrics_service import metrics_service

# Cache for model version to avoid repeated lookups
_model_version_cache: Optional[str] = None


def get_cached_model_version() -> str:
    """Get model version from cache or fetch if not cached.

    Returns:
        Model version string
    """
    global _model_version_cache

    if _model_version_cache is None:
        model_service = get_model_service()
        model_info = model_service.get_model_info()
        _model_version_cache = model_info["model_version"]
        logger.debug(f"Model version cached: {_model_version_cache}")

    return _model_version_cache


def clear_model_version_cache() -> None:
    """Clear the model version cache.

    Call this when the model is updated to force a fresh lookup.
    """
    global _model_version_cache
    _model_version_cache = None
    logger.info("Model version cache cleared")


def record_monitoring_data(
    drift_detector: DriftDetector,
    features: Dict[str, float],
    model_version: str,
    prediction_result: int,
    duration: float,
    prediction_type: str = "realtime",
    sampling_rate: Optional[float] = None,
) -> None:
    """Record drift detection and metrics in background.

    This function runs asynchronously after the response is sent to the user,
    preventing monitoring operations from adding latency to predictions.

    Drift detection is sampled based on sampling_rate to reduce
    computational overhead (default 10% for realtime, 5% for batch).

    Args:
        drift_detector: DriftDetector instance to use
        features: Feature dictionary with values
        model_version: Model version used for prediction
        prediction_result: Prediction result (0 or 1)
        duration: Prediction duration in seconds
        prediction_type: Type of prediction ("realtime" or "batch")
        sampling_rate: Custom sampling rate (uses config default if None)
    """
    try:
        # Use custom sampling rate or default based on prediction type
        if sampling_rate is None:
            sampling_rate = (
                settings.drift_batch_sampling_rate
                if prediction_type == "batch"
                else settings.drift_sampling_rate
            )

        # Sample drift detection to reduce overhead
        if random.random() < sampling_rate:
            # Add saving_rate feature
            features_with_saving_rate = {
                **features,
                "saving_rate": features["bank_balance"] / features["annual_salary"],
            }

            # Add to drift detector
            drift_detector.add_observation(features_with_saving_rate)

            # Check for drift
            drift_result = drift_detector.check_drift()
            if drift_result.get("drift_detected"):
                logger.warning(f"Drift detected: {drift_result}")
                # Record drift metrics
                if "psi_scores" in drift_result:
                    metrics_service.record_drift(
                        drift_result["psi_scores"], drift_result["drift_detected"]
                    )

            logger.debug(f"Drift check completed (sampling rate: {sampling_rate})")

        # Always record prediction metrics (lightweight operation)
        metrics_service.record_prediction(
            model_version=model_version,
            prediction_type=prediction_type,
            status="success",
            duration=duration,
            result=prediction_result,
        )

        logger.debug("Background monitoring completed for prediction")

    except Exception as e:
        # Don't let monitoring errors affect the prediction
        logger.error(f"Error in background monitoring: {e}", exc_info=True)
