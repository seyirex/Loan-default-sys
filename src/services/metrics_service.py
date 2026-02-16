"""Prometheus metrics service.

This module defines and manages custom Prometheus metrics for the application.
"""

from typing import Dict, Optional

from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, Info


class MetricsService:
    """Service for managing Prometheus metrics.

    Defines and provides methods to record custom metrics for loan predictions.
    """

    def __init__(self):
        """Initialize Prometheus metrics."""

        # Prediction counter
        self.predictions_total = Counter(
            "loan_predictions_total",
            "Total number of loan default predictions",
            ["model_version", "prediction_type", "status"],
        )

        # Prediction duration histogram
        self.prediction_duration = Histogram(
            "loan_prediction_duration_seconds",
            "Time spent processing predictions",
            ["model_version", "prediction_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        # Prediction result counter
        self.prediction_result = Counter(
            "loan_prediction_result_total",
            "Count of prediction results (default/no default)",
            ["model_version", "result"],
        )

        # Drift metrics
        self.drift_psi = Gauge(
            "loan_model_drift_psi",
            "Population Stability Index (PSI) for drift detection",
            ["feature"],
        )

        self.drift_detected = Gauge(
            "loan_model_drift_detected",
            "Binary indicator of drift detection (0 = no drift, 1 = drift detected)",
        )

        # Model info metric
        self.model_info = Info("loan_model", "Information about the deployed model")

        logger.info("Prometheus metrics initialized")

    def record_prediction(
        self,
        model_version: str,
        prediction_type: str,
        status: str,
        duration: float,
        result: Optional[int] = None,
    ) -> None:
        """Record a prediction event.

        Args:
            model_version: Version of the model used
            prediction_type: Type of prediction (realtime/batch)
            status: Status of prediction (success/failure)
            duration: Time taken for prediction in seconds
            result: Prediction result (0 = no default, 1 = default)
        """
        # Increment prediction counter
        self.predictions_total.labels(
            model_version=model_version, prediction_type=prediction_type, status=status
        ).inc()

        # Record duration
        self.prediction_duration.labels(
            model_version=model_version, prediction_type=prediction_type
        ).observe(duration)

        # Record result if provided
        if result is not None:
            result_label = "default" if result == 1 else "no_default"
            self.prediction_result.labels(model_version=model_version, result=result_label).inc()

    def record_drift(self, psi_scores: Dict[str, float], drift_detected: bool) -> None:
        """Record drift detection metrics.

        Args:
            psi_scores: Dictionary of PSI scores per feature
            drift_detected: Whether drift was detected
        """
        # Update PSI gauge for each feature
        for feature, psi_value in psi_scores.items():
            self.drift_psi.labels(feature=feature).set(psi_value)

        # Update drift detection binary indicator
        self.drift_detected.set(1 if drift_detected else 0)

    def set_model_info(
        self, model_name: str, model_version: str, model_stage: str, model_type: str
    ) -> None:
        """Set model information metric.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_stage: Stage of the model (Production/Staging)
            model_type: Type of model (e.g., XGBClassifier)
        """
        self.model_info.info(
            {
                "name": model_name,
                "version": model_version,
                "stage": model_stage,
                "type": model_type,
            }
        )
        logger.info(f"Model info metric set: {model_name} v{model_version} ({model_stage})")


# Create singleton instance
metrics_service = MetricsService()
