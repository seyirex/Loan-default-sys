"""Tests for metrics service.

This module tests the Prometheus metrics tracking functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.services.metrics_service import metrics_service


@pytest.fixture
def metrics_svc():
    """Get the singleton MetricsService instance for testing.

    Returns:
        MetricsService instance
    """
    return metrics_service


def test_metrics_svc_initialization(metrics_svc):
    """Test MetricsService initialization.

    Args:
        metrics_svc: MetricsService fixture
    """
    assert metrics_svc.predictions_total is not None
    assert metrics_svc.prediction_duration is not None
    assert metrics_svc.prediction_result is not None
    assert metrics_svc.drift_psi is not None
    assert metrics_svc.drift_detected is not None
    assert metrics_svc.model_info is not None


def test_record_prediction_success(metrics_svc):
    """Test recording a successful prediction.

    Args:
        metrics_svc: MetricsService fixture
    """
    # Mock the metric labels to track calls
    with patch.object(metrics_svc.predictions_total, "labels") as mock_total_labels, \
         patch.object(metrics_svc.prediction_duration, "labels") as mock_duration_labels, \
         patch.object(metrics_svc.prediction_result, "labels") as mock_result_labels:

        mock_total_counter = MagicMock()
        mock_duration_histogram = MagicMock()
        mock_result_counter = MagicMock()

        mock_total_labels.return_value = mock_total_counter
        mock_duration_labels.return_value = mock_duration_histogram
        mock_result_labels.return_value = mock_result_counter

        # Record a prediction
        metrics_svc.record_prediction(
            model_version="1",
            prediction_type="realtime",
            status="success",
            duration=0.05,
            result=0,
        )

        # Verify predictions_total was incremented
        mock_total_labels.assert_called_once_with(
            model_version="1", prediction_type="realtime", status="success"
        )
        mock_total_counter.inc.assert_called_once()

        # Verify prediction_duration was recorded
        mock_duration_labels.assert_called_once_with(model_version="1", prediction_type="realtime")
        mock_duration_histogram.observe.assert_called_once_with(0.05)

        # Verify prediction_result was recorded
        mock_result_labels.assert_called_once_with(model_version="1", result="no_default")
        mock_result_counter.inc.assert_called_once()


def test_record_prediction_default_result(metrics_svc):
    """Test recording a prediction with default result.

    Args:
        metrics_svc: MetricsService fixture
    """
    with patch.object(metrics_svc.prediction_result, "labels") as mock_result_labels:
        mock_result_counter = MagicMock()
        mock_result_labels.return_value = mock_result_counter

        # Record prediction with default result
        metrics_svc.record_prediction(
            model_version="1",
            prediction_type="batch",
            status="success",
            duration=0.1,
            result=1,
        )

        # Verify result was recorded as "default"
        mock_result_labels.assert_called_once_with(model_version="1", result="default")
        mock_result_counter.inc.assert_called_once()


def test_record_prediction_without_result(metrics_svc):
    """Test recording a prediction without a result.

    Args:
        metrics_svc: MetricsService fixture
    """
    with patch.object(metrics_svc.prediction_result, "labels") as mock_result_labels:
        # Record prediction without result
        metrics_svc.record_prediction(
            model_version="2",
            prediction_type="realtime",
            status="failure",
            duration=0.02,
            result=None,
        )

        # Verify prediction_result was not called when result is None
        mock_result_labels.assert_not_called()


def test_record_drift(metrics_svc):
    """Test recording drift detection metrics.

    Args:
        metrics_svc: MetricsService fixture
    """
    with patch.object(metrics_svc.drift_psi, "labels") as mock_psi_labels, \
         patch.object(metrics_svc.drift_detected, "set") as mock_drift_set:

        mock_psi_gauge = MagicMock()
        mock_psi_labels.return_value = mock_psi_gauge

        # Record drift metrics
        psi_scores = {
            "employed": 0.05,
            "bank_balance": 0.12,
            "annual_salary": 0.08,
            "saving_rate": 0.15,
        }

        metrics_svc.record_drift(psi_scores=psi_scores, drift_detected=True)

        # Verify PSI was set for each feature
        assert mock_psi_labels.call_count == 4
        assert mock_psi_gauge.set.call_count == 4

        # Verify drift detection flag was set
        mock_drift_set.assert_called_once_with(1)


def test_record_drift_no_drift_detected(metrics_svc):
    """Test recording drift metrics when no drift is detected.

    Args:
        metrics_svc: MetricsService fixture
    """
    with patch.object(metrics_svc.drift_detected, "set") as mock_drift_set:
        # Record no drift
        psi_scores = {"employed": 0.02, "bank_balance": 0.05}

        metrics_svc.record_drift(psi_scores=psi_scores, drift_detected=False)

        # Verify drift detection flag was set to 0
        mock_drift_set.assert_called_once_with(0)


def test_set_model_info(metrics_svc):
    """Test setting model information metric.

    Args:
        metrics_svc: MetricsService fixture
    """
    with patch.object(metrics_svc.model_info, "info") as mock_info:
        # Set model info
        metrics_svc.set_model_info(
            model_name="loan_default_model",
            model_version="3",
            model_stage="Production",
            model_type="XGBClassifier",
        )

        # Verify model_info was called with correct data
        mock_info.assert_called_once_with(
            {
                "name": "loan_default_model",
                "version": "3",
                "stage": "Production",
                "type": "XGBClassifier",
            }
        )


def test_record_prediction_with_batch_type(metrics_svc):
    """Test recording a batch prediction.

    Args:
        metrics_svc: MetricsService fixture
    """
    with patch.object(metrics_svc.predictions_total, "labels") as mock_labels:
        mock_counter = MagicMock()
        mock_labels.return_value = mock_counter

        # Record batch prediction
        metrics_svc.record_prediction(
            model_version="2",
            prediction_type="batch",
            status="success",
            duration=0.001,
            result=1,
        )

        # Verify correct labels were used
        mock_labels.assert_called_once_with(
            model_version="2", prediction_type="batch", status="success"
        )
        mock_counter.inc.assert_called_once()


def test_record_prediction_with_failure_status(metrics_svc):
    """Test recording a failed prediction.

    Args:
        metrics_svc: MetricsService fixture
    """
    with patch.object(metrics_svc.predictions_total, "labels") as mock_labels:
        mock_counter = MagicMock()
        mock_labels.return_value = mock_counter

        # Record failed prediction
        metrics_svc.record_prediction(
            model_version="1",
            prediction_type="realtime",
            status="failure",
            duration=0.03,
            result=None,
        )

        # Verify failure status was recorded
        mock_labels.assert_called_once_with(
            model_version="1", prediction_type="realtime", status="failure"
        )
        mock_counter.inc.assert_called_once()
