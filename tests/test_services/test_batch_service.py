"""Tests for batch service.

This module tests the Celery batch prediction task functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.services.batch_service import batch_predict_task


@pytest.fixture
def sample_batch_data():
    """Sample batch prediction data.

    Returns:
        List of prediction request dictionaries
    """
    return [
        {"employed": 1, "bank_balance": 10000.0, "annual_salary": 50000.0},
        {"employed": 0, "bank_balance": 2000.0, "annual_salary": 25000.0},
        {"employed": 1, "bank_balance": 25000.0, "annual_salary": 80000.0},
    ]


@pytest.fixture
def mock_model_service():
    """Create a mock ModelService.

    Returns:
        Mock ModelService with predict_batch method
    """
    service = MagicMock()
    service.model = MagicMock()
    service.model_version = "2"
    service.predict_batch.return_value = [
        {"prediction": 0, "probability": 0.15, "default_risk": "Low"},
        {"prediction": 1, "probability": 0.75, "default_risk": "High"},
        {"prediction": 0, "probability": 0.10, "default_risk": "Low"},
    ]
    return service


@pytest.fixture(autouse=True)
def mock_task_update_state():
    """Automatically mock update_state for all batch service tests.

    Returns:
        Mock for update_state method
    """
    with patch.object(batch_predict_task, "update_state") as mock_update:
        yield mock_update


def test_batch_predict_task_success(sample_batch_data, mock_model_service):
    """Test successful batch prediction task.

    Args:
        sample_batch_data: Sample batch data fixture
        mock_model_service: Mock ModelService fixture
    """
    with patch("src.services.model_service.ModelService", return_value=mock_model_service), \
         patch("src.services.batch_service.record_monitoring_data"):

        # Call the task's run method directly (bypasses Celery wrapping)
        results = batch_predict_task.run(sample_batch_data)

        # Verify results
        assert len(results) == 3

        # Verify first result
        assert results[0]["status"] == "success"
        assert results[0]["prediction"] == 0
        assert results[0]["probability"] == 0.15
        assert results[0]["default_risk"] == "Low"
        assert "features" in results[0]

        # Verify second result
        assert results[1]["status"] == "success"
        assert results[1]["prediction"] == 1
        assert results[1]["probability"] == 0.75
        assert results[1]["default_risk"] == "High"

        # Verify third result
        assert results[2]["status"] == "success"
        assert results[2]["prediction"] == 0

        # Verify model service was called
        mock_model_service.predict_batch.assert_called()


def test_batch_predict_task_with_progress_updates(sample_batch_data, mock_model_service, mock_task_update_state):
    """Test batch prediction task progress updates.

    Args:
        sample_batch_data: Sample batch data fixture
        mock_model_service: Mock ModelService fixture
        mock_task_update_state: Mock for update_state method
    """
    with patch("src.services.model_service.ModelService", return_value=mock_model_service), \
         patch("src.services.batch_service.record_monitoring_data"):

        results = batch_predict_task.run(sample_batch_data)

        # Verify progress updates were called
        assert mock_task_update_state.call_count > 0

        # Check last progress update
        last_call = mock_task_update_state.call_args_list[-1]
        assert last_call[1]["state"] == "PROGRESS"
        assert "meta" in last_call[1]
        meta = last_call[1]["meta"]
        assert meta["completed"] == 3
        assert meta["total"] == 3
        assert meta["progress"] == 100.0


def test_batch_predict_task_large_batch(mock_task_update_state):
    """Test batch prediction with large batch (>100 items).

    Args:
        mock_task_update_state: Mock for update_state method
    """
    # Create 250 items to test batch processing
    large_batch = [
        {"employed": 1, "bank_balance": 10000.0, "annual_salary": 50000.0}
    ] * 250

    mock_service = MagicMock()
    mock_service.model = MagicMock()
    mock_service.model_version = "1"

    # Mock returns results for each batch of 100
    mock_service.predict_batch.return_value = [
        {"prediction": 0, "probability": 0.2, "default_risk": "Low"}
    ] * 100

    with patch("src.services.model_service.ModelService", return_value=mock_service), \
         patch("src.services.batch_service.record_monitoring_data"):

        results = batch_predict_task.run(large_batch)

        # Should process all 250 items in 3 batches (100, 100, 50)
        assert len(results) == 250

        # Verify predict_batch was called 3 times
        assert mock_service.predict_batch.call_count == 3

        # Verify progress updates were made
        assert mock_task_update_state.call_count >= 3


def test_batch_predict_task_empty_batch():
    """Test batch prediction with empty batch."""
    mock_service = MagicMock()
    mock_service.model = MagicMock()
    mock_service.model_version = "1"

    with patch("src.services.model_service.ModelService", return_value=mock_service), \
         patch("src.services.batch_service.record_monitoring_data"):

        results = batch_predict_task.run([])

        # Should return empty list
        assert results == []


def test_batch_predict_task_model_loading():
    """Test that model is loaded when not already loaded."""
    mock_service = MagicMock()
    mock_service.model = None  # Model not loaded initially
    mock_service.model_version = "3"
    mock_service.predict_batch.return_value = [
        {"prediction": 0, "probability": 0.1, "default_risk": "Low"}
    ]

    # After load_model is called, set model
    def side_effect():
        mock_service.model = MagicMock()

    mock_service.load_model.side_effect = side_effect

    mock_task = MagicMock()
    sample_data = [{"employed": 1, "bank_balance": 5000.0, "annual_salary": 40000.0}]

    with patch("src.services.model_service.ModelService", return_value=mock_service), \
         patch("src.services.batch_service.record_monitoring_data"):

        results = batch_predict_task.run(sample_data)

        # Verify load_model was called
        mock_service.load_model.assert_called_once()

        # Verify prediction succeeded
        assert len(results) == 1
        assert results[0]["status"] == "success"


def test_batch_predict_task_monitoring_failure_handling(sample_batch_data):
    """Test that monitoring failures don't break predictions.

    Args:
        sample_batch_data: Sample batch data fixture
    """
    mock_service = MagicMock()
    mock_service.model = MagicMock()
    mock_service.model_version = "2"
    mock_service.predict_batch.return_value = [
        {"prediction": 0, "probability": 0.2, "default_risk": "Low"},
        {"prediction": 1, "probability": 0.8, "default_risk": "High"},
        {"prediction": 0, "probability": 0.1, "default_risk": "Low"},
    ]

    mock_task = MagicMock()

    # Make monitoring fail
    with patch("src.services.model_service.ModelService", return_value=mock_service), \
         patch("src.services.batch_service.record_monitoring_data", side_effect=Exception("Monitoring error")):

        results = batch_predict_task.run(sample_batch_data)

        # Predictions should still succeed despite monitoring failure
        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)

        # Verify all results have predictions
        assert results[0]["prediction"] == 0
        assert results[1]["prediction"] == 1
        assert results[2]["prediction"] == 0


def test_batch_predict_task_exception_handling():
    """Test batch task exception handling."""
    mock_service = MagicMock()
    mock_service.model = None
    mock_service.load_model.side_effect = Exception("Model loading failed")

    mock_task = MagicMock()
    sample_data = [{"employed": 1, "bank_balance": 5000.0, "annual_salary": 40000.0}]

    with patch("src.services.model_service.ModelService", return_value=mock_service):
        # Should raise exception
        with pytest.raises(Exception, match="Model loading failed"):
            batch_predict_task.run(sample_data)


def test_batch_predict_task_with_monitoring_data(sample_batch_data, mock_model_service):
    """Test that monitoring data is recorded correctly.

    Args:
        sample_batch_data: Sample batch data fixture
        mock_model_service: Mock ModelService fixture
    """
    mock_task = MagicMock()

    with patch("src.services.model_service.ModelService", return_value=mock_model_service), \
         patch("src.services.batch_service.record_monitoring_data") as mock_record:

        batch_predict_task.run(sample_batch_data)

        # Verify monitoring was called for each prediction
        assert mock_record.call_count == 3

        # Verify monitoring was called with correct parameters
        first_call = mock_record.call_args_list[0]
        assert "drift_detector" in first_call[1]
        assert "features" in first_call[1]
        assert "model_version" in first_call[1]
        assert "prediction_result" in first_call[1]
        assert "duration" in first_call[1]
        assert first_call[1]["prediction_type"] == "batch"


def test_batch_predict_task_batch_size_processing():
    """Test that batches are processed in chunks of 100."""
    # Create exactly 200 items
    batch_data = [
        {"employed": 1, "bank_balance": 10000.0, "annual_salary": 50000.0}
    ] * 200

    mock_service = MagicMock()
    mock_service.model = MagicMock()
    mock_service.model_version = "1"

    # Track the batch sizes
    batch_sizes = []

    def track_batch_size(features_list):
        batch_sizes.append(len(features_list))
        return [{"prediction": 0, "probability": 0.1, "default_risk": "Low"}] * len(features_list)

    mock_service.predict_batch.side_effect = track_batch_size

    mock_task = MagicMock()

    with patch("src.services.model_service.ModelService", return_value=mock_service), \
         patch("src.services.batch_service.record_monitoring_data"):

        results = batch_predict_task.run(batch_data)

        # Verify results
        assert len(results) == 200

        # Verify batches were processed in chunks of 100
        assert batch_sizes == [100, 100]


def test_batch_predict_task_uses_vectorized_prediction(sample_batch_data, mock_model_service):
    """Test that the task uses vectorized batch prediction.

    Args:
        sample_batch_data: Sample batch data fixture
        mock_model_service: Mock ModelService fixture
    """
    mock_task = MagicMock()

    with patch("src.services.model_service.ModelService", return_value=mock_model_service), \
         patch("src.services.batch_service.record_monitoring_data"):

        batch_predict_task.run(sample_batch_data)

        # Verify predict_batch was called (not predict)
        mock_model_service.predict_batch.assert_called()

        # Verify it was called with the full batch
        call_args = mock_model_service.predict_batch.call_args[0][0]
        assert len(call_args) == 3
