"""Tests for prediction endpoints.

This module tests the prediction API endpoints.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient


def test_predict_without_api_key(client: TestClient, sample_features):
    """Test prediction endpoint without API key.

    Args:
        client: FastAPI test client
        sample_features: Sample feature fixture
    """
    response = client.post("/api/v1/predict", json=sample_features)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_with_invalid_api_key(client: TestClient, sample_features):
    """Test prediction endpoint with invalid API key.

    Args:
        client: FastAPI test client
        sample_features: Sample feature fixture
    """
    response = client.post(
        "/api/v1/predict", json=sample_features, headers={"X-API-Key": "invalid-key"}
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_predict_with_valid_api_key(client: TestClient, sample_features, mock_model_service):
    """Test prediction endpoint with valid API key.

    Args:
        client: FastAPI test client
        sample_features: Sample feature fixture
        mock_model_service: Mocked model service
    """
    with patch("src.api.v1.predict.get_model_service", return_value=mock_model_service):
        response = client.post(
            "/api/v1/predict",
            json=sample_features,
            headers={"X-API-Key": "your-secret-api-key"},
        )

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["success"] is True
        assert "data" in response_data

        data = response_data["data"]
        assert "prediction" in data
        assert "probability" in data
        assert "default_risk" in data
        assert "model_version" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0


def test_predict_with_invalid_features(client: TestClient):
    """Test prediction endpoint with invalid features.

    Args:
        client: FastAPI test client
    """
    invalid_features = {
        "employed": 2,  # Should be 0 or 1
        "bank_balance": -1000,  # Should be >= 0
        "annual_salary": -50000,  # Should be > 0
    }

    response = client.post(
        "/api/v1/predict",
        json=invalid_features,
        headers={"X-API-Key": "your-secret-api-key"},
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_with_missing_features(client: TestClient):
    """Test prediction endpoint with missing features.

    Args:
        client: FastAPI test client
    """
    incomplete_features = {"employed": 1, "bank_balance": 10000.0}
    # Missing annual_salary

    response = client.post(
        "/api/v1/predict",
        json=incomplete_features,
        headers={"X-API-Key": "your-secret-api-key"},
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_batch_predict_submission(client: TestClient, sample_features):
    """Test batch prediction job submission.

    Args:
        client: FastAPI test client
        sample_features: Sample feature fixture
    """
    batch_request = {"predictions": [sample_features, sample_features]}

    with patch("src.api.v1.predict.batch_predict_task") as mock_task:
        # Create a mock with a string ID
        mock_result = MagicMock()
        mock_result.id = "test-job-123"
        mock_task.apply_async.return_value = mock_result

        response = client.post(
            "/api/v1/predict/batch",
            json=batch_request,
            headers={"X-API-Key": "your-secret-api-key"},
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()
        assert response_data["success"] is True
        assert "data" in response_data

        data = response_data["data"]
        assert "job_id" in data
        assert data["status"] == "PENDING"
        assert data["total_predictions"] == 2
