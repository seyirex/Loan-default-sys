"""Tests for model information endpoint.

This module tests the model info API endpoint.
"""

from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient


def test_model_info_without_api_key(client: TestClient):
    """Test model info endpoint without API key.

    Args:
        client: FastAPI test client
    """
    response = client.get("/api/v1/model/info")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_model_info_with_invalid_api_key(client: TestClient):
    """Test model info endpoint with invalid API key.

    Args:
        client: FastAPI test client
    """
    response = client.get("/api/v1/model/info", headers={"X-API-Key": "invalid-key"})

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_model_info_with_valid_api_key(client: TestClient, mock_model_service):
    """Test model info endpoint with valid API key.

    Args:
        client: FastAPI test client
        mock_model_service: Mocked model service
    """
    with patch("src.api.v1.model.get_model_service", return_value=mock_model_service):
        response = client.get(
            "/api/v1/model/info", headers={"X-API-Key": "your-secret-api-key"}
        )

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["success"] is True
        assert "data" in response_data

        data = response_data["data"]
        assert "model_name" in data
        assert "model_version" in data
        assert "model_stage" in data
        assert "features" in data
        assert "engineered_features" in data
        assert "model_type" in data
        assert data["model_name"] == "loan_default_model"
        assert isinstance(data["features"], list)
        assert isinstance(data["engineered_features"], list)
