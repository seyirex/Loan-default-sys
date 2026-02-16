"""Tests for health check endpoints.

This module tests the liveness and readiness probe endpoints.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


def test_liveness_check(client: TestClient):
    """Test liveness probe endpoint.

    Args:
        client: FastAPI test client
    """
    response = client.get("/healthz")

    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    assert response_data["success"] is True
    assert "data" in response_data
    assert "message" in response_data

    data = response_data["data"]
    assert data["status"] == "healthy"


def test_readiness_check_when_model_loaded(client: TestClient):
    """Test readiness probe when model is loaded.

    Args:
        client: FastAPI test client
    """
    response = client.get("/readyz")

    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    assert response_data["success"] is True
    assert "data" in response_data
    assert "message" in response_data

    data = response_data["data"]
    assert data["ready"] is True
    assert data["model_loaded"] is True


def test_readiness_check_when_model_not_loaded():
    """Test readiness probe when model is not loaded."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from src.main import app

    # Create TestClient first to let the app initialize
    with TestClient(app) as test_client:
        # Patch is_model_loaded to return False during the request
        with patch("src.api.v1.health.is_model_loaded", return_value=False):
            response = test_client.get("/readyz")

            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            response_data = response.json()
            assert response_data["success"] is False
            assert "data" in response_data

            data = response_data["data"]
            assert data["ready"] is False
            assert data["model_loaded"] is False
