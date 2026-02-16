"""Pytest configuration and fixtures.

This module provides shared fixtures for all tests.
"""

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_features():
    """Sample feature dictionary for testing.

    Returns:
        Dictionary with valid feature values
    """
    return {"employed": 1, "bank_balance": 10000.0, "annual_salary": 50000.0}


@pytest.fixture
def sample_prediction_result():
    """Sample prediction result for mocking.

    Returns:
        Dictionary with mock prediction result
    """
    return {"prediction": 0, "probability": 0.1234, "default_risk": "Low"}


@pytest.fixture
def mock_model_service():
    """Mock ModelService for testing without MLflow.

    Returns:
        Mock ModelService instance
    """
    mock = MagicMock()
    mock.model = MagicMock()
    mock.scaler = MagicMock()
    mock.model_version = "1"
    mock.model_name = "loan_default_model"
    mock.model_stage = "Production"

    # Mock get_model_info
    mock.get_model_info.return_value = {
        "model_name": "loan_default_model",
        "model_version": "1",
        "model_stage": "Production",
        "features": ["employed", "bank_balance", "annual_salary"],
        "engineered_features": ["Saving_Rate"],
        "model_type": "XGBClassifier",
    }

    # Mock predict
    mock.predict.return_value = {
        "prediction": 0,
        "probability": 0.1234,
        "default_risk": "Low",
    }

    return mock


@pytest.fixture
def client(mock_model_service) -> Generator:
    """Create FastAPI test client with mocked model service.

    Args:
        mock_model_service: Mocked ModelService fixture

    Yields:
        TestClient instance
    """
    # Patch the model service in app_state
    with patch("src.core.app_state.get_model_service", return_value=mock_model_service):
        with patch("src.core.app_state.is_model_loaded", return_value=True):
            # Mock API key verification to accept test API key
            with patch("src.api.dependencies.settings.api_key", "your-secret-api-key"):
                from src.main import app

                with TestClient(app) as test_client:
                    yield test_client
