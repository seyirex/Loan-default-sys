"""Tests for model service.

This module tests the MLflow model loading and prediction functionality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.services.model_service import ModelService


@pytest.fixture
def model_service():
    """Create a ModelService instance for testing.

    Returns:
        ModelService with mocked MLflow
    """
    with patch("src.services.model_service.mlflow.set_tracking_uri"):
        service = ModelService(
            mlflow_uri="sqlite:///test.db",
            model_name="test_model",
            model_stage="Staging",
        )
    return service


@pytest.fixture
def mock_model():
    """Create a mock ML model.

    Returns:
        Mock model with predict and predict_proba methods
    """
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.8, 0.2]])
    return model


@pytest.fixture
def mock_scaler():
    """Create a mock scaler.

    Returns:
        Mock scaler with transform method
    """
    scaler = MagicMock()
    scaler.transform.return_value = np.array([[1.0, 0.5, -0.3, 0.2]])
    return scaler


def test_model_service_initialization():
    """Test ModelService initialization."""
    with patch("src.services.model_service.mlflow.set_tracking_uri") as mock_set_uri:
        service = ModelService(
            mlflow_uri="sqlite:///mlflow.db",
            model_name="loan_model",
            model_stage="Production",
        )

        assert service.mlflow_uri == "sqlite:///mlflow.db"
        assert service.model_name == "loan_model"
        assert service.model_stage == "Production"
        assert service.model is None
        assert service.scaler is None
        assert service.model_version is None

        mock_set_uri.assert_called_once_with("sqlite:///mlflow.db")


def test_load_model_success(model_service, mock_model, mock_scaler):
    """Test successful model loading from MLflow.

    Args:
        model_service: ModelService fixture
        mock_model: Mock ML model
        mock_scaler: Mock scaler
    """
    # Mock MLflow components
    mock_version = MagicMock()
    mock_version.version = "2"
    mock_version.current_stage = "Staging"
    mock_version.run_id = "test-run-123"

    with patch("src.services.model_service.mlflow.sklearn.load_model") as mock_load, \
         patch("src.services.model_service.mlflow.MlflowClient") as mock_client_class:

        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mock_version]
        mock_client_class.return_value = mock_client

        # First call returns model, second call returns scaler
        mock_load.side_effect = [mock_model, mock_scaler]

        # Load model
        model_service.load_model()

        # Verify model and scaler were loaded
        assert model_service.model == mock_model
        assert model_service.scaler == mock_scaler
        assert model_service.model_version == "2"

        # Verify MLflow calls
        assert mock_load.call_count == 2
        mock_load.assert_any_call("models:/test_model/Staging")
        mock_load.assert_any_call("runs:/test-run-123/scaler")


def test_load_model_no_version_found(model_service):
    """Test model loading when no version exists in the specified stage.

    Args:
        model_service: ModelService fixture
    """
    # Mock empty version list
    with patch("src.services.model_service.mlflow.sklearn.load_model"), \
         patch("src.services.model_service.mlflow.MlflowClient") as mock_client_class:

        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = []
        mock_client_class.return_value = mock_client

        # Should raise ValueError
        with pytest.raises(ValueError, match="No model found in stage 'Staging'"):
            model_service.load_model()


def test_load_model_exception(model_service):
    """Test model loading with MLflow exception.

    Args:
        model_service: ModelService fixture
    """
    with patch("src.services.model_service.mlflow.sklearn.load_model", side_effect=Exception("MLflow error")):
        with pytest.raises(Exception, match="MLflow error"):
            model_service.load_model()


def test_engineer_features(model_service):
    """Test feature engineering logic.

    Args:
        model_service: ModelService fixture
    """
    features = {
        "employed": 1,
        "bank_balance": 15000.0,
        "annual_salary": 60000.0,
    }

    df = model_service._engineer_features(features)

    # Verify DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == ["Employed", "Bank Balance", "Annual Salary", "Saving_Rate"]

    # Verify values
    assert df["Employed"].iloc[0] == 1
    assert df["Bank Balance"].iloc[0] == 15000.0
    assert df["Annual Salary"].iloc[0] == 60000.0
    assert df["Saving_Rate"].iloc[0] == 0.25  # 15000 / 60000


def test_engineer_features_zero_salary(model_service):
    """Test feature engineering with zero salary.

    Args:
        model_service: ModelService fixture
    """
    features = {
        "employed": 0,
        "bank_balance": 5000.0,
        "annual_salary": 0.0,
    }

    df = model_service._engineer_features(features)

    # Verify saving rate is 0 when salary is 0
    assert df["Saving_Rate"].iloc[0] == 0


def test_preprocess_success(model_service, mock_scaler):
    """Test successful feature preprocessing.

    Args:
        model_service: ModelService fixture
        mock_scaler: Mock scaler
    """
    model_service.scaler = mock_scaler

    features = {
        "employed": 1,
        "bank_balance": 10000.0,
        "annual_salary": 50000.0,
    }

    result = model_service.preprocess(features)

    # Verify scaler was called
    mock_scaler.transform.assert_called_once()

    # Verify result is numpy array
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 4)


def test_preprocess_without_scaler(model_service):
    """Test preprocessing when scaler is not loaded.

    Args:
        model_service: ModelService fixture
    """
    features = {
        "employed": 1,
        "bank_balance": 10000.0,
        "annual_salary": 50000.0,
    }

    with pytest.raises(ValueError, match="Scaler not loaded"):
        model_service.preprocess(features)


def test_predict_success(model_service, mock_model, mock_scaler):
    """Test successful single prediction.

    Args:
        model_service: ModelService fixture
        mock_model: Mock ML model
        mock_scaler: Mock scaler
    """
    model_service.model = mock_model
    model_service.scaler = mock_scaler

    features = {
        "employed": 1,
        "bank_balance": 10000.0,
        "annual_salary": 50000.0,
    }

    result = model_service.predict(features)

    # Verify result structure
    assert "prediction" in result
    assert "probability" in result
    assert "default_risk" in result

    # Verify values
    assert result["prediction"] == 0
    assert result["probability"] == 0.2
    assert result["default_risk"] == "Low"

    # Verify model was called
    mock_model.predict.assert_called_once()
    mock_model.predict_proba.assert_called_once()


def test_predict_high_risk(model_service, mock_model, mock_scaler):
    """Test prediction with high default risk.

    Args:
        model_service: ModelService fixture
        mock_model: Mock ML model
        mock_scaler: Mock scaler
    """
    model_service.model = mock_model
    model_service.scaler = mock_scaler

    # Mock high risk prediction
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    features = {
        "employed": 0,
        "bank_balance": 1000.0,
        "annual_salary": 20000.0,
    }

    result = model_service.predict(features)

    assert result["prediction"] == 1
    assert result["probability"] == 0.7
    assert result["default_risk"] == "High"


def test_predict_without_model(model_service, mock_scaler):
    """Test prediction when model is not loaded.

    Args:
        model_service: ModelService fixture
        mock_scaler: Mock scaler
    """
    model_service.scaler = mock_scaler

    features = {
        "employed": 1,
        "bank_balance": 10000.0,
        "annual_salary": 50000.0,
    }

    with pytest.raises(ValueError, match="Model not loaded"):
        model_service.predict(features)


def test_predict_batch_success(model_service, mock_model, mock_scaler):
    """Test successful batch prediction.

    Args:
        model_service: ModelService fixture
        mock_model: Mock ML model
        mock_scaler: Mock scaler
    """
    model_service.model = mock_model
    model_service.scaler = mock_scaler

    # Mock batch predictions
    mock_model.predict.return_value = np.array([0, 1, 0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])

    features_list = [
        {"employed": 1, "bank_balance": 10000.0, "annual_salary": 50000.0},
        {"employed": 0, "bank_balance": 1000.0, "annual_salary": 20000.0},
        {"employed": 1, "bank_balance": 20000.0, "annual_salary": 80000.0},
    ]

    results = model_service.predict_batch(features_list)

    # Verify result count
    assert len(results) == 3

    # Verify first result
    assert results[0]["prediction"] == 0
    assert results[0]["probability"] == 0.2
    assert results[0]["default_risk"] == "Low"

    # Verify second result (high risk)
    assert results[1]["prediction"] == 1
    assert results[1]["probability"] == 0.7
    assert results[1]["default_risk"] == "High"

    # Verify third result
    assert results[2]["prediction"] == 0
    assert results[2]["probability"] == 0.1
    assert results[2]["default_risk"] == "Low"

    # Verify model was called once (vectorized)
    mock_model.predict.assert_called_once()
    mock_model.predict_proba.assert_called_once()


def test_predict_batch_empty_list(model_service, mock_model, mock_scaler):
    """Test batch prediction with empty list.

    Args:
        model_service: ModelService fixture
        mock_model: Mock ML model
        mock_scaler: Mock scaler
    """
    model_service.model = mock_model
    model_service.scaler = mock_scaler

    results = model_service.predict_batch([])

    # Should return empty list
    assert results == []

    # Model should not be called
    mock_model.predict.assert_not_called()


def test_predict_batch_without_model(model_service):
    """Test batch prediction when model is not loaded.

    Args:
        model_service: ModelService fixture
    """
    features_list = [{"employed": 1, "bank_balance": 10000.0, "annual_salary": 50000.0}]

    with pytest.raises(ValueError, match="Model not loaded"):
        model_service.predict_batch(features_list)


def test_get_model_info_success(model_service, mock_model):
    """Test getting model information.

    Args:
        model_service: ModelService fixture
        mock_model: Mock ML model
    """
    model_service.model = mock_model
    model_service.model_version = "3"

    info = model_service.get_model_info()

    # Verify info structure
    assert info["model_name"] == "test_model"
    assert info["model_version"] == "3"
    assert info["model_stage"] == "Staging"
    assert info["features"] == ["employed", "bank_balance", "annual_salary"]
    assert info["engineered_features"] == ["Saving_Rate"]
    assert "model_type" in info


def test_get_model_info_without_model(model_service):
    """Test getting model info when model is not loaded.

    Args:
        model_service: ModelService fixture
    """
    with pytest.raises(ValueError, match="Model not loaded"):
        model_service.get_model_info()


def test_predict_batch_with_zero_salary(model_service, mock_model, mock_scaler):
    """Test batch prediction with zero salary edge case.

    Args:
        model_service: ModelService fixture
        mock_model: Mock ML model
        mock_scaler: Mock scaler
    """
    model_service.model = mock_model
    model_service.scaler = mock_scaler

    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    features_list = [
        {"employed": 0, "bank_balance": 500.0, "annual_salary": 0.0},
    ]

    results = model_service.predict_batch(features_list)

    # Should handle zero salary gracefully
    assert len(results) == 1
    assert results[0]["prediction"] == 1
    assert results[0]["probability"] == 0.8
    assert results[0]["default_risk"] == "High"
