"""Tests for training service.

This module tests the model training pipeline functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.services.training_service import TrainingService
from src.utils.preprocessing import DataValidator, FeatureEngineeringTransformer


@pytest.fixture
def mock_training_data():
    """Create mock training data.

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    np.random.seed(42)
    n_samples = 1000

    data = {
        "Employed": np.random.choice([0, 1], n_samples),
        "Bank Balance": np.abs(np.random.normal(10000, 5000, n_samples)),
        "Annual Salary": np.abs(np.random.normal(50000, 20000, n_samples)) + 1,
        "Defaulted?": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    }

    df = pd.DataFrame(data)
    X = df[["Employed", "Bank Balance", "Annual Salary"]]
    y = df["Defaulted?"]

    return X, y


@pytest.fixture
def mock_csv_file(tmp_path, mock_training_data):
    """Create a temporary CSV file with mock training data.

    Args:
        tmp_path: Pytest temporary directory
        mock_training_data: Mock training data fixture

    Returns:
        Path to temporary CSV file
    """
    X, y = mock_training_data

    # Combine features and target
    df = X.copy()
    df["Defaulted?"] = y

    # Save to CSV
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def training_service(mock_csv_file):
    """Create TrainingService instance for testing.

    Args:
        mock_csv_file: Path to temporary CSV file

    Returns:
        TrainingService instance
    """
    return TrainingService(
        data_path=mock_csv_file,
        mlflow_uri="sqlite:///test_mlflow.db",
        experiment_name="test_experiment",
        model_name="test_model",
    )


def test_training_service_initialization(training_service):
    """Test TrainingService initialization."""
    assert training_service.model is None
    assert training_service.preprocessing_pipeline is None
    assert training_service.metrics == {}
    assert training_service.run_id is None


def test_load_data(training_service):
    """Test data loading."""
    X, y = training_service.load_data()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert list(X.columns) == ["Employed", "Bank Balance", "Annual Salary"]


def test_load_data_file_not_found():
    """Test data loading with non-existent file."""
    service = TrainingService(
        data_path=Path("/nonexistent/path.csv"),
        mlflow_uri="sqlite:///test.db",
    )

    with pytest.raises(FileNotFoundError):
        service.load_data()


def test_split_data(training_service, mock_training_data):
    """Test data splitting."""
    X, y = mock_training_data

    # Override config for testing
    training_service.config.train_size = 800
    training_service.config.test_size = 200

    X_train, X_test, y_train, y_test = training_service.split_data(X, y)

    assert len(X_train) == 800
    assert len(X_test) == 200
    assert len(y_train) == 800
    assert len(y_test) == 200


def test_create_preprocessing_pipeline(training_service):
    """Test preprocessing pipeline creation."""
    pipeline = training_service.create_preprocessing_pipeline()

    assert pipeline is not None
    assert len(pipeline.steps) == 3
    assert "validator" in pipeline.named_steps
    assert "feature_engineering" in pipeline.named_steps
    assert "scaler" in pipeline.named_steps


def test_feature_engineering_transformer():
    """Test FeatureEngineeringTransformer."""
    transformer = FeatureEngineeringTransformer()

    X = pd.DataFrame(
        {
            "Employed": [1, 0, 1],
            "Bank Balance": [10000, 5000, 15000],
            "Annual Salary": [50000, 40000, 60000],
        }
    )

    X_transformed = transformer.fit_transform(X)

    assert "Saving_Rate" in X_transformed.columns
    assert len(X_transformed.columns) == 4
    assert X_transformed["Saving_Rate"].iloc[0] == pytest.approx(0.2, rel=0.01)


def test_feature_engineering_transformer_handles_zero_salary():
    """Test FeatureEngineeringTransformer handles zero salary."""
    transformer = FeatureEngineeringTransformer()

    X = pd.DataFrame(
        {
            "Employed": [1],
            "Bank Balance": [10000],
            "Annual Salary": [0],  # Zero salary
        }
    )

    X_transformed = transformer.fit_transform(X)

    # Should handle division by zero
    assert X_transformed["Saving_Rate"].iloc[0] == 0.0


def test_data_validator_valid_data():
    """Test DataValidator with valid data."""
    validator = DataValidator(check_ranges=True)

    X = pd.DataFrame(
        {
            "Employed": [1, 0, 1],
            "Bank Balance": [10000, 5000, 15000],
            "Annual Salary": [50000, 40000, 60000],
        }
    )

    # Should not raise
    validator.fit(X)
    X_validated = validator.transform(X)

    assert len(X_validated) == 3


def test_data_validator_negative_balance():
    """Test DataValidator rejects negative balance."""
    validator = DataValidator(check_ranges=True)

    X_train = pd.DataFrame(
        {
            "Employed": [1],
            "Bank Balance": [10000],
            "Annual Salary": [50000],
        }
    )

    X_test = pd.DataFrame(
        {
            "Employed": [1],
            "Bank Balance": [-5000],  # Negative!
            "Annual Salary": [50000],
        }
    )

    validator.fit(X_train)

    with pytest.raises(ValueError, match="Negative bank balance"):
        validator.transform(X_test)


def test_data_validator_missing_values():
    """Test DataValidator rejects missing values."""
    validator = DataValidator(check_ranges=True)

    X_train = pd.DataFrame(
        {
            "Employed": [1],
            "Bank Balance": [10000],
            "Annual Salary": [50000],
        }
    )

    X_test = pd.DataFrame(
        {
            "Employed": [1],
            "Bank Balance": [np.nan],  # Missing!
            "Annual Salary": [50000],
        }
    )

    validator.fit(X_train)

    with pytest.raises(ValueError, match="Missing values"):
        validator.transform(X_test)


def test_check_promotion_criteria_production(training_service):
    """Test promotion criteria checking for production-level metrics."""
    # Metrics that meet production criteria
    metrics_prod = {
        "accuracy": 0.90,
        "recall": 0.80,
        "f1": 0.35,
        "roc_auc": 0.88,
        "pr_auc": 0.40,
        "cv_recall_mean": 0.75,
        "cv_recall_std": 0.02,
    }

    meets_staging, meets_production = training_service.check_promotion_criteria(
        metrics_prod
    )

    assert meets_staging is True
    assert meets_production is True


def test_check_promotion_criteria_staging_only(training_service):
    """Test promotion criteria checking for staging-only metrics."""
    # Metrics that meet only staging criteria
    metrics_staging = {
        "accuracy": 0.86,
        "recall": 0.72,
        "f1": 0.28,
        "roc_auc": 0.82,
        "pr_auc": 0.30,
        "cv_recall_mean": 0.70,
        "cv_recall_std": 0.08,
    }

    meets_staging, meets_production = training_service.check_promotion_criteria(
        metrics_staging
    )

    assert meets_staging is True
    assert meets_production is False


def test_check_promotion_criteria_neither(training_service):
    """Test promotion criteria checking for insufficient metrics."""
    # Metrics that meet neither staging nor production
    metrics_poor = {
        "accuracy": 0.80,
        "recall": 0.60,
        "f1": 0.20,
        "roc_auc": 0.75,
        "pr_auc": 0.25,
        "cv_recall_mean": 0.60,
        "cv_recall_std": 0.10,
    }

    meets_staging, meets_production = training_service.check_promotion_criteria(
        metrics_poor
    )

    assert meets_staging is False
    assert meets_production is False


@patch("src.services.training_service.mlflow")
def test_promote_model_to_production(mock_mlflow, training_service):
    """Test model promotion to production."""
    mock_model_version = Mock()
    mock_model_version.version = 1

    mock_client = MagicMock()
    mock_mlflow.MlflowClient.return_value = mock_client

    stage = training_service.promote_model(
        mock_model_version, meets_staging=True, meets_production=True
    )

    assert stage == "Production"
    mock_client.transition_model_version_stage.assert_called_once_with(
        name=training_service.model_name,
        version=1,
        stage="Production",
        archive_existing_versions=True,
    )


@patch("src.services.training_service.mlflow")
def test_promote_model_to_staging(mock_mlflow, training_service):
    """Test model promotion to staging."""
    mock_model_version = Mock()
    mock_model_version.version = 1

    mock_client = MagicMock()
    mock_mlflow.MlflowClient.return_value = mock_client

    stage = training_service.promote_model(
        mock_model_version, meets_staging=True, meets_production=False
    )

    assert stage == "Staging"
    mock_client.transition_model_version_stage.assert_called_once_with(
        name=training_service.model_name,
        version=1,
        stage="Staging",
        archive_existing_versions=False,
    )


@patch("src.services.training_service.mlflow")
def test_promote_model_none(mock_mlflow, training_service):
    """Test model not promoted when criteria not met."""
    mock_model_version = Mock()
    mock_model_version.version = 1

    mock_client = MagicMock()
    mock_mlflow.MlflowClient.return_value = mock_client

    stage = training_service.promote_model(
        mock_model_version, meets_staging=False, meets_production=False
    )

    assert stage == "None"
    # Should not call transition
    mock_client.transition_model_version_stage.assert_not_called()


def test_preprocess_data(training_service, mock_training_data):
    """Test data preprocessing."""
    X, y = mock_training_data

    # Split first
    X_train, X_test, y_train, y_test = training_service.split_data(X, y)

    # Preprocess
    X_train_proc, y_train_proc, X_test_proc = training_service.preprocess_data(
        X_train, y_train, X_test
    )

    # Check preprocessing pipeline was created
    assert training_service.preprocessing_pipeline is not None

    # Check shapes - SMOTE will balance classes
    assert X_train_proc.shape[1] == 4  # 4 features (including Saving_Rate)
    assert X_test_proc.shape[1] == 4
    assert len(y_train_proc) == len(X_train_proc)

    # Check SMOTE was applied (classes should be balanced)
    if training_service.config.use_smote:
        assert len(y_train_proc) > len(y_train)  # More samples after SMOTE


def test_train_model(training_service, mock_training_data):
    """Test model training."""
    X, y = mock_training_data

    # Split and preprocess
    X_train, X_test, y_train, y_test = training_service.split_data(X, y)
    X_train_proc, y_train_proc, X_test_proc = training_service.preprocess_data(
        X_train, y_train, X_test
    )

    # Train
    model = training_service.train_model(X_train_proc, y_train_proc)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
