"""Application configuration using Pydantic Settings.

This module defines all configuration settings for the application, loaded from environment
variables with validation.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StagingCriteria(BaseModel):
    """Model promotion criteria for Staging stage."""

    accuracy: float = Field(default=0.85, description="Minimum accuracy for staging")
    recall: float = Field(default=0.70, description="Minimum recall for staging")
    f1: float = Field(default=0.25, description="Minimum F1 score for staging")
    cv_recall_mean: float = Field(
        default=0.65, description="Minimum cross-validation recall mean for staging"
    )


class ProductionCriteria(BaseModel):
    """Model promotion criteria for Production stage."""

    accuracy: float = Field(default=0.88, description="Minimum accuracy for production")
    recall: float = Field(default=0.75, description="Minimum recall for production")
    f1: float = Field(default=0.30, description="Minimum F1 score for production")
    roc_auc: float = Field(default=0.85, description="Minimum ROC-AUC for production")
    pr_auc: float = Field(default=0.35, description="Minimum PR-AUC for production")
    cv_recall_std_max: float = Field(
        default=0.05, description="Maximum cross-validation recall std for production"
    )


class XGBoostParams(BaseModel):
    """XGBoost hyperparameters."""

    n_estimators: int = Field(default=100, description="Number of boosting rounds")
    max_depth: int = Field(default=5, description="Maximum tree depth")
    learning_rate: float = Field(default=0.1, description="Learning rate")
    subsample: float = Field(default=0.8, description="Subsample ratio")
    colsample_bytree: float = Field(default=0.8, description="Column subsample ratio")
    random_state: int = Field(default=42, description="Random state")
    eval_metric: str = Field(default="logloss", description="Evaluation metric")
    use_label_encoder: bool = Field(default=False, description="Use label encoder")


class TrainingConfig(BaseModel):
    """Complete training pipeline configuration."""

    # Data configuration
    data_filename: str = Field(
        default="Default_Fin.csv", description="Training data CSV filename"
    )
    train_size: int = Field(default=8000, description="Training set size")
    test_size: int = Field(default=2000, description="Test set size")
    random_state: int = Field(default=42, description="Random state for reproducibility")

    # Features
    feature_columns: List[str] = Field(
        default=["Employed", "Bank Balance", "Annual Salary", "Saving_Rate"],
        description="Feature column names",
    )
    target_column: str = Field(default="Defaulted?", description="Target column name")

    # Preprocessing
    use_smote: bool = Field(default=True, description="Whether to use SMOTE for class balancing")
    smote_random_state: int = Field(default=42, description="SMOTE random state")

    # Cross-validation
    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    cv_scoring: str = Field(default="recall", description="Cross-validation scoring metric")
    cv_n_jobs: int = Field(default=-1, description="Number of parallel jobs for CV (-1 = all cores)")

    # Model configurations
    xgboost_params: XGBoostParams = Field(
        default_factory=XGBoostParams, description="XGBoost hyperparameters"
    )

    # Promotion criteria
    staging_criteria: StagingCriteria = Field(
        default_factory=StagingCriteria, description="Staging promotion criteria"
    )
    production_criteria: ProductionCriteria = Field(
        default_factory=ProductionCriteria, description="Production promotion criteria"
    )

    # Visualization
    figure_dpi: int = Field(default=100, description="DPI for saved figures")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application Settings
    app_name: str = Field(default="loan-default-prediction", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development/production)")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")
    api_key: str = Field(default="your-secret-api-key", description="API authentication key")

    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="file:///app/mlflow", description="MLflow tracking URI"
    )
    mlflow_experiment_name: str = Field(
        default="loan-default-prediction", description="MLflow experiment name"
    )
    model_name: str = Field(default="loan_default_model", description="Model name in registry")
    model_stage: str = Field(
        default="Production", description="Model stage to load (Production/Staging)"
    )

    # Redis Configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", description="Celery broker URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0", description="Celery result backend URL"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=100, description="Maximum requests per minute per client"
    )

    # Drift Detection
    drift_reference_size: int = Field(
        default=1000, description="Number of samples for reference distribution"
    )
    drift_window_size: int = Field(
        default=100, description="Window size for drift detection"
    )
    drift_psi_threshold: float = Field(
        default=0.15, description="PSI threshold for drift alert"
    )
    drift_sampling_rate: float = Field(
        default=0.1, description="Probability of checking drift on each request (0.1 = 10%)"
    )
    drift_batch_sampling_rate: float = Field(
        default=0.05, description="Probability of checking drift on batch predictions (0.05 = 5%)"
    )

    # Model Thresholds
    production_min_accuracy: float = Field(default=0.88, description="Minimum accuracy for production")
    production_min_recall: float = Field(default=0.75, description="Minimum recall for production")
    production_min_f1: float = Field(default=0.30, description="Minimum F1 score for production")
    production_min_roc_auc: float = Field(default=0.85, description="Minimum ROC-AUC for production")
    production_min_pr_auc: float = Field(default=0.35, description="Minimum PR-AUC for production")

    # Training Configuration
    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training pipeline configuration"
    )


# Create singleton instance
settings = Settings()
