"""Prediction request and response schemas.

This module defines Pydantic models for prediction API endpoints.
"""

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class JobStatusEnum(str, Enum):
    """Enum for job statuses.

    These statuses match Celery task states and can be used for
    batch jobs, async predictions, or any background task:
    - PENDING: Job is queued but not yet started
    - PROGRESS: Job is currently being processed
    - SUCCESS: Job completed successfully
    - FAILURE: Job failed with an error
    """

    PENDING = "PENDING"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class PredictionRequest(BaseModel):
    """Request model for single prediction.

    Attributes:
        employed: Employment status (0 = unemployed, 1 = employed)
        bank_balance: Current bank balance in currency units
        annual_salary: Annual salary in currency units
    """

    employed: int = Field(..., ge=0, le=1, description="Employment status (0 or 1)", examples=[1])
    bank_balance: float = Field(
        ..., ge=0, description="Bank balance", examples=[10000.50]
    )
    annual_salary: float = Field(
        ..., gt=0, description="Annual salary (must be > 0)", examples=[50000.00]
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "employed": 1,
                    "bank_balance": 10000.50,
                    "annual_salary": 50000.00,
                }
            ]
        }


class PredictionResponse(BaseModel):
    """Response model for single prediction.

    Attributes:
        prediction: Predicted class (0 = no default, 1 = default)
        probability: Probability of default (0.0 to 1.0)
        default_risk: Risk level (Low/High)
        model_version: Version of model used for prediction
        features_used: Dictionary of features used in prediction
    """

    prediction: int = Field(..., description="Prediction (0 = no default, 1 = default)")
    probability: float = Field(..., description="Probability of default")
    default_risk: str = Field(..., description="Risk level (Low/High)")
    model_version: Optional[str] = Field(None, description="Model version")
    features_used: Optional[dict] = Field(None, description="Features used in prediction")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions.

    Attributes:
        predictions: List of prediction requests
    """

    predictions: List[PredictionRequest] = Field(
        ..., min_length=1, max_length=1000, description="List of predictions to process"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "predictions": [
                        {"employed": 1, "bank_balance": 10000, "annual_salary": 50000},
                        {"employed": 0, "bank_balance": 5000, "annual_salary": 30000},
                    ]
                }
            ]
        }


class BatchJobResponse(BaseModel):
    """Response model for batch job submission.

    Attributes:
        job_id: Unique identifier for the batch job
        status: Current status of the job (PENDING, PROGRESS, SUCCESS, FAILURE)
        message: Additional information
        total_predictions: Total number of predictions in the batch
    """

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatusEnum = Field(..., description="Job status")
    message: str = Field(..., description="Status message")
    total_predictions: int = Field(..., description="Number of predictions in batch")


class BatchJobStatus(BaseModel):
    """Response model for batch job status check.

    Attributes:
        job_id: Job identifier
        status: Job status (PENDING, PROGRESS, SUCCESS, FAILURE)
        progress: Progress percentage (0-100)
        total: Total number of predictions in batch
        completed: Number of completed predictions
        results: List of prediction results (available when SUCCESS)
        error: Error message (available when FAILURE)
    """

    job_id: str = Field(..., description="Job identifier")
    status: JobStatusEnum = Field(..., description="Job status")
    progress: Optional[float] = Field(None, description="Progress percentage")
    total: Optional[int] = Field(None, description="Total predictions")
    completed: Optional[int] = Field(None, description="Completed predictions")
    results: Optional[List[dict]] = Field(None, description="Prediction results")
    error: Optional[str] = Field(None, description="Error message if failed")


class ModelInfoResponse(BaseModel):
    """Response model for model information endpoint.

    Attributes:
        model_name: Name of the model
        model_version: Version number
        model_stage: Deployment stage (Production/Staging)
        features: List of input features
        engineered_features: List of engineered features
        model_type: Type of ML model (e.g., "RandomForest", "XGBoost")
    """

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_stage: str = Field(..., description="Model stage")
    features: List[str] = Field(..., description="Input features")
    engineered_features: List[str] = Field(..., description="Engineered features")
    model_type: str = Field(..., description="Model type")


class StandardResponse(BaseModel):
    """Standard response envelope for all API endpoints.

    Attributes:
        success: Indicates if the request was successful
        data: Response data
        error: Error message if any
        exceptionError: Exception details if any
        message: Additional message
    """
    
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    exceptionError: Optional[str] = None
    message: Optional[str] = None