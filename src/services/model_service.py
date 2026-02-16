"""Model service for loading and making predictions.

This module handles MLflow model loading and prediction logic.
"""

from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger


class ModelService:
    """Service for managing ML model lifecycle and predictions.

    This service handles loading models from MLflow, preprocessing features,
    and making predictions.

    Attributes:
        mlflow_uri: MLflow tracking URI
        model_name: Name of the model in MLflow registry
        model_stage: Model stage (Production/Staging/None)
        model: Loaded XGBoost model
        scaler: Loaded StandardScaler for preprocessing
        model_version: Current model version number
    """

    def __init__(self, mlflow_uri: str, model_name: str, model_stage: str = "Production"):
        """Initialize ModelService.

        Args:
            mlflow_uri: MLflow tracking URI
            model_name: Name of the model in registry
            model_stage: Model stage to load (default: Production)
        """
        self.mlflow_uri = mlflow_uri
        self.model_name = model_name
        self.model_stage = model_stage

        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.model_version: Optional[str] = None

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_uri)

        logger.info(f"ModelService initialized with URI: {mlflow_uri}")

    def load_model(self) -> None:
        """Load model and scaler from MLflow Model Registry.

        Raises:
            Exception: If model or scaler cannot be loaded
        """
        try:
            logger.info(f"Loading model '{self.model_name}' from stage '{self.model_stage}'...")

            # Load model from registry
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            self.model = mlflow.sklearn.load_model(model_uri)

            # Get model version info
            client = mlflow.MlflowClient()
            model_versions = client.search_model_versions(f"name='{self.model_name}'")

            # Find the version in the desired stage
            for mv in model_versions:
                if mv.current_stage == self.model_stage:
                    self.model_version = str(mv.version)
                    run_id = mv.run_id
                    break

            if self.model_version is None:
                raise ValueError(f"No model found in stage '{self.model_stage}'")

            logger.info(f"Model version {self.model_version} loaded successfully")

            # Load scaler from the same run
            scaler_uri = f"runs:/{run_id}/scaler"
            self.scaler = mlflow.sklearn.load_model(scaler_uri)
            logger.info("Scaler loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _engineer_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Engineer features including Saving Rate calculation.

        Args:
            features: Dictionary of input features

        Returns:
            DataFrame with engineered features
        """
        # Extract base features
        employed = features["employed"]
        bank_balance = features["bank_balance"]
        annual_salary = features["annual_salary"]

        # Calculate Saving Rate (feature engineering)
        saving_rate = bank_balance / annual_salary if annual_salary > 0 else 0

        # Create DataFrame with all features in correct order
        df = pd.DataFrame(
            {
                "Employed": [employed],
                "Bank Balance": [bank_balance],
                "Annual Salary": [annual_salary],
                "Saving_Rate": [saving_rate],
            }
        )

        return df

    def preprocess(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess features for prediction.

        Args:
            features: Dictionary of input features

        Returns:
            Preprocessed feature array ready for prediction

        Raises:
            ValueError: If scaler is not loaded
        """
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Call load_model() first.")

        # Engineer features
        df = self._engineer_features(features)

        # Apply scaling
        features_scaled = self.scaler.transform(df)

        return features_scaled

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single prediction.

        Args:
            features: Dictionary with keys: employed, bank_balance, annual_salary

        Returns:
            Dictionary with prediction and probability

        Raises:
            ValueError: If model is not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Preprocess features
        features_processed = self.preprocess(features)

        # Make prediction
        prediction = int(self.model.predict(features_processed)[0])
        probability = float(self.model.predict_proba(features_processed)[0][1])

        result = {
            "prediction": prediction,
            "probability": round(probability, 4),
            "default_risk": "High" if prediction == 1 else "Low",
        }

        logger.debug(f"Prediction: {result}")
        return result

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make vectorized batch predictions for optimal performance.

        This method processes all features at once using vectorized operations,
        which is significantly faster than individual predictions (10-100x speedup).

        Args:
            features_list: List of feature dictionaries with keys:
                - employed: bool or int (1/0)
                - bank_balance: float
                - annual_salary: float

        Returns:
            List of prediction result dictionaries, one per input

        Raises:
            ValueError: If model is not loaded or features are invalid
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if not features_list:
            return []

        # Build feature DataFrame (vectorized feature engineering)
        all_features = []
        for features in features_list:
            employed = features["employed"]
            bank_balance = features["bank_balance"]
            annual_salary = features["annual_salary"]

            # Calculate Saving Rate
            saving_rate = bank_balance / annual_salary if annual_salary > 0 else 0

            all_features.append({
                "Employed": employed,
                "Bank Balance": bank_balance,
                "Annual Salary": annual_salary,
                "Saving_Rate": saving_rate,
            })

        # Create single DataFrame with all features
        df = pd.DataFrame(all_features)

        # Vectorized scaling (single operation for all features)
        features_scaled = self.scaler.transform(df)

        # Vectorized predictions (single model call for all features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]

        # Build results list
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "probability": round(float(prob), 4),
                "default_risk": "High" if int(pred) == 1 else "Low",
            })

        logger.debug(f"Batch prediction completed for {len(features_list)} items")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model.

        Returns:
            Dictionary with model metadata

        Raises:
            ValueError: If model is not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_stage": self.model_stage,
            "features": ["employed", "bank_balance", "annual_salary"],
            "engineered_features": ["Saving_Rate"],
            "model_type": type(self.model).__name__,
        }
