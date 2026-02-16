"""Preprocessing utilities for feature validation and transformation.

This module provides helper functions for data preprocessing and sklearn-compatible transformers.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin



def calculate_saving_rate(bank_balance: float, annual_salary: float) -> float:
    """Calculate saving rate feature.

    Args:
        bank_balance: Bank balance amount
        annual_salary: Annual salary amount

    Returns:
        Saving rate (bank_balance / annual_salary)
    """
    if annual_salary <= 0:
        logger.warning("Annual salary is 0 or negative, returning saving rate of 0")
        return 0.0

    return bank_balance / annual_salary




class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for feature engineering.

    This transformer adds the Saving_Rate feature (Bank Balance / Annual Salary)
    and handles edge cases like division by zero and infinite values.

    Attributes:
        feature_names_: List of feature names after transformation
    """

    def __init__(self):
        """Initialize the transformer."""
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for stateless transformation).

        Args:
            X: Input features DataFrame
            y: Target (unused)

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features by adding Saving_Rate.

        Args:
            X: Input features with columns [Employed, Bank Balance, Annual Salary]

        Returns:
            Transformed features with added Saving_Rate column

        Raises:
            ValueError: If required columns are missing
        """
        logger.debug(f"Engineering features for {len(X)} samples")

        # Validate required columns
        required_cols = ["Employed", "Bank Balance", "Annual Salary"]
        missing = set(required_cols) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Create copy to avoid modifying original
        X_transformed = X.copy()

        # Calculate Saving_Rate
        X_transformed["Saving_Rate"] = (
            X_transformed["Bank Balance"] / X_transformed["Annual Salary"]
        )

        # Handle inf and nan values
        X_transformed["Saving_Rate"] = X_transformed["Saving_Rate"].replace(
            [np.inf, -np.inf], np.nan
        )
        X_transformed["Saving_Rate"] = X_transformed["Saving_Rate"].fillna(0)

        self.feature_names_ = X_transformed.columns.tolist()

        logger.debug(f"Feature engineering complete. Features: {self.feature_names_}")
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get feature names for output.

        Args:
            input_features: Ignored (for sklearn compatibility)

        Returns:
            List of feature names
        """
        return self.feature_names_ if self.feature_names_ else []


class DataValidator(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for validating data quality.

    Checks for missing values, data types, and value ranges.

    Attributes:
        check_ranges: Whether to validate value ranges
        column_types_: Expected data types for each column
        validation_errors_: List of validation errors found
    """

    def __init__(self, check_ranges: bool = True):
        """Initialize validator.

        Args:
            check_ranges: Whether to validate value ranges
        """
        self.check_ranges = check_ranges
        self.column_types_: Optional[Dict[str, Any]] = None
        self.validation_errors_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Learn expected data types from training data.

        Args:
            X: Training features
            y: Target (unused)

        Returns:
            self
        """
        self.column_types_ = X.dtypes.to_dict()
        logger.debug(f"Learned column types: {self.column_types_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate data and return unchanged if valid.

        Args:
            X: Features to validate

        Returns:
            Unchanged features if valid

        Raises:
            ValueError: If validation fails
        """
        self.validation_errors_ = []

        # Check for missing values
        if X.isnull().any().any():
            missing_cols = X.columns[X.isnull().any()].tolist()
            error_msg = f"Missing values found in columns: {missing_cols}"
            logger.error(error_msg)
            self.validation_errors_.append(error_msg)

        # Check for negative values where not allowed
        if self.check_ranges:
            if "Bank Balance" in X.columns and (X["Bank Balance"] < 0).any():
                error_msg = "Negative bank balance values found"
                logger.error(error_msg)
                self.validation_errors_.append(error_msg)

            if "Annual Salary" in X.columns and (X["Annual Salary"] <= 0).any():
                error_msg = "Non-positive annual salary values found"
                logger.error(error_msg)
                self.validation_errors_.append(error_msg)

        if self.validation_errors_:
            raise ValueError(f"Data validation failed: {self.validation_errors_}")

        logger.debug(f"Data validation passed for {len(X)} samples")
        return X