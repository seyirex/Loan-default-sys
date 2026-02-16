"""Drift detection service using Population Stability Index (PSI).

This module implements PSI-based drift detection to monitor feature distribution changes.
"""

from collections import deque
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from src.config import settings


class DriftDetector:
    """Service for detecting data drift using PSI (Population Stability Index).

    PSI measures the change in distribution between a reference dataset and current data.
    PSI < 0.1: No drift
    PSI 0.1-0.2: Moderate drift
    PSI > 0.2: Significant drift

    Attributes:
        reference_data: Reference distribution (first N observations)
        current_window: Rolling window of recent observations
        reference_size: Size of reference dataset
        window_size: Size of rolling window
        psi_threshold: Threshold for drift alert
    """

    def __init__(
        self,
        reference_size: int = settings.drift_reference_size,
        window_size: int = settings.drift_window_size,
        psi_threshold: float = settings.drift_psi_threshold,
    ):
        """Initialize DriftDetector.

        Args:
            reference_size: Number of samples for reference distribution
            window_size: Size of rolling window for current data
            psi_threshold: PSI threshold for drift alert
        """
        self.reference_data: deque = deque(maxlen=reference_size)
        self.current_window: deque = deque(maxlen=window_size)
        self.reference_size = reference_size
        self.window_size = window_size
        self.psi_threshold = psi_threshold

        logger.info(
            f"DriftDetector initialized - Reference: {reference_size}, Window: {window_size}, Threshold: {psi_threshold}"
        )

    def add_observation(self, features: Dict[str, float]) -> None:
        """Add a new observation to the detector.

        Observations are added to reference data until it's full, then to current window.

        Args:
            features: Dictionary of feature values
        """
        # Convert features dict to list of values in consistent order
        feature_values = [
            features.get("employed", 0),
            features.get("bank_balance", 0),
            features.get("annual_salary", 0),
            features.get("saving_rate", 0),
        ]

        # Fill reference data first
        if len(self.reference_data) < self.reference_size:
            self.reference_data.append(feature_values)
        else:
            # Then fill current window
            self.current_window.append(feature_values)

    def _calculate_psi(
        self, reference: np.ndarray, current: np.ndarray, num_bins: int = 10
    ) -> float:
        """Calculate Population Stability Index (PSI) for a single feature.

        PSI = Σ[(Current% - Reference%) × ln(Current% / Reference%)]

        Args:
            reference: Reference distribution values
            current: Current distribution values
            num_bins: Number of bins for discretization

        Returns:
            PSI value (float)
        """
        # Handle edge cases
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Create bins based on reference data
        bins = np.percentile(reference, np.linspace(0, 100, num_bins + 1))
        bins = np.unique(bins)  # Remove duplicates

        # Handle case where all values are the same
        if len(bins) <= 1:
            return 0.0

        # Bin the data
        ref_binned = np.histogram(reference, bins=bins)[0]
        cur_binned = np.histogram(current, bins=bins)[0]

        # Calculate percentages (with smoothing to avoid division by zero)
        ref_percent = (ref_binned + 0.001) / len(reference)
        cur_percent = (cur_binned + 0.001) / len(current)

        # Calculate PSI
        psi = np.sum((cur_percent - ref_percent) * np.log(cur_percent / ref_percent))

        return float(psi)

    def check_drift(self) -> Dict[str, any]:
        """Check for drift in current window vs reference data.

        Returns:
            Dictionary with drift metrics for each feature and overall status
        """
        # Need both reference data and current window
        if len(self.reference_data) < self.reference_size:
            return {
                "drift_detected": False,
                "message": f"Collecting reference data: {len(self.reference_data)}/{self.reference_size}",
                "psi_scores": {},
            }

        if len(self.current_window) < self.window_size:
            return {
                "drift_detected": False,
                "message": f"Collecting current window: {len(self.current_window)}/{self.window_size}",
                "psi_scores": {},
            }

        # Convert to numpy arrays
        reference_array = np.array(list(self.reference_data))
        current_array = np.array(list(self.current_window))

        # Feature names
        feature_names = ["employed", "bank_balance", "annual_salary", "saving_rate"]

        # Calculate PSI for each feature
        psi_scores = {}
        for i, feature_name in enumerate(feature_names):
            psi = self._calculate_psi(reference_array[:, i], current_array[:, i])
            psi_scores[feature_name] = round(psi, 4)

        # Check if any feature exceeds threshold
        max_psi = max(psi_scores.values())
        drift_detected = max_psi > self.psi_threshold

        result = {
            "drift_detected": drift_detected,
            "max_psi": round(max_psi, 4),
            "threshold": self.psi_threshold,
            "psi_scores": psi_scores,
            "message": f"Drift {'DETECTED' if drift_detected else 'not detected'} (max PSI: {max_psi:.4f})",
        }

        if drift_detected:
            logger.warning(f"Drift detected! {result}")
        else:
            logger.debug(f"No drift detected. PSI scores: {psi_scores}")

        return result

    def get_status(self) -> Dict[str, any]:
        """Get current status of drift detector.

        Returns:
            Dictionary with detector status and data collection progress
        """
        return {
            "reference_collected": len(self.reference_data),
            "reference_required": self.reference_size,
            "current_window_size": len(self.current_window),
            "window_required": self.window_size,
            "ready_for_detection": len(self.reference_data) >= self.reference_size
            and len(self.current_window) >= self.window_size,
        }
