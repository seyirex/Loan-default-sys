"""Production training script entry point.

This script provides a simple CLI interface to the TrainingService.
All training logic is in src.services.training_service.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import logger  # noqa: E402
from src.services.training_service import TrainingService  # noqa: E402


def main():
    """Run the training pipeline."""
    try:
        # Initialize training service
        service = TrainingService()

        # Execute training pipeline
        results = service.run()

        # Exit with success
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
