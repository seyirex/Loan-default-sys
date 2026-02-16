"""Training service for loan default prediction model.

This module provides production-ready model training with MLflow integration,
comprehensive evaluation, and automated model promotion.

"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import settings
from src.utils.preprocessing import DataValidator, FeatureEngineeringTransformer

warnings.filterwarnings("ignore")


class TrainingService:
    """Service for training and evaluating loan default prediction models.

    This service orchestrates the entire training pipeline including:
    - Data loading and validation
    - Feature engineering
    - Preprocessing with sklearn Pipeline
    - Model training with XGBoost
    - Comprehensive evaluation with cross-validation
    - MLflow experiment tracking and model registry
    - Automated model promotion based on performance criteria

    Attributes:
        data_path: Path to training data CSV
        config: Training configuration from settings
        mlflow_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        model_name: Model name in registry
        preprocessing_pipeline: Sklearn pipeline for preprocessing
        model: Trained XGBoost model
        run_id: Current MLflow run ID
        metrics: Dictionary of evaluation metrics
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        mlflow_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize TrainingService.

        Args:
            data_path: Path to training data (default: from config)
            mlflow_uri: MLflow tracking URI (default: from settings)
            experiment_name: MLflow experiment name (default: from settings)
            model_name: Model registry name (default: from settings)
        """
        # Configuration
        self.config = settings.training
        self.data_path = data_path or (
            Path(__file__).parent.parent.parent / "training" / self.config.data_filename
        )
        self.mlflow_uri = mlflow_uri or settings.mlflow_tracking_uri
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self.model_name = model_name or settings.model_name

        # Pipeline components
        self.preprocessing_pipeline: Optional[Pipeline] = None
        self.model: Optional[XGBClassifier] = None

        # Tracking
        self.run_id: Optional[str] = None
        self.metrics: Dict[str, float] = {}

        # Data holders
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.X_train_processed: Optional[np.ndarray] = None
        self.X_test_processed: Optional[np.ndarray] = None
        self.y_train_balanced: Optional[np.ndarray] = None

        logger.info("TrainingService initialized")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"MLflow URI: {self.mlflow_uri}")
        logger.info(f"Experiment: {self.experiment_name}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and validate training data.

        Returns:
            Tuple of (features DataFrame, target Series)

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        logger.info(f"Loading data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # TODO: Add data versioning with DVC
        # - Track data file versions with `dvc add training/Default_Fin.csv`
        # - Log data hash/version to MLflow for reproducibility
        # - Support loading specific data versions for experimentation

        # Load CSV
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.debug(f"Columns: {df.columns.tolist()}")

        # Drop Index column if present
        if "Index" in df.columns:
            df = df.drop("Index", axis=1)
            logger.debug("Dropped Index column")

        # Validate required columns
        required_cols = [
            "Employed",
            "Bank Balance",
            "Annual Salary",
            self.config.target_column,
        ]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Extract features and target
        X = df[["Employed", "Bank Balance", "Annual Salary"]]
        y = df[self.config.target_column]

        # Log class distribution
        class_counts = y.value_counts()
        logger.info(f"Target distribution:\n{class_counts.to_dict()}")
        imbalance_ratio = class_counts[0] / class_counts[1]
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")

        return X, y

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.

        Args:
            X: Features
            y: Target

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train/test sets")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=self.config.train_size,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.debug(f"Train class distribution: {y_train.value_counts().to_dict()}")
        logger.debug(f"Test class distribution: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def create_preprocessing_pipeline(self) -> Pipeline:
        """Create sklearn preprocessing pipeline.

        Returns:
            Configured sklearn Pipeline
        """
        logger.info("Creating preprocessing pipeline")

        pipeline = Pipeline(
            [
                ("validator", DataValidator(check_ranges=True)),
                ("feature_engineering", FeatureEngineeringTransformer()),
                ("scaler", StandardScaler()),
            ]
        )

        logger.debug(f"Pipeline steps: {[name for name, _ in pipeline.steps]}")
        return pipeline

    def preprocess_data(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data using pipeline and apply SMOTE.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features

        Returns:
            Tuple of (X_train_balanced, y_train_balanced, X_test_processed)
        """
        logger.info("Preprocessing data")

        # Create and fit pipeline on training data
        self.preprocessing_pipeline = self.create_preprocessing_pipeline()
        X_train_processed = self.preprocessing_pipeline.fit_transform(X_train)
        logger.info("Fitted preprocessing pipeline on training data")

        # Transform test data
        X_test_processed = self.preprocessing_pipeline.transform(X_test)
        logger.info("Transformed test data")

        # Apply SMOTE if enabled
        if self.config.use_smote:
            logger.info("Applying SMOTE for class balancing")
            smote = SMOTE(random_state=self.config.smote_random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(
                X_train_processed, y_train
            )

            logger.info(f"After SMOTE - Training samples: {len(X_train_balanced)}")
            logger.info(f"Class distribution: {np.bincount(y_train_balanced).tolist()}")
        else:
            logger.info("SMOTE disabled, using original training data")
            X_train_balanced = X_train_processed
            y_train_balanced = y_train.values

        return X_train_balanced, y_train_balanced, X_test_processed

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        """Train XGBoost model.

        Args:
            X_train: Preprocessed training features
            y_train: Training target

        Returns:
            Trained XGBoost classifier
        """
        logger.info("Training XGBoost model")

        # TODO: Add hyperparameter tuning with Optuna
        # - Add config flag: enable_hyperparameter_tuning: bool
        # - Implement tune_hyperparameters() method using Optuna
        # - Log best hyperparameters and tuning history to MLflow
        # - Example: optuna.create_study(direction="maximize", study_name="xgboost_tuning")

        # TODO: Support multiple model types
        # - Add config field: model_type: str = "xgboost" | "lightgbm" | "randomforest" | "catboost"
        # - Implement _create_model() factory method based on model_type
        # - Add LightGBMParams, RandomForestParams config classes
        # - Log model type to MLflow for easy comparison

        # Get hyperparameters from config
        params = self.config.xgboost_params.model_dump()
        logger.debug(f"Hyperparameters: {params}")

        # Train model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        logger.info("Model training complete")
        return model

    def evaluate_model(
        self,
        model: XGBClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model and calculate all metrics.

        Args:
            model: Trained model
            X_train: Training features (for CV)
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "pr_auc": average_precision_score(y_test, y_pred_proba),
        }

        # Cross-validation scores
        logger.info("Performing cross-validation")
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=self.config.cv_folds,
            scoring=self.config.cv_scoring,
            n_jobs=self.config.cv_n_jobs,
        )
        metrics["cv_recall_mean"] = cv_scores.mean()
        metrics["cv_recall_std"] = cv_scores.std()

        logger.info("Model Performance Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name:20s}: {value:.4f}")

        return metrics

    def create_visualizations(
        self, y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, plt.Figure]:
        """Create all visualization plots.

        Args:
            y_test: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of figure name to matplotlib Figure
        """
        logger.info("Creating visualizations")
        figures = {}

        # Confusion Matrix
        logger.debug("Creating confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(8, 6), dpi=self.config.figure_dpi)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        figures["confusion_matrix"] = fig_cm

        # ROC Curve
        logger.debug("Creating ROC curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc_val = auc(fpr, tpr)
        fig_roc, ax = plt.subplots(figsize=(8, 6), dpi=self.config.figure_dpi)
        ax.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc_val:.2f})"
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures["roc_curve"] = fig_roc

        # Feature Importance
        logger.debug("Creating feature importance plot")
        importance = self.model.feature_importances_
        feature_names = self.config.feature_columns
        indices = np.argsort(importance)[::-1]
        fig_fi, ax = plt.subplots(figsize=(10, 6), dpi=self.config.figure_dpi)
        ax.bar(range(len(importance)), importance[indices])
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels(
            [feature_names[i] for i in indices], rotation=45, ha="right"
        )
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance")
        plt.tight_layout()
        figures["feature_importance"] = fig_fi

        logger.info(f"Created {len(figures)} visualizations")
        return figures

    def log_to_mlflow(self, figures: Dict[str, plt.Figure]) -> str:
        """Log all artifacts to MLflow.

        Args:
            figures: Dictionary of visualization figures

        Returns:
            MLflow run ID
        """
        logger.info("Logging to MLflow")

        # Log parameters
        params = {
            **self.config.xgboost_params.model_dump(),
            "train_size": self.config.train_size,
            "test_size": self.config.test_size,
            "smote_applied": self.config.use_smote,
            "cv_folds": self.config.cv_folds,
        }
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")

        # Log metrics
        mlflow.log_metrics(self.metrics)
        logger.debug(f"Logged {len(self.metrics)} metrics")

        # Log visualizations
        for name, fig in figures.items():
            mlflow.log_figure(fig, f"{name}.png")
            plt.close(fig)
        logger.debug(f"Logged {len(figures)} figures")

        # Log model and preprocessing pipeline
        logger.info("Logging model and preprocessing artifacts")
        mlflow.sklearn.log_model(self.model, "model")

        # Extract and log scaler separately for compatibility
        scaler = self.preprocessing_pipeline.named_steps["scaler"]
        mlflow.sklearn.log_model(scaler, "scaler")

        # Log full preprocessing pipeline
        mlflow.sklearn.log_model(self.preprocessing_pipeline, "preprocessing_pipeline")

        logger.info("MLflow logging complete")
        return self.run_id

    def check_promotion_criteria(
        self, metrics: Dict[str, float]
    ) -> Tuple[bool, bool]:
        """Check if model meets promotion criteria.

        Args:
            metrics: Dictionary of evaluation metrics

        Returns:
            Tuple of (meets_staging, meets_production)
        """
        logger.info("Checking promotion criteria")

        # Check STAGING criteria
        staging_config = self.config.staging_criteria
        meets_staging = (
            metrics["accuracy"] >= staging_config.accuracy
            and metrics["recall"] >= staging_config.recall
            and metrics["f1"] >= staging_config.f1
            and metrics["cv_recall_mean"] >= staging_config.cv_recall_mean
        )

        logger.debug(
            f"Staging criteria - Accuracy: {metrics['accuracy']:.4f} >= {staging_config.accuracy}"
        )
        logger.debug(
            f"Staging criteria - Recall: {metrics['recall']:.4f} >= {staging_config.recall}"
        )
        logger.debug(
            f"Staging criteria - F1: {metrics['f1']:.4f} >= {staging_config.f1}"
        )
        logger.debug(
            f"Staging criteria - CV Recall: {metrics['cv_recall_mean']:.4f} >= {staging_config.cv_recall_mean}"
        )

        # Check PRODUCTION criteria
        prod_config = self.config.production_criteria
        meets_production = (
            metrics["accuracy"] >= prod_config.accuracy
            and metrics["recall"] >= prod_config.recall
            and metrics["f1"] >= prod_config.f1
            and metrics["roc_auc"] >= prod_config.roc_auc
            and metrics["pr_auc"] >= prod_config.pr_auc
            and metrics["cv_recall_std"] < prod_config.cv_recall_std_max
        )

        logger.debug(
            f"Production criteria - Accuracy: {metrics['accuracy']:.4f} >= {prod_config.accuracy}"
        )
        logger.debug(
            f"Production criteria - Recall: {metrics['recall']:.4f} >= {prod_config.recall}"
        )
        logger.debug(
            f"Production criteria - F1: {metrics['f1']:.4f} >= {prod_config.f1}"
        )
        logger.debug(
            f"Production criteria - ROC-AUC: {metrics['roc_auc']:.4f} >= {prod_config.roc_auc}"
        )
        logger.debug(
            f"Production criteria - PR-AUC: {metrics['pr_auc']:.4f} >= {prod_config.pr_auc}"
        )
        logger.debug(
            f"Production criteria - CV Std: {metrics['cv_recall_std']:.4f} < {prod_config.cv_recall_std_max}"
        )

        logger.info(f"Meets STAGING: {meets_staging}")
        logger.info(f"Meets PRODUCTION: {meets_production}")

        return meets_staging, meets_production

    def promote_model(
        self, model_version: Any, meets_staging: bool, meets_production: bool
    ) -> str:
        """Promote model to appropriate stage.

        Args:
            model_version: MLflow model version object
            meets_staging: Whether model meets staging criteria
            meets_production: Whether model meets production criteria

        Returns:
            Promotion stage (Production/Staging/None)
        """
        logger.info("Evaluating model promotion")

        client = mlflow.MlflowClient()

        if meets_production:
            logger.info(
                "Model meets PRODUCTION criteria - promoting to Production"
            )
            client.transition_model_version_stage(
                name=self.model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True,
            )
            mlflow.set_tag("promotion_stage", "Production")
            return "Production"

        elif meets_staging:
            logger.warning(
                "Model meets STAGING criteria only - promoting to Staging"
            )
            client.transition_model_version_stage(
                name=self.model_name,
                version=model_version.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            mlflow.set_tag("promotion_stage", "Staging")
            return "Staging"

        else:
            logger.error(
                "Model does NOT meet promotion criteria - keeping in None stage"
            )
            mlflow.set_tag("promotion_stage", "None")
            return "None"

    def run(self) -> Dict[str, Any]:
        """Execute complete training pipeline.

        Returns:
            Dictionary with training results and metadata

        Raises:
            Exception: If any step in the pipeline fails
        """
        # TODO: Add CI/CD integration for automated training
        # - Create GitHub Actions / GitLab CI workflow for training
        # - Trigger training on: new data upload, scheduled intervals, manual dispatch
        # - Run tests before training, promote model only if tests pass
        # - Notify team on Slack/email when training completes or fails
        # - Example: .github/workflows/train-model.yml

        # TODO: Create training schedule with Airflow/Prefect
        # - Set up DAG for periodic retraining (daily/weekly/monthly)
        # - Add data quality checks before training
        # - Monitor data drift and trigger retraining automatically
        # - Orchestrate: data fetch → validation → training → testing → deployment
        # - Example: airflow/dags/loan_default_training_dag.py

        try:
            logger.info("=" * 80)
            logger.info("STARTING TRAINING PIPELINE")
            logger.info("=" * 80)

            # Setup MLflow
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment: {self.experiment_name}")

            # Start MLflow run
            with mlflow.start_run() as run:
                self.run_id = run.info.run_id
                logger.info(f"MLflow Run ID: {self.run_id}")

                # 1. Load data
                X, y = self.load_data()

                # 2. Split data
                self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
                    X, y
                )

                # 3. Preprocess data
                (
                    self.X_train_processed,
                    self.y_train_balanced,
                    self.X_test_processed,
                ) = self.preprocess_data(self.X_train, self.y_train, self.X_test)

                # 4. Train model
                self.model = self.train_model(
                    self.X_train_processed, self.y_train_balanced
                )

                # 5. Evaluate model
                self.metrics = self.evaluate_model(
                    self.model,
                    self.X_train_processed,
                    self.y_train_balanced,
                    self.X_test_processed,
                    self.y_test,
                )

                # 6. Create visualizations
                y_pred = self.model.predict(self.X_test_processed)
                y_pred_proba = self.model.predict_proba(self.X_test_processed)[:, 1]
                figures = self.create_visualizations(
                    self.y_test, y_pred, y_pred_proba
                )

                # 7. Log to MLflow
                self.log_to_mlflow(figures)

                # 8. Register model
                logger.info("Registering model in MLflow Model Registry")
                model_uri = f"runs:/{self.run_id}/model"
                model_version = mlflow.register_model(model_uri, self.model_name)
                logger.info(
                    f"Model registered: {self.model_name} version {model_version.version}"
                )

                # 9. Check promotion and promote
                meets_staging, meets_production = self.check_promotion_criteria(
                    self.metrics
                )
                promotion_stage = self.promote_model(
                    model_version, meets_staging, meets_production
                )

                # Prepare results
                results = {
                    "run_id": self.run_id,
                    "model_name": self.model_name,
                    "model_version": model_version.version,
                    "promotion_stage": promotion_stage,
                    "metrics": self.metrics,
                    "meets_staging": meets_staging,
                    "meets_production": meets_production,
                }

                logger.info("=" * 80)
                logger.info("TRAINING PIPELINE COMPLETE")
                logger.info("=" * 80)
                logger.info(f"Run ID: {self.run_id}")
                logger.info(f"Model: {self.model_name} v{model_version.version}")
                logger.info(f"Stage: {promotion_stage}")
                logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
                logger.info(f"Recall: {self.metrics['recall']:.4f}")
                logger.info(f"F1: {self.metrics['f1']:.4f}")
                logger.info(f"ROC-AUC: {self.metrics['roc_auc']:.4f}")
                logger.info("=" * 80)

                return results

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            logger.exception("Full traceback:")
            raise
