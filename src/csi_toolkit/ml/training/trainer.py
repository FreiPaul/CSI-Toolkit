"""Model training functionality for CSI Toolkit."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError(
        "scikit-learn is required for ML functionality. "
        "Install with: pip install -e '.[ml]'"
    )

from ..models.registry import registry as model_registry
from ..models.base import BaseModel
from ..metrics.registry import registry as metric_registry
from ...core.constants import (
    DEFAULT_MODEL_TYPE,
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_VAL_SPLIT,
    DEFAULT_TEST_SPLIT,
    DEFAULT_RANDOM_SEED,
    MODEL_DIR_PREFIX,
    MODEL_FILENAME,
    METADATA_FILENAME,
    TRAINING_LOG_FILENAME,
)


class ModelTrainer:
    """
    Handle training of ML models on CSI feature data.

    Manages the complete training pipeline:
    - Data loading and validation
    - Train/val/test splitting
    - Model instantiation and training
    - Model and metadata serialization
    """

    # Columns to exclude from feature extraction (metadata columns)
    METADATA_COLUMNS = ['window_id', 'start_seq', 'end_seq', 'label']

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        train_split: float = DEFAULT_TRAIN_SPLIT,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        random_seed: int = DEFAULT_RANDOM_SEED,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of model to train (e.g., 'mlp')
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            random_seed: Random seed for reproducibility
            model_params: Optional model hyperparameters to override defaults
        """
        # Validate splits sum to 1.0
        total_split = train_split + val_split + test_split
        if not (0.99 <= total_split <= 1.01):  # Allow small floating point errors
            raise ValueError(
                f"Train/val/test splits must sum to 1.0, got {total_split}"
            )

        self.model_type = model_type
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.model_params = model_params or {}

        # Will be populated during training
        self.model: Optional[BaseModel] = None
        self.feature_names: Optional[List[str]] = None
        self.class_names: Optional[List[int]] = None

    def load_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load feature data from CSV file.

        Args:
            csv_path: Path to CSV file with features and labels

        Returns:
            Tuple of (X, y, feature_names) where:
                X: Feature matrix of shape (n_samples, n_features)
                y: Label vector of shape (n_samples,)
                feature_names: List of feature column names

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If label column is missing or no features found
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError("CSV file is empty")

        # Check for label column
        if 'label' not in rows[0]:
            raise ValueError(
                "CSV file must contain a 'label' column. "
                "Use 'csi_toolkit process --labeled' to generate labeled features."
            )

        # Extract feature names (exclude metadata columns)
        all_columns = list(rows[0].keys())
        feature_names = [
            col for col in all_columns
            if col not in self.METADATA_COLUMNS
        ]

        if not feature_names:
            raise ValueError("No feature columns found in CSV")

        # Extract features and labels
        X = []
        y = []

        for row in rows:
            # Extract features
            features = []
            for col in feature_names:
                try:
                    features.append(float(row[col]))
                except (ValueError, KeyError):
                    raise ValueError(f"Invalid feature value in column '{col}'")

            # Extract label
            try:
                label = int(row['label'])
            except (ValueError, KeyError):
                raise ValueError("Invalid or missing label value")

            X.append(features)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        print(f"Loaded {len(X)} samples with {len(feature_names)} features")
        print(f"Classes: {sorted(np.unique(y).tolist())}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        return X, y, feature_names

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Feature matrix
            y: Label vector

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        test_size = self.test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y
        )

        # Second split: separate train and validation from remaining data
        val_size_adjusted = self.val_split / (self.train_split + self.val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_seed,
            stratify=y_temp
        )

        print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> BaseModel:
        """
        Train a model on the provided data.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for evaluation)
            y_val: Validation labels (optional, for evaluation)

        Returns:
            Trained model
        """
        print(f"\nTraining {self.model_type} model...")

        # Create model instance
        self.model = model_registry.create_model(self.model_type, **self.model_params)

        # Train model
        self.model.fit(X_train, y_train)

        print(f"Training complete!")

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = (y_val_pred == y_val).mean()
            print(f"Validation accuracy: {val_accuracy:.4f}")

        return self.model

    def create_model_directory(self, output_dir: Optional[str] = None) -> str:
        """
        Create a timestamped directory for model storage.

        Args:
            output_dir: Optional custom directory name. If None, auto-generates
                       timestamp-based name like "models/model_20250113_143022"

        Returns:
            Path to created directory
        """
        if output_dir is None:
            # Generate timestamped directory name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("models", f"{MODEL_DIR_PREFIX}_{timestamp}")

        # Create directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        return output_dir

    def save_model(
        self,
        output_dir: str,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Save trained model and metadata to directory.

        Args:
            output_dir: Directory where model should be saved
            X_val: Validation features (optional, for metadata)
            y_val: Validation labels (optional, for metadata)
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        # Save model
        model_path = os.path.join(output_dir, MODEL_FILENAME)
        self.model.save(model_path)
        print(f"Saved model to: {model_path}")

        # Compute validation metrics if available
        val_accuracy = None
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = float((y_val_pred == y_val).mean())

        # Create metadata
        metadata = {
            "model_type": self.model_type,
            "features": self.feature_names,
            "n_features": len(self.feature_names),
            "n_classes": int(self.model.n_classes_),
            "class_names": self.model.classes_.tolist(),
            "training_date": datetime.now().isoformat(),
            "hyperparameters": self.model.hyperparameters,
            "model_specific_params": self.model.get_model_specific_params(),
            "splits": {
                "train": self.train_split,
                "val": self.val_split,
                "test": self.test_split,
            },
            "random_seed": self.random_seed,
            "val_accuracy": val_accuracy,
        }

        # Save metadata as JSON
        metadata_path = os.path.join(output_dir, METADATA_FILENAME)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")

        # Create training log
        log_path = os.path.join(output_dir, TRAINING_LOG_FILENAME)
        with open(log_path, 'w') as f:
            f.write("CSI Toolkit Model Training Log\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Training Date: {metadata['training_date']}\n")
            f.write(f"Number of Features: {len(self.feature_names)}\n")
            f.write(f"Number of Classes: {self.model.n_classes_}\n")
            f.write(f"Class Labels: {self.model.classes_.tolist()}\n")
            f.write(f"\nData Splits:\n")
            f.write(f"  Train: {self.train_split:.1%}\n")
            f.write(f"  Val:   {self.val_split:.1%}\n")
            f.write(f"  Test:  {self.test_split:.1%}\n")
            if val_accuracy is not None:
                f.write(f"\nValidation Accuracy: {val_accuracy:.4f}\n")
            f.write(f"\nFeatures Used:\n")
            for i, feat in enumerate(self.feature_names, 1):
                f.write(f"  {i:2d}. {feat}\n")
        print(f"Saved training log to: {log_path}")

    def train(
        self,
        input_csv: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Complete training pipeline: load data, train model, save results.

        Args:
            input_csv: Path to CSV file with features and labels
            output_dir: Optional output directory (auto-generated if None)

        Returns:
            Path to output directory containing trained model
        """
        # Load data
        X, y, feature_names = self.load_data(input_csv)
        self.feature_names = feature_names
        self.class_names = sorted(np.unique(y).tolist())

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Train model
        self.train_model(X_train, y_train, X_val, y_val)

        # Create output directory
        output_dir = self.create_model_directory(output_dir)

        # Save model and metadata
        self.save_model(output_dir, X_val, y_val)

        print(f"\nTraining complete! Model saved to: {output_dir}")

        return output_dir
