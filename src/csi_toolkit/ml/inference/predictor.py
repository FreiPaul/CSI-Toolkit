"""Model inference functionality for CSI Toolkit."""

import csv
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np

from ..utils import load_model_with_metadata, validate_feature_compatibility
from ..models.base import BaseModel


class ModelPredictor:
    """
    Handle inference (prediction) using trained models.

    Loads a trained model and generates predictions on new CSI feature data.
    Does not require labels in the input data.
    """

    # Columns to exclude from feature extraction (metadata columns)
    METADATA_COLUMNS = ['window_id', 'start_seq', 'end_seq', 'label']

    def __init__(self, model_dir: str):
        """
        Initialize the predictor with a trained model.

        Args:
            model_dir: Path to directory containing trained model and metadata

        Raises:
            FileNotFoundError: If model directory or required files don't exist
            ValueError: If model or metadata are invalid
        """
        self.model_dir = model_dir
        self.model: BaseModel
        self.metadata: Dict[str, Any]

        # Load model and metadata
        self.model, self.metadata = load_model_with_metadata(model_dir)

        print(f"Loaded {self.metadata['model_type']} model from {model_dir}")
        print(f"Model expects {self.metadata['n_features']} features")
        print(f"Model predicts {self.metadata['n_classes']} classes: {self.metadata['class_names']}")

    def load_features(self, csv_path: str) -> tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """
        Load feature data from CSV file.

        Args:
            csv_path: Path to CSV file with features (labels not required)

        Returns:
            Tuple of (X, feature_names, metadata_rows) where:
                X: Feature matrix of shape (n_samples, n_features)
                feature_names: List of feature column names
                metadata_rows: List of metadata dictionaries for each sample

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If features don't match model expectations
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError("CSV file is empty")

        # Extract feature names (exclude metadata columns)
        all_columns = list(rows[0].keys())
        feature_names = [
            col for col in all_columns
            if col not in self.METADATA_COLUMNS
        ]

        if not feature_names:
            raise ValueError("No feature columns found in CSV")

        # Validate feature compatibility with model
        validate_feature_compatibility(
            feature_names,
            self.metadata['features'],
            strict=True
        )

        # Extract features and metadata
        X = []
        metadata_rows = []

        for row in rows:
            # Extract features (in the same order as model training)
            features = []
            for col in self.metadata['features']:
                try:
                    features.append(float(row[col]))
                except (ValueError, KeyError):
                    raise ValueError(f"Invalid or missing feature value in column '{col}'")

            # Store metadata columns
            metadata_dict = {
                col: row.get(col, '')
                for col in self.METADATA_COLUMNS
                if col in row
            }

            X.append(features)
            metadata_rows.append(metadata_dict)

        X = np.array(X)

        print(f"Loaded {len(X)} samples with {len(feature_names)} features")

        return X, feature_names, metadata_rows

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            predictions: Predicted labels of shape (n_samples,)
        """
        return self.model.predict(X)

    def predict_with_probabilities(self, X: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions and class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Tuple of (predictions, probabilities) where:
                predictions: Predicted labels of shape (n_samples,)
                probabilities: Class probabilities of shape (n_samples, n_classes)
                              or None if model doesn't support probabilities
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return predictions, probabilities

    def save_predictions(
        self,
        predictions: np.ndarray,
        metadata_rows: List[Dict[str, Any]],
        output_path: str,
        probabilities: Optional[np.ndarray] = None
    ) -> None:
        """
        Save predictions to CSV file.

        Args:
            predictions: Predicted labels
            metadata_rows: Metadata for each sample (window_id, etc.)
            output_path: Path where predictions CSV should be saved
            probabilities: Optional class probabilities
        """
        # Prepare output rows
        output_rows = []

        for i, pred in enumerate(predictions):
            row = {**metadata_rows[i]}  # Start with metadata
            row['predicted_label'] = int(pred)

            # Add probabilities if available
            if probabilities is not None:
                for class_idx, class_name in enumerate(self.metadata['class_names']):
                    row[f'prob_class_{class_name}'] = float(probabilities[i, class_idx])

            output_rows.append(row)

        # Determine CSV columns
        if output_rows:
            fieldnames = list(output_rows[0].keys())
        else:
            fieldnames = ['predicted_label']

        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"Saved {len(predictions)} predictions to: {output_path}")

    def run_inference(
        self,
        input_csv: str,
        output_csv: Optional[str] = None,
        include_probabilities: bool = False
    ) -> str:
        """
        Complete inference pipeline: load data, predict, save results.

        Args:
            input_csv: Path to CSV file with features
            output_csv: Path for output CSV. If None, auto-generates in model directory
            include_probabilities: Whether to include class probabilities in output

        Returns:
            Path to output CSV file
        """
        # Load features
        X, feature_names, metadata_rows = self.load_features(input_csv)

        # Generate predictions
        print(f"\nGenerating predictions...")
        if include_probabilities:
            predictions, probabilities = self.predict_with_probabilities(X)
        else:
            predictions = self.predict(X)
            probabilities = None

        # Generate output path if not provided
        if output_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = os.path.join(self.model_dir, f"predictions_{timestamp}.csv")

        # Save predictions
        self.save_predictions(predictions, metadata_rows, output_csv, probabilities)

        # Print summary
        unique_preds, counts = np.unique(predictions, return_counts=True)
        print(f"\nPrediction summary:")
        for pred, count in zip(unique_preds, counts):
            print(f"  Class {pred}: {count} samples ({count/len(predictions)*100:.1f}%)")

        return output_csv
