"""Model evaluation functionality for CSI Toolkit."""

import csv
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np

from ..utils import load_model_with_metadata, validate_feature_compatibility, format_metrics
from ..models.base import BaseModel
from ..metrics.registry import registry as metric_registry


class ModelEvaluator:
    """
    Handle evaluation of trained models on labeled test data.

    Loads a trained model and computes evaluation metrics on labeled CSI feature data.
    Requires labels in the input data.
    """

    # Columns to exclude from feature extraction (metadata columns)
    METADATA_COLUMNS = ['window_id', 'start_seq', 'end_seq', 'label']

    def __init__(self, model_dir: str, metric_names: Optional[List[str]] = None):
        """
        Initialize the evaluator with a trained model.

        Args:
            model_dir: Path to directory containing trained model and metadata
            metric_names: Optional list of specific metrics to compute.
                         If None, computes all registered metrics.

        Raises:
            FileNotFoundError: If model directory or required files don't exist
            ValueError: If model or metadata are invalid
        """
        self.model_dir = model_dir
        self.model: BaseModel
        self.metadata: Dict[str, Any]

        # Load model and metadata
        self.model, self.metadata = load_model_with_metadata(model_dir)

        # Set up metrics
        if metric_names is None:
            self.metrics = metric_registry.get_all()
        else:
            self.metrics = metric_registry.get_by_names(metric_names)

        print(f"Loaded {self.metadata['model_type']} model from {model_dir}")
        print(f"Model expects {self.metadata['n_features']} features")
        print(f"Model predicts {self.metadata['n_classes']} classes: {self.metadata['class_names']}")
        print(f"Will compute {len(self.metrics)} metrics")

    def load_labeled_data(self, csv_path: str) -> tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load labeled feature data from CSV file.

        Args:
            csv_path: Path to CSV file with features and labels

        Returns:
            Tuple of (X, y, feature_names) where:
                X: Feature matrix of shape (n_samples, n_features)
                y: Label vector of shape (n_samples,)
                feature_names: List of feature column names

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If label column is missing or features don't match
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
                "CSV file must contain a 'label' column for evaluation. "
                "Use labeled data generated with 'csi_toolkit process --labeled'."
            )

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

        # Extract features and labels (in the same order as model training)
        X = []
        y = []

        for row in rows:
            # Extract features
            features = []
            for col in self.metadata['features']:
                try:
                    features.append(float(row[col]))
                except (ValueError, KeyError):
                    raise ValueError(f"Invalid or missing feature value in column '{col}'")

            # Extract label
            try:
                label = int(row['label'])
            except (ValueError, KeyError):
                raise ValueError("Invalid or missing label value")

            X.append(features)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        print(f"Loaded {len(X)} labeled samples with {len(feature_names)} features")
        print(f"True class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        return X, y, feature_names

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute all registered metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Class probabilities (optional)

        Returns:
            Dictionary mapping metric names to computed values
        """
        results = {}

        for metric_config in self.metrics:
            try:
                if metric_config.requires_proba:
                    if y_proba is None:
                        results[metric_config.name] = "N/A (no probabilities)"
                    else:
                        results[metric_config.name] = metric_config.func(y_true, y_proba)
                else:
                    results[metric_config.name] = metric_config.func(y_true, y_pred)
            except Exception as e:
                results[metric_config.name] = f"Error: {str(e)}"

        return results

    def save_evaluation(
        self,
        metrics: Dict[str, Any],
        output_path_json: str,
        output_path_txt: str
    ) -> None:
        """
        Save evaluation metrics to files.

        Args:
            metrics: Dictionary of computed metrics
            output_path_json: Path for JSON output
            output_path_txt: Path for text report output
        """
        # Prepare JSON-serializable metrics
        json_metrics = {}
        text_only_metrics = {}

        for name, value in metrics.items():
            # Check if value is JSON-serializable
            try:
                json.dumps(value)
                json_metrics[name] = value
            except (TypeError, ValueError):
                # Store as string for JSON, keep original for text
                json_metrics[name] = str(value)
                text_only_metrics[name] = value

        # Save JSON
        with open(output_path_json, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        print(f"Saved metrics (JSON) to: {output_path_json}")

        # Save text report
        with open(output_path_txt, 'w') as f:
            f.write("CSI Toolkit Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.metadata['model_type']}\n")
            f.write(f"Model Directory: {self.model_dir}\n")
            f.write(f"Evaluation Date: {datetime.now().isoformat()}\n")
            f.write(f"\nModel Info:\n")
            f.write(f"  Classes: {self.metadata['class_names']}\n")
            f.write(f"  Features: {self.metadata['n_features']}\n")
            f.write(f"\nMetrics:\n")
            f.write(format_metrics(metrics, indent=2))
            f.write("\n")

        print(f"Saved evaluation report to: {output_path_txt}")

    def run_evaluation(
        self,
        input_csv: str,
        output_json: Optional[str] = None,
        output_txt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete evaluation pipeline: load data, predict, compute metrics, save results.

        Args:
            input_csv: Path to CSV file with labeled features
            output_json: Path for JSON output. If None, auto-generates in model directory
            output_txt: Path for text report. If None, auto-generates in model directory

        Returns:
            Dictionary of computed metrics
        """
        # Load labeled data
        X, y_true, feature_names = self.load_labeled_data(input_csv)

        # Generate predictions
        print(f"\nGenerating predictions...")
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        # Compute metrics
        print(f"Computing metrics...")
        metrics = self.compute_metrics(y_true, y_pred, y_proba)

        # Generate output paths if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_json is None:
            output_json = os.path.join(self.model_dir, f"evaluation_{timestamp}.json")
        if output_txt is None:
            output_txt = os.path.join(self.model_dir, f"evaluation_{timestamp}.txt")

        # Save results
        self.save_evaluation(metrics, output_json, output_txt)

        # Print summary
        print(f"\nEvaluation Summary:")
        print(f"  Samples evaluated: {len(y_true)}")
        if 'accuracy' in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if 'f1_macro' in metrics:
            print(f"  F1 (macro): {metrics['f1_macro']:.4f}")

        return metrics
