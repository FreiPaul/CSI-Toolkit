"""Utility functions for CSI Toolkit ML operations."""

import json
import os
from typing import Dict, Any, Tuple
from pathlib import Path

from .models.registry import registry as model_registry
from .models.base import BaseModel
from ..core.constants import MODEL_FILENAME, METADATA_FILENAME


def load_metadata(model_dir: str) -> Dict[str, Any]:
    """
    Load model metadata from directory.

    Args:
        model_dir: Path to model directory

    Returns:
        Dictionary containing model metadata

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If metadata is invalid
    """
    metadata_path = os.path.join(model_dir, METADATA_FILENAME)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. "
            "Make sure the model directory contains metadata.json"
        )

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Validate required fields
    required_fields = ['model_type', 'features', 'n_features', 'n_classes']
    missing_fields = [f for f in required_fields if f not in metadata]

    if missing_fields:
        raise ValueError(
            f"Invalid metadata: missing required fields {missing_fields}"
        )

    return metadata


def load_model_with_metadata(model_dir: str) -> Tuple[BaseModel, Dict[str, Any]]:
    """
    Load a trained model and its metadata from directory.

    This function handles the complete model loading process:
    1. Load and validate metadata
    2. Instantiate the correct model class
    3. Load model weights
    4. Return both model and metadata

    Args:
        model_dir: Path to directory containing model and metadata

    Returns:
        Tuple of (model, metadata) where:
            model: Loaded BaseModel instance
            metadata: Dictionary containing model metadata

    Raises:
        FileNotFoundError: If model directory or files don't exist
        ValueError: If model type is not registered or files are invalid

    Example:
        model, metadata = load_model_with_metadata('models/model_20250113_143022')
        predictions = model.predict(X)
        print(f"Model uses {len(metadata['features'])} features")
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load metadata
    metadata = load_metadata(model_dir)

    # Get model type
    model_type = metadata['model_type']

    # Create model instance (empty, will be populated by load())
    try:
        model = model_registry.create_model(model_type)
    except ValueError:
        available = ', '.join(model_registry.list_names())
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available types: {available}. "
            "You may need to register custom models."
        )

    # Load model weights
    model_path = os.path.join(model_dir, MODEL_FILENAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure the model directory contains model.pkl"
        )

    model.load(model_path)

    return model, metadata


def validate_feature_compatibility(
    csv_features: list,
    model_features: list,
    strict: bool = True
) -> None:
    """
    Validate that CSV features match model's expected features.

    Args:
        csv_features: List of feature names from CSV
        model_features: List of feature names from model metadata
        strict: If True, raise error on mismatch. If False, only warn.

    Raises:
        ValueError: If features don't match and strict=True
    """
    csv_set = set(csv_features)
    model_set = set(model_features)

    if csv_set == model_set:
        return  # Perfect match

    missing = model_set - csv_set
    extra = csv_set - model_set

    error_msg = []
    if missing:
        error_msg.append(f"Missing features: {sorted(missing)}")
    if extra:
        error_msg.append(f"Extra features: {sorted(extra)}")

    message = "Feature mismatch between CSV and model:\n" + "\n".join(error_msg)

    if strict:
        raise ValueError(message)
    else:
        print(f"Warning: {message}")


def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def format_metrics(metrics: Dict[str, Any], indent: int = 0) -> str:
    """
    Format metrics dictionary as a human-readable string.

    Args:
        metrics: Dictionary of metric names to values
        indent: Number of spaces to indent

    Returns:
        Formatted string representation
    """
    lines = []
    indent_str = " " * indent

    for name, value in sorted(metrics.items()):
        if isinstance(value, dict):
            # Nested dictionary (e.g., per-class metrics)
            lines.append(f"{indent_str}{name}:")
            for k, v in sorted(value.items()):
                if isinstance(v, float):
                    lines.append(f"{indent_str}  {k}: {v:.4f}")
                else:
                    lines.append(f"{indent_str}  {k}: {v}")
        elif isinstance(value, float):
            lines.append(f"{indent_str}{name}: {value:.4f}")
        elif isinstance(value, str):
            # Multi-line string (e.g., classification report)
            if '\n' in value:
                lines.append(f"{indent_str}{name}:")
                for line in value.split('\n'):
                    lines.append(f"{indent_str}  {line}")
            else:
                lines.append(f"{indent_str}{name}: {value}")
        else:
            lines.append(f"{indent_str}{name}: {value}")

    return "\n".join(lines)
