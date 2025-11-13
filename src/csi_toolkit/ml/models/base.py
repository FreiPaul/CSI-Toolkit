"""Base model interface for CSI Toolkit ML models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all ML models in CSI Toolkit.

    Custom models should inherit from this class and implement all abstract methods.
    Models are responsible for training, prediction, and serialization.
    """

    def __init__(self, **kwargs):
        """
        Initialize the model with hyperparameters.

        Args:
            **kwargs: Model-specific hyperparameters
        """
        self.hyperparameters = kwargs
        self.is_fitted = False
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Train the model on the provided data.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)

        Returns:
            self: The fitted model
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for the input data.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            predictions: Predicted labels of shape (n_samples,)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate class probabilities for the input data.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            probabilities: Class probabilities of shape (n_samples, n_classes)
                          or None if model doesn't support probability estimates
        """
        pass

    @abstractmethod
    def get_model_specific_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters for metadata serialization.

        Returns:
            Dictionary containing model-specific configuration and learned parameters
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get all model parameters including hyperparameters and learned parameters.

        Returns:
            Dictionary containing all model parameters
        """
        params = {
            'hyperparameters': self.hyperparameters,
            'n_features': self.n_features_,
            'n_classes': self.n_classes_,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
        }
        params.update(self.get_model_specific_params())
        return params

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: File path where the model should be saved
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path: File path from which to load the model
        """
        pass

    def _validate_fitted(self) -> None:
        """Check if model is fitted, raise error if not."""
        if not self.is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before calling predict. "
                "Call .fit(X, y) first."
            )

    def _validate_features(self, X: np.ndarray) -> None:
        """
        Validate input features match expected dimensions.

        Args:
            X: Input features to validate

        Raises:
            ValueError: If feature dimensions don't match
        """
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}. "
                "Model was trained on different features."
            )
