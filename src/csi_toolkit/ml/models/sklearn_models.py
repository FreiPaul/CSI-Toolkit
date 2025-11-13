"""Scikit-learn model implementations for CSI Toolkit."""

import pickle
from typing import Any, Dict, Optional
import numpy as np

try:
    from sklearn.neural_network import MLPClassifier as SklearnMLP
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    raise ImportError(
        "scikit-learn is required for ML functionality. "
        "Install with: pip install -e '.[ml]'"
    )

from .base import BaseModel
from .registry import registry


@registry.register(
    'mlp',
    description='Multi-layer Perceptron neural network',
    default_params={
        'hidden_layer_sizes': (100, 50),
        'max_iter': 500,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'activation': 'relu',
        'solver': 'adam',
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'verbose': False
    }
)
class MLPModel(BaseModel):
    """
    Multi-layer Perceptron (MLP) classifier using scikit-learn.

    This is a feedforward artificial neural network that can learn non-linear
    decision boundaries. Well-suited for CSI-based activity recognition.

    The model uses:
    - Adam optimizer for efficient training
    - ReLU activation for non-linearity
    - Early stopping to prevent overfitting
    - Adaptive learning rate for faster convergence
    """

    def __init__(self, **kwargs):
        """
        Initialize MLP model.

        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes, e.g., (100, 50)
            max_iter: Maximum number of training iterations
            learning_rate: Learning rate schedule ('constant', 'adaptive', 'invscaling')
            learning_rate_init: Initial learning rate
            activation: Activation function ('relu', 'tanh', 'logistic')
            solver: Optimizer ('adam', 'sgd', 'lbfgs')
            random_state: Random seed for reproducibility
            early_stopping: Whether to use early stopping
            validation_fraction: Fraction of training data for validation
            n_iter_no_change: Iterations with no improvement before stopping
            **kwargs: Additional scikit-learn MLPClassifier parameters
        """
        super().__init__(**kwargs)
        self.model = SklearnMLP(**kwargs)
        self.label_encoder = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPModel':
        """
        Train the MLP model.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)

        Returns:
            self: The fitted model
        """
        # Encode labels to ensure they're sequential integers starting from 0
        y_encoded = self.label_encoder.fit_transform(y)

        # Fit the model
        self.model.fit(X, y_encoded)

        # Store metadata
        self.n_features_ = X.shape[1]
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            predictions: Predicted labels in original label space
        """
        self._validate_fitted()
        self._validate_features(X)

        # Get predictions and decode back to original labels
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate class probability estimates.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            probabilities: Class probabilities of shape (n_samples, n_classes)
        """
        self._validate_fitted()
        self._validate_features(X)

        return self.model.predict_proba(X)

    def get_model_specific_params(self) -> Dict[str, Any]:
        """
        Get MLP-specific parameters.

        Returns:
            Dictionary containing model architecture and training info
        """
        if not self.is_fitted:
            return {}

        return {
            'hidden_layer_sizes': self.hyperparameters.get('hidden_layer_sizes'),
            'n_layers': len(self.model.coefs_),
            'n_iter': self.model.n_iter_,
            'loss': float(self.model.loss_) if hasattr(self.model, 'loss_') and self.model.loss_ is not None else None,
            'best_loss': float(self.model.best_loss_) if hasattr(self.model, 'best_loss_') and self.model.best_loss_ is not None else None,
            'activation': self.hyperparameters.get('activation'),
            'solver': self.hyperparameters.get('solver'),
        }

    def save(self, path: str) -> None:
        """
        Save the model to disk using pickle.

        Args:
            path: File path where the model should be saved
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call .fit() first.")

        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'hyperparameters': self.hyperparameters,
            'n_features_': self.n_features_,
            'n_classes_': self.n_classes_,
            'classes_': self.classes_,
            'is_fitted': self.is_fitted,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path: File path from which to load the model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.hyperparameters = model_data['hyperparameters']
        self.n_features_ = model_data['n_features_']
        self.n_classes_ = model_data['n_classes_']
        self.classes_ = model_data['classes_']
        self.is_fitted = model_data['is_fitted']
