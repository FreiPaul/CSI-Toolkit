# Adding Custom Models

Guide for implementing custom ML models in the CSI Toolkit.

## Overview

The CSI Toolkit uses a registry-based model system that allows integration of custom ML models without modifying core code.

## BaseModel Interface

All models must inherit from `BaseModel` and implement required methods.

### Required Methods

```python
class BaseModel:
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Train the model."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Generate class probabilities (optional)."""
        pass

    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    def get_model_specific_params(self) -> dict:
        """Get model-specific metadata."""
        pass
```

## Basic Model Example

Create `src/csi_toolkit/ml/models/custom_models.py`:

```python
"""Custom model implementations."""

import numpy as np
from .base import BaseModel
from .registry import registry


@registry.register(
    'simple_knn',
    description='Simple K-Nearest Neighbors classifier',
    default_params={
        'n_neighbors': 5,
        'metric': 'euclidean',
    }
)
class SimpleKNN(BaseModel):
    """K-Nearest Neighbors classifier."""

    def __init__(self, n_neighbors=5, metric='euclidean', **kwargs):
        super().__init__(n_neighbors=n_neighbors, metric=metric, **kwargs)

        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleKNN':
        """Train the KNN model."""
        # Store metadata
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Train model
        self.model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self._validate_fitted()
        self._validate_features(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probabilities."""
        self._validate_fitted()
        self._validate_features(X)
        return self.model.predict_proba(X)

    def get_model_specific_params(self) -> dict:
        """Get KNN-specific parameters."""
        if not self.is_fitted:
            return {}

        return {
            'n_neighbors': self.hyperparameters.get('n_neighbors'),
            'metric': self.hyperparameters.get('metric'),
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        import pickle

        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            'model': self.model,
            'hyperparameters': self.hyperparameters,
            'n_features_': self.n_features_,
            'n_classes_': self.n_classes_,
            'classes_': self.classes_,
            'is_fitted': self.is_fitted,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        import pickle

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.hyperparameters = model_data['hyperparameters']
        self.n_features_ = model_data['n_features_']
        self.n_classes_ = model_data['n_classes_']
        self.classes_ = model_data['classes_']
        self.is_fitted = model_data['is_fitted']
```

## Registering Models

Add your module to `src/csi_toolkit/ml/models/__init__.py`:

```python
"""ML model implementations."""

from . import sklearn_models
from . import custom_models  # Add this line

__all__ = ['registry']
```

## Registration Options

The `@registry.register()` decorator accepts:

```python
@registry.register(
    name='model_name',           # Required: Model identifier
    description='Description',   # Optional: Human-readable description
    default_params={             # Optional: Default hyperparameters
        'param1': value1,
        'param2': value2,
    }
)
```

## Advanced Example: Neural Network

```python
@registry.register(
    'custom_nn',
    description='Custom neural network with dropout',
    default_params={
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
    }
)
class CustomNN(BaseModel):
    """Custom neural network implementation."""

    def __init__(self, hidden_layers=[128, 64, 32], dropout_rate=0.3,
                 learning_rate=0.001, epochs=100, batch_size=32, **kwargs):
        super().__init__(
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )

        self.model = None
        self.history = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomNN':
        """Train the neural network."""
        from sklearn.preprocessing import LabelEncoder
        import tensorflow as tf

        # Store metadata
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Build model
        self.model = self._build_model()

        # Train
        self.history = self.model.fit(
            X, y_encoded,
            epochs=self.hyperparameters['epochs'],
            batch_size=self.hyperparameters['batch_size'],
            validation_split=0.2,
            verbose=0
        )

        self.is_fitted = True
        return self

    def _build_model(self):
        """Build neural network architecture."""
        import tensorflow as tf

        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Input(shape=(self.n_features_,)))

        # Hidden layers with dropout
        for units in self.hyperparameters['hidden_layers']:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dropout(self.hyperparameters['dropout_rate']))

        # Output layer
        model.add(tf.keras.layers.Dense(self.n_classes_, activation='softmax'))

        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.hyperparameters['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self._validate_fitted()
        self._validate_features(X)

        y_pred_proba = self.model.predict(X, verbose=0)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)

        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probabilities."""
        self._validate_fitted()
        self._validate_features(X)

        return self.model.predict(X, verbose=0)

    def get_model_specific_params(self) -> dict:
        """Get neural network specific parameters."""
        if not self.is_fitted or self.history is None:
            return {}

        return {
            'hidden_layers': self.hyperparameters['hidden_layers'],
            'dropout_rate': self.hyperparameters['dropout_rate'],
            'final_loss': float(self.history.history['loss'][-1]),
            'final_accuracy': float(self.history.history['accuracy'][-1]),
            'epochs_trained': len(self.history.history['loss']),
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        import pickle

        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        # Save Keras model separately
        self.model.save(path + '.keras')

        # Save other attributes
        model_data = {
            'hyperparameters': self.hyperparameters,
            'n_features_': self.n_features_,
            'n_classes_': self.n_classes_,
            'classes_': self.classes_,
            'is_fitted': self.is_fitted,
            'label_encoder': self.label_encoder,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        import pickle
        import tensorflow as tf

        # Load Keras model
        self.model = tf.keras.models.load_model(path + '.keras')

        # Load other attributes
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.hyperparameters = model_data['hyperparameters']
        self.n_features_ = model_data['n_features_']
        self.n_classes_ = model_data['n_classes_']
        self.classes_ = model_data['classes_']
        self.is_fitted = model_data['is_fitted']
        self.label_encoder = model_data['label_encoder']
```

## BaseModel Helper Methods

The `BaseModel` class provides utility methods:

### Validation Methods

```python
def _validate_fitted(self):
    """Ensure model is fitted before use."""
    if not self.is_fitted:
        raise ValueError("Model must be fitted before calling this method")

def _validate_features(self, X):
    """Ensure feature count matches training."""
    if X.shape[1] != self.n_features_:
        raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
```

Use these in your predict methods:

```python
def predict(self, X):
    self._validate_fitted()
    self._validate_features(X)
    # ... prediction logic
```

## Required Attributes

Models must set these attributes during `fit()`:

```python
self.n_features_ = X.shape[1]         # Number of features
self.classes_ = np.unique(y)          # Unique class labels
self.n_classes_ = len(self.classes_)  # Number of classes
self.is_fitted = True                  # Fitted flag
```

## Using Custom Models

After registration:

### List Models

```bash
python -m csi_toolkit train --list-models
```

### Train with Custom Model

```bash
python -m csi_toolkit train features.csv --model simple_knn
```

### Override Hyperparameters

```bash
python -m csi_toolkit train features.csv \
  --model simple_knn \
  --params n_neighbors=10,metric=manhattan
```

## Testing Models

1. **Test Training**:

```python
# test_model.py
from csi_toolkit.ml.models import registry
import numpy as np

# Get model
ModelClass = registry.get('simple_knn')
model = ModelClass(n_neighbors=5)

# Create dummy data
X = np.random.rand(100, 10)
y = np.random.randint(0, 3, 100)

# Train
model.fit(X, y)

# Predict
predictions = model.predict(X[:10])
print(predictions)
```

2. **Test Save/Load**:

```python
# Save
model.save('/tmp/test_model.pkl')

# Load
new_model = ModelClass()
new_model.load('/tmp/test_model.pkl')

# Verify
assert np.array_equal(model.predict(X[:10]), new_model.predict(X[:10]))
```

## Best Practices

1. **Inherit from BaseModel**: Always extend the base class
2. **Call super().__init__()**: Initialize parent class properly
3. **Set Required Attributes**: Set `n_features_`, `classes_`, `n_classes_`, `is_fitted`
4. **Use Validation Methods**: Call `_validate_fitted()` and `_validate_features()`
5. **Handle Errors**: Provide clear error messages
6. **Document Parameters**: Add docstrings for all hyperparameters
7. **Test Thoroughly**: Test fit, predict, save, load
8. **Support Probabilities**: Implement `predict_proba()` if possible
9. **Pickle Compatibility**: Ensure all attributes are picklable

## Common Patterns

### Wrapper for sklearn Models

```python
@registry.register('svm', default_params={'C': 1.0, 'kernel': 'rbf'})
class SVMModel(BaseModel):
    def __init__(self, C=1.0, kernel='rbf', **kwargs):
        super().__init__(C=C, kernel=kernel, **kwargs)
        from sklearn.svm import SVC
        self.model = SVC(C=C, kernel=kernel, probability=True)

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        self._validate_fitted()
        self._validate_features(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        self._validate_fitted()
        self._validate_features(X)
        return self.model.predict_proba(X)

    # ... save/load methods
```

## Next Steps

- Study existing models in `src/csi_toolkit/ml/models/sklearn_models.py`
- See [Architecture Guide](architecture.md) for system overview
- Check [Adding Metrics](adding-metrics.md) for custom evaluation metrics
