# Adding Custom Metrics

Guide for implementing custom evaluation metrics in the CSI Toolkit.

## Overview

The CSI Toolkit uses a registry-based metric system allowing easy addition of custom evaluation metrics.

## Metric Function Signature

Metrics are functions that take ground truth and predictions and return computed values.

### Basic Metric

```python
def metric_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute evaluation metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Metric value (float or dict for per-class metrics)
    """
    pass
```

### Probability-Based Metric

```python
def metric_function(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute metric using probabilities.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities (n_samples, n_classes)

    Returns:
        Metric value
    """
    pass
```

## Basic Metric Example

Create `src/csi_toolkit/ml/metrics/custom_metrics.py`:

```python
"""Custom evaluation metrics."""

import numpy as np
from .registry import registry


@registry.register('balanced_accuracy', description='Balanced accuracy score')
def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate balanced accuracy (average of per-class accuracy).

    Useful for imbalanced datasets.
    """
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


@registry.register('top_class_accuracy', description='Accuracy of most frequent class')
def top_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy for the most frequent class."""
    # Find most frequent class
    unique, counts = np.unique(y_true, return_counts=True)
    top_class = unique[np.argmax(counts)]

    # Calculate accuracy for that class only
    mask = y_true == top_class
    if mask.sum() == 0:
        return 0.0

    return float((y_true[mask] == y_pred[mask]).sum() / mask.sum())
```

## Registering Metrics

Add your module to `src/csi_toolkit/ml/metrics/__init__.py`:

```python
"""Evaluation metrics."""

from . import classification_metrics
from . import custom_metrics  # Add this line

__all__ = ['registry']
```

## Registration Options

```python
@registry.register(
    name='metric_name',           # Required: Metric identifier
    description='Description',    # Optional: Human-readable description
    requires_proba=False,         # Optional: True if needs probabilities
)
```

## Per-Class Metrics

Return a dictionary for per-class metrics:

```python
@registry.register(
    'accuracy_per_class',
    description='Accuracy computed separately for each class'
)
def accuracy_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate per-class accuracy."""
    classes = np.unique(y_true)
    accuracies = {}

    for cls in classes:
        mask = y_true == cls
        if mask.sum() == 0:
            accuracies[str(cls)] = 0.0
        else:
            accuracies[str(cls)] = float(
                (y_true[mask] == y_pred[mask]).sum() / mask.sum()
            )

    return accuracies
```

## Probability-Based Metrics

For metrics that need class probabilities:

```python
@registry.register(
    'cross_entropy',
    description='Cross-entropy loss',
    requires_proba=True
)
def cross_entropy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate cross-entropy loss."""
    from sklearn.metrics import log_loss
    return float(log_loss(y_true, y_proba))


@registry.register(
    'brier_score',
    description='Brier score for probability calibration',
    requires_proba=True
)
def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate Brier score."""
    from sklearn.metrics import brier_score_loss
    from sklearn.preprocessing import label_binarize

    # Binarize labels
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)

    # Calculate average Brier score across classes
    scores = []
    for i in range(len(classes)):
        score = brier_score_loss(y_true_bin[:, i], y_proba[:, i])
        scores.append(score)

    return float(np.mean(scores))
```

## Advanced Examples

### Weighted Metrics

```python
@registry.register(
    'weighted_f1',
    description='F1 score weighted by class support'
)
def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate weighted F1 score."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average='weighted'))
```

### Custom Confusion-Based Metrics

```python
@registry.register(
    'misclassification_cost',
    description='Custom cost based on confusion matrix'
)
def misclassification_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate custom misclassification cost.

    Assigns different costs to different types of errors.
    """
    from sklearn.metrics import confusion_matrix

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Define cost matrix (example: different costs for different errors)
    # cost[i][j] = cost of predicting class j when true class is i
    n_classes = cm.shape[0]
    cost_matrix = np.ones((n_classes, n_classes))
    np.fill_diagonal(cost_matrix, 0)  # No cost for correct predictions

    # You can customize costs here
    # Example: higher cost for specific misclassifications
    # cost_matrix[0, 1] = 2.0  # High cost for mistaking class 0 as 1

    # Calculate total cost
    total_cost = np.sum(cm * cost_matrix)

    return float(total_cost / len(y_true))
```

### Statistical Metrics

```python
@registry.register(
    'cohens_kappa',
    description="Cohen's kappa coefficient"
)
def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Cohen's kappa coefficient."""
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred))


@registry.register(
    'matthews_correlation',
    description='Matthews correlation coefficient'
)
def matthews_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Matthews correlation coefficient."""
    from sklearn.metrics import matthews_corrcoef
    return float(matthews_corrcoef(y_true, y_pred))
```

## Using Custom Metrics

After registration:

### List Metrics

```bash
python -m csi_toolkit evaluate --list-metrics
```

### Evaluate with Custom Metrics

```bash
python -m csi_toolkit evaluate \
  --dataset test.csv \
  --model-dir models/model_X \
  --metrics balanced_accuracy,cohens_kappa
```

## Testing Metrics

```python
# test_metric.py
from csi_toolkit.ml.metrics import registry
import numpy as np

# Get metric function
metric_func = registry.get('balanced_accuracy')

# Create test data
y_true = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
y_pred = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])

# Compute metric
score = metric_func(y_true, y_pred)
print(f"Balanced Accuracy: {score:.4f}")
```

## Best Practices

1. **Return Type**: Always return `float` or `dict` (for per-class metrics)
2. **Handle Edge Cases**: Check for empty arrays, division by zero
3. **Clear Names**: Use descriptive metric names
4. **Documentation**: Add docstrings explaining the metric
5. **Validation**: Test with known inputs and expected outputs
6. **Performance**: Use NumPy/sklearn for efficiency
7. **Error Handling**: Provide meaningful error messages

## Error Handling Example

```python
@registry.register('robust_accuracy', description='Accuracy with error handling')
def robust_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy with robust error handling."""
    try:
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
            )

        correct = (y_true == y_pred).sum()
        total = len(y_true)

        return float(correct / total)

    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return 0.0
```

## Complex Example: Per-Class with Probabilities

```python
@registry.register(
    'calibration_per_class',
    description='Probability calibration error per class',
    requires_proba=True
)
def calibration_per_class(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Calculate Expected Calibration Error (ECE) for each class.

    Measures how well predicted probabilities match actual frequencies.
    """
    classes = np.unique(y_true)
    ece_scores = {}

    for i, cls in enumerate(classes):
        # Binary problem: this class vs others
        y_binary = (y_true == cls).astype(int)
        proba_cls = y_proba[:, i]

        # Bin predictions
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)

        ece = 0.0
        for j in range(n_bins):
            # Find samples in this bin
            in_bin = (proba_cls >= bins[j]) & (proba_cls < bins[j + 1])

            if in_bin.sum() == 0:
                continue

            # Average confidence and accuracy in bin
            avg_confidence = proba_cls[in_bin].mean()
            avg_accuracy = y_binary[in_bin].mean()

            # Weight by number of samples
            weight = in_bin.sum() / len(y_true)

            # Add to ECE
            ece += weight * abs(avg_confidence - avg_accuracy)

        ece_scores[str(cls)] = float(ece)

    return ece_scores
```

## Integration with Evaluation Pipeline

Metrics are automatically integrated:

1. **JSON Output**: Metric values saved to evaluation JSON
2. **Text Report**: Formatted in human-readable report
3. **Per-Class Handling**: Dict returns expanded into separate entries

Example output:

```json
{
  "balanced_accuracy": 0.9123,
  "cohens_kappa": 0.8654,
  "calibration_per_class": {
    "1": 0.0234,
    "2": 0.0312,
    "3": 0.0198
  }
}
```

## Next Steps

- Review existing metrics in `src/csi_toolkit/ml/metrics/classification_metrics.py`
- See [Machine Learning Guide](../user-guide/machine-learning.md) for evaluation workflow
- Check [Adding Models](adding-models.md) for custom model development
