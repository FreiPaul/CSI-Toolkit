"""Classification metrics for CSI Toolkit ML evaluation."""

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report as sklearn_classification_report,
        confusion_matrix as sklearn_confusion_matrix
    )
except ImportError:
    raise ImportError(
        "scikit-learn is required for ML functionality. "
        "Install with: pip install -e '.[ml]'"
    )

from .registry import registry


@registry.register('accuracy', description='Overall classification accuracy')
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy score (fraction of correct predictions)
    """
    return float(accuracy_score(y_true, y_pred))


@registry.register('precision_macro', description='Macro-averaged precision across all classes')
def precision_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged precision.

    Calculates precision for each class and averages them (unweighted).
    Useful when all classes are equally important.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Macro-averaged precision score
    """
    return float(precision_score(y_true, y_pred, average='macro', zero_division=0))


@registry.register('precision_micro', description='Micro-averaged precision across all classes')
def precision_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute micro-averaged precision.

    Aggregates contributions of all classes to compute the average.
    Useful when class imbalance exists.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Micro-averaged precision score
    """
    return float(precision_score(y_true, y_pred, average='micro', zero_division=0))


@registry.register('recall_macro', description='Macro-averaged recall across all classes')
def recall_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged recall.

    Calculates recall for each class and averages them (unweighted).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Macro-averaged recall score
    """
    return float(recall_score(y_true, y_pred, average='macro', zero_division=0))


@registry.register('recall_micro', description='Micro-averaged recall across all classes')
def recall_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute micro-averaged recall.

    Aggregates contributions of all classes to compute the average.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Micro-averaged recall score
    """
    return float(recall_score(y_true, y_pred, average='micro', zero_division=0))


@registry.register('f1_macro', description='Macro-averaged F1 score across all classes')
def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged F1 score.

    Calculates F1 score for each class and averages them (unweighted).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Macro-averaged F1 score
    """
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))


@registry.register('f1_micro', description='Micro-averaged F1 score across all classes')
def f1_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute micro-averaged F1 score.

    Aggregates contributions of all classes to compute the average.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Micro-averaged F1 score
    """
    return float(f1_score(y_true, y_pred, average='micro', zero_division=0))


@registry.register('precision_per_class', description='Precision for each class individually')
def precision_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute precision for each class.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary mapping class labels to precision scores
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)

    return {str(cls): float(prec) for cls, prec in zip(classes, precisions)}


@registry.register('recall_per_class', description='Recall for each class individually')
def recall_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute recall for each class.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary mapping class labels to recall scores
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)

    return {str(cls): float(rec) for cls, rec in zip(classes, recalls)}


@registry.register('f1_per_class', description='F1 score for each class individually')
def f1_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute F1 score for each class.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary mapping class labels to F1 scores
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)

    return {str(cls): float(f1) for cls, f1 in zip(classes, f1_scores)}


@registry.register('classification_report', description='Comprehensive classification report with all metrics')
def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate a comprehensive classification report.

    Includes precision, recall, F1 score, and support for each class,
    plus macro and weighted averages.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Formatted classification report string
    """
    return sklearn_classification_report(y_true, y_pred, zero_division=0)


@registry.register('confusion_matrix', description='Confusion matrix showing predictions vs actual')
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with 'matrix' (2D list) and 'labels' (class labels)
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=classes)

    return {
        'matrix': cm.tolist(),
        'labels': classes.tolist()
    }
