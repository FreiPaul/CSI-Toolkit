"""Metrics registry for CSI Toolkit ML evaluation."""

from dataclasses import dataclass
from typing import Dict, List, Callable, Any
import numpy as np


@dataclass
class MetricConfig:
    """Configuration for a registered metric."""

    name: str
    func: Callable[[np.ndarray, np.ndarray], Any]
    description: str
    requires_proba: bool = False  # Whether metric needs probability predictions


class MetricRegistry:
    """
    Registry for evaluation metrics.

    Metrics are registered using the @register decorator and can be retrieved by name.
    Follows the same pattern as feature and model registries.
    """

    def __init__(self):
        self._metrics: Dict[str, MetricConfig] = {}

    def register(
        self,
        name: str,
        description: str = "",
        requires_proba: bool = False
    ):
        """
        Decorator to register a metric function.

        Args:
            name: Unique identifier for the metric
            description: Human-readable description of the metric
            requires_proba: Whether the metric needs probability predictions

        Returns:
            Decorator function

        Example:
            @registry.register('accuracy', description='Classification accuracy')
            def accuracy_score(y_true, y_pred):
                return (y_true == y_pred).mean()
        """
        def decorator(func: Callable):
            if name in self._metrics:
                raise ValueError(f"Metric '{name}' is already registered")

            self._metrics[name] = MetricConfig(
                name=name,
                func=func,
                description=description,
                requires_proba=requires_proba
            )

            return func

        return decorator

    def get(self, name: str) -> MetricConfig:
        """
        Get a metric configuration by name.

        Args:
            name: Metric name

        Returns:
            MetricConfig for the requested metric

        Raises:
            ValueError: If metric is not registered
        """
        if name not in self._metrics:
            available = ', '.join(self.list_names())
            raise ValueError(
                f"Metric '{name}' not found. Available metrics: {available}"
            )
        return self._metrics[name]

    def get_all(self) -> List[MetricConfig]:
        """
        Get all registered metrics.

        Returns:
            List of all MetricConfig objects
        """
        return list(self._metrics.values())

    def get_by_names(self, names: List[str]) -> List[MetricConfig]:
        """
        Get multiple metrics by name.

        Args:
            names: List of metric names

        Returns:
            List of MetricConfig objects

        Raises:
            ValueError: If any metric is not found
        """
        return [self.get(name) for name in names]

    def list_names(self) -> List[str]:
        """
        Get names of all registered metrics.

        Returns:
            List of metric names
        """
        return list(self._metrics.keys())

    def compute_metric(self, name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        Compute a metric by name.

        Args:
            name: Metric name
            y_true: Ground truth labels
            y_pred: Predicted labels (or probabilities if requires_proba=True)

        Returns:
            Computed metric value
        """
        config = self.get(name)
        return config.func(y_true, y_pred)

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Compute all registered metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for probability-based metrics)

        Returns:
            Dictionary mapping metric names to their computed values
        """
        results = {}

        for name, config in self._metrics.items():
            try:
                if config.requires_proba:
                    if y_proba is None:
                        results[name] = None
                    else:
                        results[name] = config.func(y_true, y_proba)
                else:
                    results[name] = config.func(y_true, y_pred)
            except Exception as e:
                results[name] = f"Error: {str(e)}"

        return results

    def list_metrics(self) -> str:
        """
        Get a formatted string listing all available metrics.

        Returns:
            Human-readable string listing metrics and their descriptions
        """
        if not self._metrics:
            return "No metrics registered"

        lines = ["Available metrics:"]
        for config in sorted(self._metrics.values(), key=lambda c: c.name):
            proba_note = " (requires probabilities)" if config.requires_proba else ""
            lines.append(f"  {config.name:20} - {config.description}{proba_note}")

        return "\n".join(lines)


# Global registry instance
registry = MetricRegistry()
