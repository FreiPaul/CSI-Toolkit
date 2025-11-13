"""Feature registration system for window-based feature extraction."""

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class FeatureConfig:
    """Configuration for a registered feature."""

    name: str
    func: Callable
    n_prev_windows: int = 0
    n_next_windows: int = 0
    description: str = ""


class FeatureRegistry:
    """Registry for window-based feature functions."""

    def __init__(self):
        self._features = {}

    def register(
        self,
        name: str,
        n_prev: int = 0,
        n_next: int = 0,
        description: str = ""
    ):
        """
        Decorator to register a feature function.

        Args:
            name: Feature name (used as column name in output)
            n_prev: Number of previous windows required
            n_next: Number of next windows required
            description: Human-readable description

        Returns:
            Decorator function

        Example:
            @registry.register('mean_amp', description='Mean amplitude')
            def mean_amplitude(current_samples, prev_samples, next_samples):
                return np.mean([s.amplitude for s in current_samples])
        """
        def decorator(func: Callable) -> Callable:
            config = FeatureConfig(
                name=name,
                func=func,
                n_prev_windows=n_prev,
                n_next_windows=n_next,
                description=description
            )
            self._features[name] = config
            return func
        return decorator

    def get(self, name: str) -> Optional[FeatureConfig]:
        """
        Get feature configuration by name.

        Args:
            name: Feature name

        Returns:
            FeatureConfig or None if not found
        """
        return self._features.get(name)

    def get_all(self) -> List[FeatureConfig]:
        """
        Get all registered features.

        Returns:
            List of FeatureConfig objects
        """
        return list(self._features.values())

    def list_names(self) -> List[str]:
        """
        Get list of all feature names.

        Returns:
            List of feature names
        """
        return list(self._features.keys())

    def get_by_names(self, names: List[str]) -> List[FeatureConfig]:
        """
        Get feature configurations for specific names.

        Args:
            names: List of feature names

        Returns:
            List of FeatureConfig objects

        Raises:
            ValueError: If any feature name is not registered
        """
        configs = []
        for name in names:
            config = self.get(name)
            if config is None:
                available = ', '.join(self.list_names())
                raise ValueError(
                    f"Feature '{name}' not registered. "
                    f"Available features: {available}"
                )
            configs.append(config)
        return configs


# Global registry instance
registry = FeatureRegistry()
