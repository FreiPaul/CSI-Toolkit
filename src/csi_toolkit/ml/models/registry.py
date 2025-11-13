"""Model registry for CSI Toolkit ML models."""

from dataclasses import dataclass
from typing import Dict, List, Type, Optional, Any
from .base import BaseModel


@dataclass
class ModelConfig:
    """Configuration for a registered model."""

    name: str
    model_class: Type[BaseModel]
    description: str
    default_params: Dict[str, Any]


class ModelRegistry:
    """
    Registry for ML models.

    Models are registered using the @register decorator and can be retrieved by name.
    This follows the same pattern as the feature registry for consistency.
    """

    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}

    def register(
        self,
        name: str,
        description: str = "",
        default_params: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator to register a model class.

        Args:
            name: Unique identifier for the model
            description: Human-readable description of the model
            default_params: Default hyperparameters for the model

        Returns:
            Decorator function

        Example:
            @registry.register('mlp', description='Multi-layer Perceptron')
            class MLPModel(BaseModel):
                ...
        """
        if default_params is None:
            default_params = {}

        def decorator(model_class: Type[BaseModel]):
            if not issubclass(model_class, BaseModel):
                raise TypeError(
                    f"Model class {model_class.__name__} must inherit from BaseModel"
                )

            if name in self._models:
                raise ValueError(f"Model '{name}' is already registered")

            self._models[name] = ModelConfig(
                name=name,
                model_class=model_class,
                description=description,
                default_params=default_params
            )

            return model_class

        return decorator

    def get(self, name: str) -> ModelConfig:
        """
        Get a model configuration by name.

        Args:
            name: Model name

        Returns:
            ModelConfig for the requested model

        Raises:
            ValueError: If model is not registered
        """
        if name not in self._models:
            available = ', '.join(self.list_names())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )
        return self._models[name]

    def get_all(self) -> List[ModelConfig]:
        """
        Get all registered models.

        Returns:
            List of all ModelConfig objects
        """
        return list(self._models.values())

    def list_names(self) -> List[str]:
        """
        Get names of all registered models.

        Returns:
            List of model names
        """
        return list(self._models.keys())

    def create_model(self, name: str, **kwargs) -> BaseModel:
        """
        Create an instance of a registered model.

        Args:
            name: Model name
            **kwargs: Hyperparameters to override defaults

        Returns:
            Instantiated model object

        Example:
            model = registry.create_model('mlp', hidden_layer_sizes=(100, 50))
        """
        config = self.get(name)

        # Merge default params with user-provided params
        params = {**config.default_params, **kwargs}

        return config.model_class(**params)

    def list_models(self) -> str:
        """
        Get a formatted string listing all available models.

        Returns:
            Human-readable string listing models and their descriptions
        """
        if not self._models:
            return "No models registered"

        lines = ["Available models:"]
        for config in sorted(self._models.values(), key=lambda c: c.name):
            lines.append(f"  {config.name:15} - {config.description}")

        return "\n".join(lines)


# Global registry instance
registry = ModelRegistry()
