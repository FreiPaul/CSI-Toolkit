"""Machine learning functionality for CSI Toolkit."""

from .training import ModelTrainer
from .inference import ModelPredictor
from .evaluation import ModelEvaluator

# Import registries to trigger auto-registration
from .models import registry as model_registry
from .metrics import registry as metric_registry

__all__ = [
    'ModelTrainer',
    'ModelPredictor',
    'ModelEvaluator',
    'model_registry',
    'metric_registry',
]