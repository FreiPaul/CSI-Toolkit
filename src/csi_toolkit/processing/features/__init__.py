"""Feature extraction for windowed CSI data."""

# Import registry first
from .registry import registry, FeatureConfig

# Import feature modules to trigger registration
from . import basic
from . import temporal

# Import utilities
from .utils import get_sample_mean_amplitudes

__all__ = [
    'registry',
    'FeatureConfig',
    'get_sample_mean_amplitudes',
]
