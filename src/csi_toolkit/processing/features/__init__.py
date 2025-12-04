"""Feature extraction for windowed CSI data."""

# Import registry first
from .registry import registry, FeatureConfig

# Import feature modules to trigger registration
from . import basic
from . import temporal
from . import statistical
from . import subcarrier
from . import correlation
from . import frequency

# Import utilities
from .utils import (
    get_sample_mean_amplitudes,
    get_amplitudes_matrix,
    get_subcarrier_timeseries,
)

__all__ = [
    'registry',
    'FeatureConfig',
    'get_sample_mean_amplitudes',
    'get_amplitudes_matrix',
    'get_subcarrier_timeseries',
]
