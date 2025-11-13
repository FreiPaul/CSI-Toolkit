"""Signal processing utilities for CSI Toolkit."""

from .amplitude import (
    calculate_amplitudes,
    compute_mean_amplitude,
    compute_amplitude_statistics,
    extract_subcarrier_amplitudes,
)

# Legacy feature extraction (simple statistical features)
# Note: There's a naming conflict - features.py file vs features/ directory
# The directory takes precedence, so we need special handling
import importlib.util
import sys
from pathlib import Path

# Try to import extract_features from the features.py file
try:
    _features_file = Path(__file__).parent / 'features.py'
    if _features_file.exists():
        spec = importlib.util.spec_from_file_location("_features_legacy", _features_file)
        _features_legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_features_legacy)
        extract_features = _features_legacy.extract_features
    else:
        extract_features = None
except Exception:
    extract_features = None

# New windowed feature extraction system
from .windowing import CSISample, WindowData, create_windows
from .feature_extractor import FeatureExtractor

__all__ = [
    # Amplitude processing
    'calculate_amplitudes',
    'compute_mean_amplitude',
    'compute_amplitude_statistics',
    'extract_subcarrier_amplitudes',
    # Legacy feature extraction
    'extract_features',
    # Windowed feature extraction
    'CSISample',
    'WindowData',
    'create_windows',
    'FeatureExtractor',
]