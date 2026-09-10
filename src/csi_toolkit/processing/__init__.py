"""Signal processing utilities for CSI Toolkit."""

from .amplitude import (
    calculate_amplitudes,
    compute_mean_amplitude,
    compute_amplitude_statistics,
    extract_subcarrier_amplitudes,
)

# Windowed feature extraction system
from .windowing import CSISample, WindowData, create_windows
from .feature_extractor import FeatureExtractor

__all__ = [
    # Amplitude processing
    'calculate_amplitudes',
    'compute_mean_amplitude',
    'compute_amplitude_statistics',
    'extract_subcarrier_amplitudes',
    # Windowed feature extraction
    'CSISample',
    'WindowData',
    'create_windows',
    'FeatureExtractor',
]