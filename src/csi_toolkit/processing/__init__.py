"""Signal processing utilities for CSI Toolkit."""

from .amplitude import (
    calculate_amplitudes,
    compute_mean_amplitude,
    compute_amplitude_statistics,
    extract_subcarrier_amplitudes,
)

from .features import (
    extract_features,
)

__all__ = [
    # Amplitude processing
    'calculate_amplitudes',
    'compute_mean_amplitude',
    'compute_amplitude_statistics',
    'extract_subcarrier_amplitudes',
    # Feature extraction
    'extract_features',
]