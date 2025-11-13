"""Feature extraction utilities for CSI data."""

from typing import List, Dict, Any
import statistics


def extract_features(amplitudes: List[float], window_size: int = None) -> Dict[str, float]:
    """
    Extract features from amplitude data.

    This is a placeholder for future feature extraction functionality.
    Features might include:
    - Statistical features (mean, std, skewness, kurtosis)
    - Frequency domain features (FFT peaks, spectral centroid)
    - Time domain features (zero crossing rate, peak count)
    - Energy features (total energy, band energy)

    Args:
        amplitudes: List of amplitude values
        window_size: Optional window size for windowed features

    Returns:
        Dictionary of extracted features
    """
    if not amplitudes:
        return {}

    features = {}

    # Basic statistical features
    features['mean'] = statistics.mean(amplitudes)
    features['std'] = statistics.stdev(amplitudes) if len(amplitudes) > 1 else 0.0
    features['min'] = min(amplitudes)
    features['max'] = max(amplitudes)
    features['range'] = features['max'] - features['min']
    features['median'] = statistics.median(amplitudes)

    # Energy feature
    features['energy'] = sum(a**2 for a in amplitudes)

    # TODO: Add more sophisticated features:
    # - Frequency domain features (requires FFT)
    # - Wavelet features
    # - Statistical moments (skewness, kurtosis)
    # - Cross-correlation features (if multiple subcarriers)

    return features


def extract_windowed_features(
    amplitudes: List[float],
    window_size: int,
    overlap: float = 0.5
) -> List[Dict[str, float]]:
    """
    Extract features using sliding window approach.

    Args:
        amplitudes: List of amplitude values
        window_size: Size of the sliding window
        overlap: Overlap ratio (0.0 to 1.0)

    Returns:
        List of feature dictionaries for each window
    """
    if not amplitudes or window_size <= 0:
        return []

    step = int(window_size * (1 - overlap))
    features_list = []

    for i in range(0, len(amplitudes) - window_size + 1, step):
        window = amplitudes[i:i + window_size]
        features = extract_features(window)
        features['window_start'] = i
        features['window_end'] = i + window_size
        features_list.append(features)

    return features_list