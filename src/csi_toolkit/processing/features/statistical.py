"""Statistical features for CSI amplitude analysis."""

from typing import List

import numpy as np
from scipy import stats

from ..windowing import CSISample
from .registry import registry
from .utils import get_sample_mean_amplitudes


@registry.register('range_amp', description='Range of carrier amplitude (max - min) in window')
def range_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate amplitude range in window.

    Larger ranges indicate more dynamic activities (e.g., walking).
    Smaller ranges indicate static activities (e.g., sitting).
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(np.max(sample_means) - np.min(sample_means))


@registry.register('median_amp', description='Median carrier amplitude in window')
def median_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate median amplitude in window.

    More robust to outliers than mean.
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(np.median(sample_means))


@registry.register('var_amp', description='Variance of carrier amplitude in window')
def variance_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate variance of amplitude in window.

    Alternative to std_amp, useful for some ML models.
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(np.var(sample_means))


@registry.register('iqr_amp', description='Interquartile range (75th - 25th percentile) of amplitude')
def iqr_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate interquartile range of amplitude.

    Robust measure of spread, less sensitive to outliers than range.
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    q75, q25 = np.percentile(sample_means, [75, 25])
    return float(q75 - q25)


@registry.register('skewness_amp', description='Skewness of amplitude distribution')
def skewness_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate skewness of amplitude distribution.

    Measures asymmetry of the distribution.
    Different activities produce different distribution shapes.
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(stats.skew(sample_means))


@registry.register('kurtosis_amp', description='Kurtosis of amplitude distribution')
def kurtosis_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate kurtosis of amplitude distribution.

    Measures "peakedness" of the distribution.
    High kurtosis indicates heavy tails/outliers.
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(stats.kurtosis(sample_means))


@registry.register('energy_amp', description='Total energy (sum of squared amplitudes) in window')
def energy_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate total signal energy in window.

    Sum of squared amplitudes. Higher during movement.
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(np.sum(np.square(sample_means)))


@registry.register('rms_amp', description='Root mean square of amplitude in window')
def rms_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate root mean square of amplitude.

    Alternative energy measure, normalized by window size.
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(np.sqrt(np.mean(np.square(sample_means))))
