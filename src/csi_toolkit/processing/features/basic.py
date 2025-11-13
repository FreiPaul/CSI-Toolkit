"""Basic window-based features."""

from typing import List
import numpy as np

from ..windowing import CSISample
from .registry import registry
from .utils import get_sample_mean_amplitudes


@registry.register('mean_amp', description='Mean carrier amplitude in window')
def mean_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate mean carrier amplitude in window.

    For each sample: calculate mean amplitude across all subcarriers.
    Then: calculate mean of those means across the window.

    Args:
        current_samples: Samples in current window
        prev_samples: Samples from previous windows (unused)
        next_samples: Samples from next windows (unused)

    Returns:
        Mean amplitude value
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(np.mean(sample_means))


@registry.register('std_amp', description='Standard deviation of carrier amplitude in window')
def std_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate standard deviation of carrier amplitude in window.

    For each sample: calculate mean amplitude across all subcarriers.
    Then: calculate std of those means across the window.

    Args:
        current_samples: Samples in current window
        prev_samples: Samples from previous windows (unused)
        next_samples: Samples from next windows (unused)

    Returns:
        Standard deviation value
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(np.std(sample_means))


@registry.register('max_amp', description='Maximum carrier amplitude in window')
def max_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate maximum carrier amplitude in window.

    For each sample: calculate mean amplitude across all subcarriers.
    Then: find max of those means across the window.

    Args:
        current_samples: Samples in current window
        prev_samples: Samples from previous windows (unused)
        next_samples: Samples from next windows (unused)

    Returns:
        Maximum amplitude value
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(np.max(sample_means))


@registry.register('min_amp', description='Minimum carrier amplitude in window')
def min_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate minimum carrier amplitude in window.

    For each sample: calculate mean amplitude across all subcarriers.
    Then: find min of those means across the window.

    Args:
        current_samples: Samples in current window
        prev_samples: Samples from previous windows (unused)
        next_samples: Samples from next windows (unused)

    Returns:
        Minimum amplitude value
    """
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(np.min(sample_means))
