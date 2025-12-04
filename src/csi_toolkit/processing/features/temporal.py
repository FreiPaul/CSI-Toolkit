"""Temporal features that use multiple windows."""

from typing import List
import numpy as np

from ..windowing import CSISample
from .registry import registry
from .utils import get_sample_mean_amplitudes


@registry.register(
    'mean_last3',
    n_prev=2,
    description='Mean carrier amplitude across last 3 windows (including current)'
)
def mean_amplitude_last3(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate mean carrier amplitude across last 3 windows.

    "Last 3 windows" includes the current window:
    - prev_samples[0]: window i-2
    - prev_samples[1]: window i-1
    - current_samples: window i

    Args:
        current_samples: Samples in current window
        prev_samples: List of 2 previous windows' samples
        next_samples: Samples from next windows (unused)

    Returns:
        Mean amplitude value across 3 windows
    """
    # Collect all samples from the 3 windows
    all_samples = []
    for prev_window in prev_samples:
        all_samples.extend(prev_window)
    all_samples.extend(current_samples)

    # Calculate per-sample means across subcarriers
    sample_means = get_sample_mean_amplitudes(all_samples)

    # Calculate mean across all samples in all 3 windows
    return float(np.mean(sample_means))


@registry.register(
    'std_last3',
    n_prev=2,
    description='Standard deviation of carrier amplitude across last 3 windows (including current)'
)
def std_amplitude_last3(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate standard deviation of carrier amplitude across last 3 windows.

    "Last 3 windows" includes the current window:
    - prev_samples[0]: window i-2
    - prev_samples[1]: window i-1
    - current_samples: window i

    Args:
        current_samples: Samples in current window
        prev_samples: List of 2 previous windows' samples
        next_samples: Samples from next windows (unused)

    Returns:
        Standard deviation value across 3 windows
    """
    # Collect all samples from the 3 windows
    all_samples = []
    for prev_window in prev_samples:
        all_samples.extend(prev_window)
    all_samples.extend(current_samples)

    # Calculate per-sample means across subcarriers
    sample_means = get_sample_mean_amplitudes(all_samples)

    # Calculate std across all samples in all 3 windows
    return float(np.std(sample_means))


@registry.register(
    'delta_mean',
    n_prev=1,
    description='Change in mean amplitude from previous window'
)
def delta_mean_amplitude(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate change in mean amplitude from previous window.

    Positive values indicate increasing amplitude (approaching movement).
    Negative values indicate decreasing amplitude (receding movement).
    Near zero indicates stable activity.
    """
    current_means = get_sample_mean_amplitudes(current_samples)
    current_mean = np.mean(current_means)

    if prev_samples and len(prev_samples) > 0:
        prev_means = get_sample_mean_amplitudes(prev_samples[0])
        prev_mean = np.mean(prev_means)
        return float(current_mean - prev_mean)

    return 0.0


@registry.register(
    'zero_crossing_rate',
    description='Rate at which amplitude crosses its mean value'
)
def zero_crossing_rate(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate zero-crossing rate of amplitude signal.

    Measures how often the signal crosses its mean value.
    Higher rates indicate oscillating signals (e.g., walking).
    Lower rates indicate stable signals (e.g., sitting).
    """
    sample_means = np.array(get_sample_mean_amplitudes(current_samples))

    # Subtract mean to center around zero
    centered = sample_means - np.mean(sample_means)

    # Count sign changes
    sign_changes = np.sum(np.abs(np.diff(np.sign(centered))) > 0)

    # Normalize by number of samples
    return float(sign_changes / (len(sample_means) - 1)) if len(sample_means) > 1 else 0.0
