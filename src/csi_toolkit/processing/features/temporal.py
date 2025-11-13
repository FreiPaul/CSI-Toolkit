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
    'mean_last10',
    n_prev=9,
    description='Mean carrier amplitude across last 10 windows (including current)'
)
def mean_amplitude_last10(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate mean carrier amplitude across last 10 windows.

    "Last 10 windows" includes the current window:
    - prev_samples[0..8]: windows i-9 through i-1
    - current_samples: window i

    Args:
        current_samples: Samples in current window
        prev_samples: List of 9 previous windows' samples
        next_samples: Samples from next windows (unused)

    Returns:
        Mean amplitude value across 10 windows
    """
    # Collect all samples from the 10 windows
    all_samples = []
    for prev_window in prev_samples:
        all_samples.extend(prev_window)
    all_samples.extend(current_samples)

    # Calculate per-sample means across subcarriers
    sample_means = get_sample_mean_amplitudes(all_samples)

    # Calculate mean across all samples in all 10 windows
    return float(np.mean(sample_means))
