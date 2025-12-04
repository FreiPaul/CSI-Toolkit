"""Per-subcarrier analysis features for CSI data."""

from typing import List

import numpy as np

from ..windowing import CSISample
from .registry import registry
from .utils import get_amplitudes_matrix


@registry.register(
    'mean_subcarrier_var',
    description='Mean variance across all subcarriers (per-sample variability)'
)
def mean_subcarrier_variance(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate mean variance across all subcarriers.

    For each sample, compute variance across 64 subcarriers,
    then average these variances across the window.

    High values indicate samples have diverse subcarrier responses.
    """
    amp_matrix = get_amplitudes_matrix(current_samples)  # (n_samples, 64)
    # Variance across subcarriers for each sample
    per_sample_var = np.var(amp_matrix, axis=1)
    return float(np.mean(per_sample_var))


@registry.register(
    'max_subcarrier_var',
    description='Maximum per-sample subcarrier variance in window'
)
def max_subcarrier_variance(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate maximum subcarrier variance across samples.

    Identifies the sample with most diverse subcarrier responses.
    """
    amp_matrix = get_amplitudes_matrix(current_samples)
    per_sample_var = np.var(amp_matrix, axis=1)
    return float(np.max(per_sample_var))


@registry.register(
    'min_subcarrier_var',
    description='Minimum per-sample subcarrier variance in window'
)
def min_subcarrier_variance(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate minimum subcarrier variance across samples.

    Identifies the sample with most uniform subcarrier responses.
    """
    amp_matrix = get_amplitudes_matrix(current_samples)
    per_sample_var = np.var(amp_matrix, axis=1)
    return float(np.min(per_sample_var))


@registry.register(
    'subcarrier_range',
    description='Range of per-subcarrier mean amplitudes'
)
def subcarrier_amplitude_range(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate range of mean amplitudes across subcarriers.

    For each subcarrier, compute mean amplitude across window,
    then find range (max - min) of these means.

    Indicates how differently subcarriers respond to the activity.
    """
    amp_matrix = get_amplitudes_matrix(current_samples)  # (n_samples, 64)
    # Mean amplitude for each subcarrier across the window
    per_subcarrier_mean = np.mean(amp_matrix, axis=0)
    return float(np.max(per_subcarrier_mean) - np.min(per_subcarrier_mean))
