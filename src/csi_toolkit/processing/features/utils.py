"""Utility functions for feature extraction."""

from typing import List
import numpy as np
from ..windowing import CSISample


def get_sample_mean_amplitudes(samples: List[CSISample]) -> List[float]:
    """
    Calculate mean amplitude across all subcarriers for each sample.

    This aggregates across all subcarriers to make features robust to
    single-subcarrier fluctuations.

    Args:
        samples: List of CSI samples

    Returns:
        List of mean amplitude values (one per sample)

    Example:
        For a sample with 64 subcarriers:
        - amplitudes = [10.5, 11.2, ..., 9.8]  (64 values)
        - result = np.mean([10.5, 11.2, ..., 9.8])  (single value)
    """
    return [np.mean(sample.amplitudes) for sample in samples]


def get_amplitudes_matrix(samples: List[CSISample]) -> np.ndarray:
    """
    Get all subcarrier amplitudes as a 2D matrix.

    Args:
        samples: List of CSI samples

    Returns:
        numpy array of shape (n_samples, n_subcarriers)
        where n_subcarriers is typically 64

    Note:
        Handles samples with inconsistent subcarrier counts by
        padding shorter arrays or truncating longer ones to match
        the most common length.
    """
    if not samples:
        return np.array([])

    # Find the most common length
    lengths = [len(sample.amplitudes) for sample in samples]
    most_common_len = max(set(lengths), key=lengths.count)

    # Build matrix, padding/truncating as needed
    rows = []
    for sample in samples:
        amps = sample.amplitudes
        if len(amps) == most_common_len:
            rows.append(amps)
        elif len(amps) < most_common_len:
            # Pad with zeros
            padded = list(amps) + [0.0] * (most_common_len - len(amps))
            rows.append(padded)
        else:
            # Truncate
            rows.append(amps[:most_common_len])

    return np.array(rows)


def get_subcarrier_timeseries(samples: List[CSISample], subcarrier_idx: int) -> np.ndarray:
    """
    Get amplitude time series for a specific subcarrier.

    Args:
        samples: List of CSI samples
        subcarrier_idx: Index of the subcarrier (0-63)

    Returns:
        1D numpy array of amplitudes for the specified subcarrier
    """
    return np.array([sample.amplitudes[subcarrier_idx] for sample in samples])
