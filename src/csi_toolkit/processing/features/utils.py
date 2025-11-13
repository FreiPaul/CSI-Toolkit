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
