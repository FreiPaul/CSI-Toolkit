"""Cross-subcarrier correlation features for CSI data.

Correlation between subcarriers is a key feature for HAR:
- Walking causes synchronized amplitude changes across subcarriers (high correlation)
- Static activities show more independent subcarrier behavior (lower correlation)
"""

from typing import List

import numpy as np

from ..windowing import CSISample
from .registry import registry
from .utils import get_amplitudes_matrix


def _compute_adjacent_correlations(amp_matrix: np.ndarray) -> np.ndarray:
    """
    Compute correlation coefficients between adjacent subcarriers.

    Args:
        amp_matrix: (n_samples, n_subcarriers) amplitude matrix

    Returns:
        Array of correlation coefficients between adjacent subcarrier pairs
    """
    n_subcarriers = amp_matrix.shape[1]
    correlations = []

    for i in range(n_subcarriers - 1):
        subcarrier_i = amp_matrix[:, i]
        subcarrier_j = amp_matrix[:, i + 1]

        # Handle constant signals (would cause division by zero)
        if np.std(subcarrier_i) < 1e-10 or np.std(subcarrier_j) < 1e-10:
            correlations.append(0.0)
        else:
            corr = np.corrcoef(subcarrier_i, subcarrier_j)[0, 1]
            # Handle NaN from corrcoef
            correlations.append(0.0 if np.isnan(corr) else corr)

    return np.array(correlations)


@registry.register(
    'mean_subcarrier_corr',
    description='Mean correlation between adjacent subcarriers'
)
def mean_subcarrier_correlation(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate mean correlation between adjacent subcarriers.

    High values indicate synchronized subcarrier responses (common during movement).
    Low values indicate independent subcarrier behavior (common during static activities).
    """
    amp_matrix = get_amplitudes_matrix(current_samples)
    correlations = _compute_adjacent_correlations(amp_matrix)
    return float(np.mean(correlations))


@registry.register(
    'std_subcarrier_corr',
    description='Standard deviation of correlations between adjacent subcarriers'
)
def std_subcarrier_correlation(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate standard deviation of adjacent subcarrier correlations.

    Indicates uniformity of correlation patterns across the frequency band.
    """
    amp_matrix = get_amplitudes_matrix(current_samples)
    correlations = _compute_adjacent_correlations(amp_matrix)
    return float(np.std(correlations))


@registry.register(
    'max_subcarrier_corr',
    description='Maximum correlation between adjacent subcarriers'
)
def max_subcarrier_correlation(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate maximum correlation between adjacent subcarriers.

    Identifies the most strongly correlated subcarrier pair.
    """
    amp_matrix = get_amplitudes_matrix(current_samples)
    correlations = _compute_adjacent_correlations(amp_matrix)
    return float(np.max(correlations))


@registry.register(
    'min_subcarrier_corr',
    description='Minimum correlation between adjacent subcarriers'
)
def min_subcarrier_correlation(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate minimum correlation between adjacent subcarriers.

    Identifies the most independently behaving subcarrier pair.
    """
    amp_matrix = get_amplitudes_matrix(current_samples)
    correlations = _compute_adjacent_correlations(amp_matrix)
    return float(np.min(correlations))
