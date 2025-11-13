"""Amplitude calculation and processing utilities."""

import math
from typing import List, Tuple, Dict, Optional
import statistics


def calculate_amplitudes(q_i_values: List[float]) -> List[float]:
    """
    Calculate amplitudes from interleaved Q, I values.

    The input contains interleaved [Q, I, Q, I, ...] values.
    Amplitude is calculated as sqrt(I^2 + Q^2) for each pair.

    Args:
        q_i_values: List of interleaved Q, I values

    Returns:
        List of amplitude values

    Raises:
        ValueError: If the input length is odd
    """
    if len(q_i_values) % 2 != 0:
        raise ValueError(f"Expected even number of Q,I values, got {len(q_i_values)}")

    amplitudes = []
    for i in range(0, len(q_i_values), 2):
        q_value = q_i_values[i]
        i_value = q_i_values[i + 1] if i + 1 < len(q_i_values) else 0

        # Calculate amplitude: sqrt(I^2 + Q^2)
        amplitude = math.sqrt(i_value**2 + q_value**2)
        amplitudes.append(amplitude)

    return amplitudes


def compute_mean_amplitude(amplitudes: List[float]) -> float:
    """
    Compute the mean amplitude across all subcarriers.

    Args:
        amplitudes: List of amplitude values

    Returns:
        Mean amplitude value
    """
    if not amplitudes:
        return 0.0
    return statistics.mean(amplitudes)


def compute_amplitude_statistics(amplitudes: List[float]) -> Dict[str, float]:
    """
    Compute statistics for amplitude values.

    Args:
        amplitudes: List of amplitude values

    Returns:
        Dictionary containing mean, std, min, max statistics
    """
    if not amplitudes:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
        }

    return {
        'mean': statistics.mean(amplitudes),
        'std': statistics.stdev(amplitudes) if len(amplitudes) > 1 else 0.0,
        'min': min(amplitudes),
        'max': max(amplitudes),
        'median': statistics.median(amplitudes),
    }


def extract_subcarrier_amplitudes(
    amplitude_series: List[List[float]],
    subcarrier_index: int
) -> List[float]:
    """
    Extract amplitude values for a specific subcarrier across time.

    Args:
        amplitude_series: List of amplitude arrays (one per time point)
        subcarrier_index: Index of the subcarrier to extract

    Returns:
        List of amplitude values for the specified subcarrier
    """
    subcarrier_values = []

    for amplitudes in amplitude_series:
        if amplitudes and 0 <= subcarrier_index < len(amplitudes):
            subcarrier_values.append(amplitudes[subcarrier_index])
        else:
            # Use 0 if subcarrier is out of range or amplitudes are empty
            subcarrier_values.append(0.0)

    return subcarrier_values


def normalize_amplitudes(
    amplitudes: List[float],
    method: str = 'max'
) -> List[float]:
    """
    Normalize amplitude values.

    Args:
        amplitudes: List of amplitude values
        method: Normalization method ('max', 'zscore', 'minmax')

    Returns:
        Normalized amplitude values
    """
    if not amplitudes:
        return []

    if method == 'max':
        max_val = max(amplitudes)
        if max_val > 0:
            return [a / max_val for a in amplitudes]
        return amplitudes

    elif method == 'zscore':
        mean = statistics.mean(amplitudes)
        std = statistics.stdev(amplitudes) if len(amplitudes) > 1 else 1.0
        if std > 0:
            return [(a - mean) / std for a in amplitudes]
        return [a - mean for a in amplitudes]

    elif method == 'minmax':
        min_val = min(amplitudes)
        max_val = max(amplitudes)
        range_val = max_val - min_val
        if range_val > 0:
            return [(a - min_val) / range_val for a in amplitudes]
        return [0.5 for _ in amplitudes]

    else:
        raise ValueError(f"Unknown normalization method: {method}")