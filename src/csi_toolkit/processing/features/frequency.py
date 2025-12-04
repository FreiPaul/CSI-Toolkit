"""Frequency domain features for CSI data analysis.

These features extract information from the frequency spectrum of the amplitude signal,
which is useful for detecting periodic activities like walking.
"""

from typing import List

import numpy as np
from scipy import signal
from scipy.stats import entropy

from ..windowing import CSISample
from .registry import registry
from .utils import get_sample_mean_amplitudes


@registry.register(
    'dominant_freq',
    description='Dominant frequency in amplitude signal (normalized)'
)
def dominant_frequency(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate dominant frequency of amplitude signal.

    Uses FFT to find the frequency with highest power.
    Walking typically shows ~1-2 Hz (normalized to sampling rate).
    Static activities show near-zero dominant frequency.

    Returns normalized frequency (0 to 0.5, where 0.5 = Nyquist).
    """
    sample_means = np.array(get_sample_mean_amplitudes(current_samples))

    if len(sample_means) < 4:
        return 0.0

    # Remove DC component
    centered = sample_means - np.mean(sample_means)

    # Compute FFT
    fft_vals = np.abs(np.fft.rfft(centered))

    # Find index of maximum (excluding DC at index 0)
    if len(fft_vals) > 1:
        peak_idx = np.argmax(fft_vals[1:]) + 1
        # Normalize frequency to [0, 0.5]
        normalized_freq = peak_idx / len(sample_means)
        return float(normalized_freq)

    return 0.0


@registry.register(
    'spectral_entropy',
    description='Entropy of power spectral density'
)
def spectral_entropy(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate spectral entropy of amplitude signal.

    Measures the "disorder" of the frequency spectrum.
    Lower entropy = more concentrated spectrum (periodic signal, e.g., walking)
    Higher entropy = more spread spectrum (random signal, e.g., noise)
    """
    sample_means = np.array(get_sample_mean_amplitudes(current_samples))

    if len(sample_means) < 4:
        return 0.0

    # Remove DC component
    centered = sample_means - np.mean(sample_means)

    # Compute power spectral density
    freqs, psd = signal.welch(centered, nperseg=min(len(centered), 64))

    # Normalize PSD to form a probability distribution
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd

    # Calculate Shannon entropy
    # Filter out zero values to avoid log(0)
    psd_positive = psd_norm[psd_norm > 0]
    if len(psd_positive) > 0:
        return float(entropy(psd_positive))

    return 0.0


@registry.register(
    'spectral_centroid',
    description='Center of mass of the frequency spectrum'
)
def spectral_centroid(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Calculate spectral centroid of amplitude signal.

    Indicates the "center of gravity" of the spectrum.
    Higher values indicate more high-frequency content.
    Lower values indicate more low-frequency content.

    Returns normalized frequency (0 to 0.5).
    """
    sample_means = np.array(get_sample_mean_amplitudes(current_samples))

    if len(sample_means) < 4:
        return 0.0

    # Remove DC component
    centered = sample_means - np.mean(sample_means)

    # Compute FFT magnitude
    fft_vals = np.abs(np.fft.rfft(centered))

    # Create frequency bins (normalized)
    freq_bins = np.arange(len(fft_vals)) / len(sample_means)

    # Compute weighted average of frequencies
    total_magnitude = np.sum(fft_vals)
    if total_magnitude > 0:
        centroid = np.sum(freq_bins * fft_vals) / total_magnitude
        return float(centroid)

    return 0.0
