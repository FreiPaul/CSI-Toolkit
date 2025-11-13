"""Signal processing filters for CSI data visualization."""

from typing import List, Optional, Union
import warnings


def moving_average(
    data: List[float],
    window_size: int = 10
) -> List[float]:
    """
    Apply bidirectional moving average filter.

    This implements a zero-phase filter by applying the moving average
    forward and then backward, preserving signal phase.

    Args:
        data: Input signal data
        window_size: Size of the moving average window

    Returns:
        Filtered signal
    """
    if not data or window_size <= 0:
        return data

    if len(data) < window_size:
        return data

    # Forward pass
    forward_filtered = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        window = data[start:end]
        forward_filtered.append(sum(window) / len(window))

    # Backward pass for zero-phase filtering
    backward_filtered = []
    for i in range(len(forward_filtered) - 1, -1, -1):
        start = i
        end = min(len(forward_filtered), i + window_size)
        window = forward_filtered[start:end]
        backward_filtered.append(sum(window) / len(window))

    # Reverse to get correct order
    backward_filtered.reverse()

    return backward_filtered


def butterworth_lowpass(
    data: List[float],
    sampling_rate: float = 100.0,
    cutoff_freq: float = 2.0,
    order: int = 4
) -> List[float]:
    """
    Apply Butterworth low-pass filter.

    Requires scipy for implementation. Falls back to moving average
    if scipy is not available.

    Args:
        data: Input signal data
        sampling_rate: Sampling rate in Hz
        cutoff_freq: Cutoff frequency in Hz
        order: Filter order

    Returns:
        Filtered signal
    """
    if not data:
        return data

    try:
        import numpy as np
        from scipy import signal

        # Convert to numpy array
        data_array = np.array(data)

        # Design Butterworth filter
        nyquist_freq = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist_freq

        # Ensure cutoff is valid
        if normalized_cutoff >= 1.0:
            warnings.warn(
                f"Cutoff frequency {cutoff_freq} Hz is too high for "
                f"sampling rate {sampling_rate} Hz. Using 0.9 * Nyquist instead."
            )
            normalized_cutoff = 0.9

        # Create filter
        b, a = signal.butter(order, normalized_cutoff, btype='low')

        # Apply filter (zero-phase using filtfilt)
        filtered_data = signal.filtfilt(b, a, data_array)

        return filtered_data.tolist()

    except ImportError:
        warnings.warn(
            "scipy not available. Falling back to moving average filter. "
            "Install scipy for Butterworth filtering: pip install scipy"
        )
        # Approximate window size based on cutoff frequency
        window_size = int(sampling_rate / (2 * cutoff_freq))
        return moving_average(data, window_size)


def apply_filter(
    data: List[float],
    filter_type: str = 'moving_average',
    **kwargs
) -> List[float]:
    """
    Apply a filter to the data.

    Args:
        data: Input signal data
        filter_type: Type of filter ('moving_average', 'butterworth', 'none')
        **kwargs: Additional arguments for the filter

    Returns:
        Filtered signal
    """
    if filter_type == 'none' or not filter_type:
        return data

    if filter_type == 'moving_average':
        window_size = kwargs.get('window_size', 10)
        return moving_average(data, window_size)

    elif filter_type == 'butterworth':
        sampling_rate = kwargs.get('sampling_rate', 100.0)
        cutoff_freq = kwargs.get('cutoff_freq', 2.0)
        order = kwargs.get('order', 4)
        return butterworth_lowpass(data, sampling_rate, cutoff_freq, order)

    else:
        warnings.warn(f"Unknown filter type: {filter_type}")
        return data


def median_filter(
    data: List[float],
    window_size: int = 5
) -> List[float]:
    """
    Apply median filter to remove spikes.

    Args:
        data: Input signal data
        window_size: Size of the median filter window

    Returns:
        Filtered signal
    """
    if not data or window_size <= 0:
        return data

    import statistics

    filtered = []
    half_window = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        filtered.append(statistics.median(window))

    return filtered


def savitzky_golay_filter(
    data: List[float],
    window_size: int = 11,
    poly_order: int = 3
) -> List[float]:
    """
    Apply Savitzky-Golay filter for smoothing.

    Args:
        data: Input signal data
        window_size: Size of the filter window (must be odd)
        poly_order: Order of the polynomial fit

    Returns:
        Filtered signal
    """
    if not data:
        return data

    try:
        import numpy as np
        from scipy.signal import savgol_filter

        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1

        # Ensure we have enough data points
        if len(data) <= window_size:
            return data

        # Apply filter
        data_array = np.array(data)
        filtered = savgol_filter(data_array, window_size, poly_order)
        return filtered.tolist()

    except ImportError:
        warnings.warn(
            "scipy not available for Savitzky-Golay filter. "
            "Using moving average instead."
        )
        return moving_average(data, window_size)