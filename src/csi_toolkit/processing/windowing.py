"""Windowing utilities for CSI data processing."""

from dataclasses import dataclass
from typing import List


@dataclass
class CSISample:
    """Single CSI sample from CSV data."""

    seq: int
    timestamp: str
    mac: str
    amplitudes: List[float]  # Amplitude values for all subcarriers

    @classmethod
    def from_csv_row(cls, row: dict) -> 'CSISample':
        """
        Create CSISample from CSV row dictionary.

        Args:
            row: Dictionary from CSV DictReader

        Returns:
            CSISample instance

        Raises:
            ValueError: If row data is invalid
        """
        from ..core.parser import parse_amplitude_json
        from ..processing.amplitude import calculate_amplitudes

        seq = int(row.get('seq', 0))
        timestamp = row.get('local_timestamp', '')
        mac = row.get('mac', '')

        # Try to get pre-calculated amplitudes first
        amplitudes_str = row.get('amplitudes', '')
        if amplitudes_str and amplitudes_str != '[]':
            try:
                amplitudes = parse_amplitude_json(amplitudes_str)
                if amplitudes:
                    return cls(
                        seq=seq,
                        timestamp=timestamp,
                        mac=mac,
                        amplitudes=amplitudes
                    )
            except:
                pass  # Fall through to calculate from Q,I data

        # If no amplitudes, calculate from Q,I data
        data_str = row.get('data', '')
        if not data_str or data_str == '[]':
            raise ValueError("Empty or missing data (Q,I values)")

        qi_values = parse_amplitude_json(data_str)  # This parses Q,I JSON
        if not qi_values:
            raise ValueError("Failed to parse Q,I data")

        # Calculate amplitudes from Q,I values
        amplitudes = calculate_amplitudes(qi_values)

        return cls(
            seq=seq,
            timestamp=timestamp,
            mac=mac,
            amplitudes=amplitudes
        )


@dataclass
class WindowData:
    """A window of CSI samples."""

    window_id: int
    start_seq: int
    end_seq: int
    samples: List[CSISample]


def create_windows(samples: List[CSISample], window_size: int) -> List[WindowData]:
    """
    Split samples into non-overlapping windows.

    Args:
        samples: List of CSI samples
        window_size: Number of samples per window

    Returns:
        List of WindowData objects

    Raises:
        ValueError: If window_size is invalid or insufficient samples
    """
    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}")

    if len(samples) < window_size:
        raise ValueError(
            f"Insufficient samples: need at least {window_size}, got {len(samples)}"
        )

    windows = []
    num_complete_windows = len(samples) // window_size

    for i in range(num_complete_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_samples = samples[start_idx:end_idx]

        window = WindowData(
            window_id=i,
            start_seq=window_samples[0].seq,
            end_seq=window_samples[-1].seq,
            samples=window_samples
        )
        windows.append(window)

    # Note: Incomplete final window is discarded
    if len(samples) % window_size != 0:
        num_discarded = len(samples) % window_size
        print(f"Note: Discarded {num_discarded} samples from incomplete final window")

    return windows
