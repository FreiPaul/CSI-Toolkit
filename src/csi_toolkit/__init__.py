"""
CSI Toolkit - A modular pipeline for Channel State Information data processing.

This package provides tools for:
- Data collection from serial devices
- Real-time visualization with filtering
- Signal processing and feature extraction
- Machine learning integration (future)

Main modules:
- collection: Serial data acquisition
- visualization: Plotting and filtering
- processing: Amplitude calculation and feature extraction
- io: File I/O operations (CSV, SSH)
- core: Shared utilities and constants
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main components for easier access
from .collection import SerialCollector, CollectorConfig
from .visualization import LivePlotter, moving_average, butterworth_lowpass
from .processing import calculate_amplitudes, compute_mean_amplitude, extract_features
from .io import CSVWriter, CSVReader, CSVTailer, SSHReader

# Package metadata
__all__ = [
    # Collection
    'SerialCollector',
    'CollectorConfig',
    # Visualization
    'LivePlotter',
    'moving_average',
    'butterworth_lowpass',
    # Processing
    'calculate_amplitudes',
    'compute_mean_amplitude',
    'extract_features',
    # I/O
    'CSVWriter',
    'CSVReader',
    'CSVTailer',
    'SSHReader',
]

# Optional: Print version when imported
def get_version():
    """Get the package version."""
    return __version__