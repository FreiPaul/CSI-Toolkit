"""Core utilities and constants for CSI Toolkit."""

from .constants import (
    CSV_HEADER,
    DEFAULT_BAUDRATE,
    DEFAULT_FLUSH_INTERVAL,
    DEFAULT_REFRESH_RATE,
    DEFAULT_MAX_POINTS,
    DEFAULT_WINDOW_SIZE,
)

from .parser import (
    parse_csi_line,
    parse_amplitude_json,
    extract_amplitudes_from_row,
)

from .exceptions import (
    CSIToolkitError,
    ParsingError,
    ConnectionError,
)

__all__ = [
    # Constants
    'CSV_HEADER',
    'DEFAULT_BAUDRATE',
    'DEFAULT_FLUSH_INTERVAL',
    'DEFAULT_REFRESH_RATE',
    'DEFAULT_MAX_POINTS',
    'DEFAULT_WINDOW_SIZE',
    # Parser functions
    'parse_csi_line',
    'parse_amplitude_json',
    'extract_amplitudes_from_row',
    # Exceptions
    'CSIToolkitError',
    'ParsingError',
    'ConnectionError',
]