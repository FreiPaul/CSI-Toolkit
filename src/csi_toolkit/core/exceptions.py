"""Custom exceptions for CSI Toolkit."""


class CSIToolkitError(Exception):
    """Base exception for CSI Toolkit."""
    pass


class ParsingError(CSIToolkitError):
    """Raised when parsing CSI data fails."""
    pass


class ConnectionError(CSIToolkitError):
    """Raised when connection to device or remote host fails."""
    pass


class ConfigurationError(CSIToolkitError):
    """Raised when configuration is invalid."""
    pass


class DataError(CSIToolkitError):
    """Raised when data is invalid or corrupted."""
    pass