"""Shared constants for CSI Toolkit."""

# CSV Header schema
CSV_HEADER = [
    "type",
    "seq",
    "mac",
    "rssi",
    "rate",
    "noise_floor",
    "fft_gain",
    "agc_gain",
    "channel",
    "local_timestamp",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
    "data",
    "amplitudes"
]

# Serial communication defaults
DEFAULT_BAUDRATE = 921600
DEFAULT_SERIAL_PORT = "/dev/cu.usbmodem1101"  # Default for macOS
DEFAULT_FLUSH_INTERVAL = 1  # Flush after every packet for real-time

# Visualization defaults
DEFAULT_REFRESH_RATE = 0.2  # seconds
DEFAULT_MAX_POINTS = 20000  # Maximum points in ring buffer
DEFAULT_WINDOW_SIZE = 10  # Moving average window size
DEFAULT_SUBCARRIER = 10  # Default subcarrier to plot

# Signal processing defaults
DEFAULT_CUTOFF_FREQ = 2.0  # Hz for low-pass filter
DEFAULT_FILTER_ORDER = 4  # Butterworth filter order
DEFAULT_SAMPLING_RATE = 100.0  # Hz (estimated)

# Data format constants
CSI_DATA_PREFIX = "CSI_DATA"
AMPLITUDE_COLUMN = "amplitudes"
SEQ_COLUMN = "seq"
RSSI_COLUMN = "rssi"

# File naming patterns
CSV_FILE_PREFIX = "csi_log"
CSV_TEMP_FILENAME = "current.csv"
CSV_DATE_FORMAT = "%Y%m%d_%H%M%S"