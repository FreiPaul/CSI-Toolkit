# CSI Toolkit

A modular pipeline toolkit for Channel State Information (CSI) data processing. This package provides a structured framework for collecting, processing, and visualizing CSI data from WiFi devices.

## Features

- **Modular Architecture**: Independent modules that can be used separately or combined
- **Data Collection**: Serial port data acquisition from ESP32 devices
- **Real-time Visualization**: Live plotting with various filtering options
- **Signal Processing**: Amplitude calculations, filtering, and feature extraction
- **Remote Access**: SSH support for reading remote CSV files
- **Extensible**: Easy to add new processing modules and ML models

## Package Structure

```
src/csi_toolkit/
├── __init__.py              # Main package interface
├── __main__.py              # CLI entry point
├── main.py                  # Command-line interface
├── core/                    # Shared utilities
│   ├── constants.py         # Global constants
│   ├── parser.py           # CSI data parsing
│   └── exceptions.py       # Custom exceptions
├── collection/              # Data acquisition
│   ├── serial_collector.py # Serial port reader
│   └── config.py           # Configuration management
├── visualization/           # Plotting and display
│   ├── live_plotter.py     # Real-time plotting
│   └── filters.py          # Signal filters
├── processing/              # Signal processing
│   ├── amplitude.py        # Amplitude calculations
│   └── features.py         # Feature extraction
├── io/                      # File operations
│   ├── csv_writer.py       # CSV writing
│   ├── csv_reader.py       # Local file reading
│   └── ssh_reader.py       # SSH file access
└── ml/                      # Machine learning (future)
    └── __init__.py         # Placeholder
```

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone <your-repo-url>
cd "Seminar Code"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development with editable install
pip install -e .
```

### Using pip (Future)

```bash
pip install csi-toolkit
```

## Usage

### Command-Line Interface

The package provides a unified CLI with multiple subcommands:

```bash
# Add src to PYTHONPATH if not installed
export PYTHONPATH=src:$PYTHONPATH

# Show available commands
python -m csi_toolkit --help

# Show package information
python -m csi_toolkit info
```

### Data Collection

Collect CSI data from a serial device:

```bash
# Use default settings from .env file
python -m csi_toolkit collect

# Specify serial port and baudrate
python -m csi_toolkit collect --port /dev/ttyUSB0 --baudrate 921600

# Custom output directory and flush interval
python -m csi_toolkit collect --output-dir mydata --flush 10
```

Configuration via `.env` file:
```env
SERIAL_PORT=/dev/cu.usbmodem1101
BAUDRATE=921600
FLUSH_INTERVAL=1
```

### Live Visualization

Plot CSI amplitude data in real-time:

```bash
# Plot local CSV file
python -m csi_toolkit plot data/csi_log_20240101_120000.csv

# Plot with specific subcarrier
python -m csi_toolkit plot data/current.csv --subcarrier 15

# Plot remote file via SSH
python -m csi_toolkit plot user@host:/path/to/file.csv

# Apply Butterworth filter
python -m csi_toolkit plot data/current.csv --fs 100 --fc 2.0 --order 4

# Limit display to last 1000 points
python -m csi_toolkit plot data/current.csv --limit 1000
```

### Python API

Use the package programmatically:

```python
import csi_toolkit as csi

# Data Collection
from csi_toolkit.collection import SerialCollector, CollectorConfig

config = CollectorConfig(
    serial_port="/dev/ttyUSB0",
    baudrate=921600,
    flush_interval=10,
    output_dir="data"
)
collector = SerialCollector(config)
collector.start()

# Live Plotting
from csi_toolkit.visualization import LivePlotter

plotter = LivePlotter(
    file_path="data/current.csv",
    subcarrier=10,
    refresh_rate=0.2,
    filter_type='butterworth',
    filter_params={'sampling_rate': 100, 'cutoff_freq': 2.0}
)
plotter.start()

# Processing
from csi_toolkit.processing import calculate_amplitudes, extract_features

# Calculate amplitudes from Q,I values
q_i_values = [1, 2, 3, 4, 5, 6]  # [Q, I, Q, I, Q, I]
amplitudes = calculate_amplitudes(q_i_values)

# Extract features
features = extract_features(amplitudes)

# Filtering
from csi_toolkit.visualization import moving_average, butterworth_lowpass

filtered_data = moving_average(data, window_size=10)
filtered_data = butterworth_lowpass(data, sampling_rate=100, cutoff_freq=2.0)

# CSV I/O
from csi_toolkit.io import CSVWriter, CSVReader, SSHReader

# Write CSV
writer = CSVWriter(output_dir="data", flush_interval=1)
writer.open()
writer.write_row(["data", "row"])
writer.close()

# Read CSV
reader = CSVReader("data/file.csv")
rows = reader.read_all()
last_100 = reader.read_last_n(100)

# Read remote CSV via SSH
ssh_reader = SSHReader("user@host:/path/to/file.csv")
remote_data = ssh_reader.read_all()
```

## Data Format

The CSI data follows this CSV schema:

| Column | Description |
|--------|-------------|
| type | Packet type |
| seq | Sequence number |
| mac | MAC address |
| rssi | Signal strength |
| rate | Data rate |
| noise_floor | Noise floor |
| fft_gain | FFT gain |
| agc_gain | AGC gain |
| channel | WiFi channel |
| local_timestamp | Collection timestamp |
| sig_len | Signal length |
| rx_state | Receive state |
| len | Data length |
| first_word | First data word |
| data | Raw Q,I values (JSON) |
| amplitudes | Calculated amplitudes (JSON) |

## Module Details

### Core Module
- **constants.py**: Global constants (CSV headers, defaults)
- **parser.py**: CSI data parsing utilities
- **exceptions.py**: Custom exception classes

### Collection Module
- **serial_collector.py**: SerialCollector class for data acquisition
- **config.py**: Configuration management with .env support

### Visualization Module
- **live_plotter.py**: Real-time plotting with matplotlib
- **filters.py**: Signal filters (moving average, Butterworth, etc.)

### Processing Module
- **amplitude.py**: Amplitude calculations from Q,I values
- **features.py**: Feature extraction utilities

### I/O Module
- **csv_writer.py**: CSV writing with flush control
- **csv_reader.py**: Local CSV reading and tailing
- **ssh_reader.py**: Remote CSV access via SSH

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=csi_toolkit
```

### Code Style

```bash
# Format code
black src/csi_toolkit

# Check linting
flake8 src/csi_toolkit

# Type checking
mypy src/csi_toolkit
```

## Migration from Original Scripts

The package is backward-compatible with the original `collect.py` and `live_plot.py` scripts:

```bash
# Original collect.py equivalent
python -m csi_toolkit collect

# Original live_plot.py equivalent
python -m csi_toolkit plot <csv_file> --subcarrier 10 --refresh 0.2
```

## Future Enhancements

- **Machine Learning Module**: Activity recognition, gesture detection
- **Advanced Processing**: Wavelet transforms, spectral analysis
- **Database Support**: Store data in databases instead of CSV
- **Web Interface**: Browser-based visualization
- **Real-time Streaming**: WebSocket-based data streaming
- **Batch Processing**: Process multiple files efficiently

## Troubleshooting

### Serial Port Access

If you get permission errors:
```bash
# Linux: Add user to dialout group
sudo usermod -a -G dialout $USER

# macOS: Check port with
ls /dev/cu.*
```

### Missing Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For Butterworth filter (optional)
pip install scipy
```

### Import Errors

Make sure to set PYTHONPATH:
```bash
export PYTHONPATH=src:$PYTHONPATH
```

Or install the package:
```bash
pip install -e .
```

## License

MIT License (or your preferred license)

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Authors

Your Name

## Acknowledgments

- ESP32 CSI Tool developers
- Contributors to the original scripts