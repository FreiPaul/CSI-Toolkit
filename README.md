# CSI Toolkit

A modular pipeline toolkit for Channel State Information (CSI) data processing. This package provides a structured framework for collecting, processing, and visualizing CSI data from WiFi devices.

## Features

- **Modular Architecture**: Independent modules that can be used separately or combined
- **Data Collection**: Serial port data acquisition from ESP32 devices
- **Real-time Visualization**: Live plotting with various filtering options
- **Signal Processing**: Amplitude calculations, filtering, and feature extraction
- **Remote Access**: SSH support for reading remote CSV files
- **Extensible**: Easy to add new processing modules and ML models

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/FreiPaul/CSI-Collector
cd "CSI-Collector"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development with editable install
pip install -e .
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

## Data Format

The CSI data follows this CSV schema:

| Column          | Description                  |
| --------------- | ---------------------------- |
| type            | Packet type                  |
| seq             | Sequence number              |
| mac             | MAC address                  |
| rssi            | Signal strength              |
| rate            | Data rate                    |
| noise_floor     | Noise floor                  |
| fft_gain        | FFT gain                     |
| agc_gain        | AGC gain                     |
| channel         | WiFi channel                 |
| local_timestamp | Collection timestamp         |
| sig_len         | Signal length                |
| rx_state        | Receive state                |
| len             | Data length                  |
| first_word      | First data word              |
| data            | Raw Q,I values (JSON)        |
| amplitudes      | Calculated amplitudes (JSON) |

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

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Authors

Paul Herrmann (paul.herrmann@rwth-aachen.de)
