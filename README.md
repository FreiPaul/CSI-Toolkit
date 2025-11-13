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

### Feature Extraction

Convert raw CSI data into windowed features for machine learning:

```bash
# Extract all features with default window size (100 samples)
python -m csi_toolkit process input.csv output.csv

# Custom window size
python -m csi_toolkit process input.csv output.csv --window-size 50

# Extract specific features only
python -m csi_toolkit process input.csv output.csv --features mean_amp,std_amp,mean_last3

# List available features
python -m csi_toolkit process --list-features
```

#### Available Features

**Basic Features** (calculated from single window):
- `mean_amp`: Mean carrier amplitude in window
- `std_amp`: Standard deviation of carrier amplitude in window
- `max_amp`: Maximum carrier amplitude in window
- `min_amp`: Minimum carrier amplitude in window

**Temporal Features** (calculated from multiple windows):
- `mean_last3`: Mean carrier amplitude across last 3 windows (requires 2 previous windows)
- `std_last3`: Standard deviation of carrier amplitude across last 3 windows (requires 2 previous windows)
- `mean_last10`: Mean carrier amplitude across last 10 windows (requires 9 previous windows)

#### Windowing Behavior

- Data is split into **non-overlapping windows** of size N samples
- Each feature aggregates across all subcarriers (robust to single-subcarrier fluctuations)
- For each sample: calculate mean amplitude across all 64 subcarriers
- For each window: apply aggregation function (mean, std, max, min) to those per-sample means

#### Edge Window Handling

Features requiring N previous windows **cannot be calculated for the first N windows**. These windows are skipped in the output.

For example, with `mean_last10` (requires 9 previous windows):
- Input: 1500 samples → 15 windows (window size = 100)
- Output: Only windows 9-14 (6 windows)
- Windows 0-8 are skipped due to insufficient context

#### Output Format

The output CSV contains one row per window with the following structure:

```csv
window_id,start_seq,end_seq,mean_amp,std_amp,max_amp,min_amp,mean_last3,std_last3,mean_last10
9,900,999,45.2,3.1,52.3,38.1,45.0,1.9,44.8
10,1000,1099,46.1,2.9,51.8,39.0,45.5,0.7,45.1
```

- `window_id`: Index of the window
- `start_seq`: Sequence number of first sample in window
- `end_seq`: Sequence number of last sample in window
- Feature columns: One per registered feature

#### Custom Features

To add custom features, edit `src/csi_toolkit/processing/features/`:

```python
# In features/custom.py
from .registry import registry
from .utils import get_sample_mean_amplitudes

@registry.register('my_feature', n_prev=5, description='My custom feature')
def my_feature(current_samples, prev_samples, next_samples):
    """Calculate a custom feature across 6 windows (current + 5 previous)."""
    # Combine all samples
    all_samples = []
    for prev_window in prev_samples:
        all_samples.extend(prev_window)
    all_samples.extend(current_samples)

    # Calculate per-sample means
    sample_means = get_sample_mean_amplitudes(all_samples)

    # Return your computed value
    return float(np.mean(sample_means))
```

Then import your module in `features/__init__.py`:

```python
from . import custom  # Trigger registration
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
- **windowing.py**: Window creation and data structures (CSISample, WindowData)
- **feature_extractor.py**: Feature extraction pipeline
- **features/**: Modular feature system
  - **registry.py**: Feature registration system
  - **basic.py**: Basic window features (mean, std, max, min)
  - **temporal.py**: Multi-window features (mean_last3, mean_last10)
  - **utils.py**: Helper functions for feature calculation

### I/O Module

- **csv_writer.py**: CSV writing with flush control
- **csv_reader.py**: Local CSV reading and tailing
- **ssh_reader.py**: Remote CSV access via SSH

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Authors

Paul Herrmann (paul.herrmann@rwth-aachen.de)
