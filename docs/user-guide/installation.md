# Installation Guide

This guide covers detailed installation instructions for CSI Toolkit.

## Quick Start with Docker (Recommended for Data Processing)

If you only want to process CSI data (e.g., using the sample dataset `data/csi-60k.csv`), Docker is the easiest way to get started. No Python version conflicts, no dependency issues.

```bash
# Clone the repository
git clone https://github.com/FreiPaul/CSI-Collector
cd "CSI-Collector"

# Build the Docker image
docker build -t csi-toolkit:latest .

# Process the sample dataset
docker compose run --rm csi-toolkit process data/csi-60k.csv output/features.csv --labeled

# Train a model
docker compose run --rm csi-toolkit train output/features.csv

# Evaluate
docker compose run --rm csi-toolkit evaluate --dataset output/features.csv --model-dir models/<model_name>
```

The Docker image includes all ML dependencies (TensorFlow, PyTorch, scikit-learn) pre-installed with Python 3.11.

> **Note**: For real-time data collection from an ESP32 device, use the native installation below (Docker cannot access serial ports easily).

---

## Native Installation

### Requirements

- Python 3.8 - 3.12 (Python 3.13+ not yet supported by TensorFlow/PyTorch)
- pip package manager
- Virtual environment (recommended)

## Installation from Source

### Step 1: Clone the Repository

```bash
git clone https://github.com/FreiPaul/CSI-Collector
cd "CSI-Collector"
```

### Step 2: Create Virtual Environment

Creating a virtual environment isolates the package dependencies:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### Step 3: Install Core Dependencies

Install the base package with core functionality:

```bash
pip install -e .
```

This installs:
- Serial communication libraries (pyserial)
- CSV handling utilities
- Data processing tools
- Visualization dependencies (matplotlib, numpy)
- Keyboard input handling (pynput)

### Step 4: Install ML Dependencies (Optional)

For machine learning functionality, install with ML extras:

```bash
pip install -e ".[ml]"
```

This adds:
- scikit-learn for MLP models and metrics
- Optimized NumPy operations (Accelerate framework on Apple Silicon)

## Verifying Installation

Check that the package is installed correctly:

```bash
python -m csi_toolkit info
```

This should display package information including version and available modules.

## Platform-Specific Notes

### macOS

On macOS, the package uses the Accelerate framework for optimized NumPy operations when available.

Serial port devices typically appear as `/dev/cu.usbmodem*`.

### Linux

Serial port permissions may require adding your user to the `dialout` group:

```bash
sudo usermod -a -G dialout $USER
```

Log out and back in for changes to take effect.

Serial port devices typically appear as `/dev/ttyUSB*` or `/dev/ttyACM*`.

### Windows

Serial port devices appear as `COM*` ports (e.g., `COM3`).

You may need to install USB-to-serial drivers for your ESP32 device.

## Development Setup

For development with editable install and additional tools:

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install -e .

# Add src to PYTHONPATH
export PYTHONPATH=src:$PYTHONPATH
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure the virtual environment is activated:

```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate      # Windows
```

### Serial Port Access

If you cannot access the serial port:

1. Check that the device is connected: `ls /dev/cu.*` (macOS) or `ls /dev/ttyUSB*` (Linux)
2. Verify permissions (Linux): Check you are in the `dialout` group
3. Close other applications that may be using the serial port

### ML Dependencies

If ML functionality is not working:

```bash
pip install -e ".[ml]"
```

Verify scikit-learn is installed:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Data Collection](data-collection.md)
- [CLI Reference](cli-reference.md)
