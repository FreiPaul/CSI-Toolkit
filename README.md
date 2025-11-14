# CSI Toolkit

A modular toolkit for WiFi Channel State Information (CSI) data collection, processing, and machine learning-based activity recognition.

## Overview

CSI Toolkit provides a complete pipeline for CSI-based sensing applications. Collect CSI data from ESP32 devices, process it into features, train machine learning models, and deploy them for real-time activity recognition.

## Features

- **Real-time Data Collection**: Acquire CSI data from ESP32 devices via serial port with live keyboard labeling
- **Live Visualization**: Plot CSI amplitudes in real-time with filtering options
- **Feature Extraction**: Convert raw CSI data into windowed features with non-overlapping windows
- **Machine Learning Pipeline**: Train, evaluate, and deploy ML models for activity recognition
- **Live Inference**: Real-time predictions during data collection
- **Modular Architecture**: Extensible system for adding custom features, models, and metrics

## Quick Start

### Installation

```bash
git clone https://github.com/FreiPaul/CSI-Collector
cd "CSI-Collector"

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e .            # Core functionality
pip install -e ".[ml]"      # ML functionality
```

### Basic Workflow

```bash
# 1. Collect labeled data (press keys 0-9 to label activities)
python -m csi_toolkit collect

# 2. Visualize collected data
python -m csi_toolkit plot data/csi_log_20250113_120000.csv

# 3. Extract features from raw data
python -m csi_toolkit process data/csi_log_20250113_120000.csv features/output.csv --labeled

# 4. Train a model
python -m csi_toolkit train features/output.csv

# 5. Evaluate model performance
python -m csi_toolkit evaluate --dataset features/output.csv --model-dir models/model_20250113_143022

# 6. Run live inference during collection
python -m csi_toolkit collect --live-inference --model-dir models/model_20250113_143022
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `collect` | Collect CSI data from serial port with optional live inference |
| `plot` | Real-time visualization with filtering options |
| `process` | Extract features from raw CSI data |
| `train` | Train machine learning models |
| `inference` | Run batch predictions on feature data |
| `evaluate` | Compute evaluation metrics on test data |

## Documentation

- [Installation Guide](docs/user-guide/installation.md)
- [Quick Start Tutorial](docs/user-guide/quickstart.md)
- [Data Collection Guide](docs/user-guide/data-collection.md)
- [Machine Learning Guide](docs/user-guide/machine-learning.md)
- [Complete CLI Reference](docs/user-guide/cli-reference.md)
- [Adding Custom Features](docs/developer-guide/adding-features.md)
- [Adding Custom Models](docs/developer-guide/adding-models.md)
- [Architecture Overview](docs/developer-guide/architecture.md)

See [docs/](docs/) for complete documentation.

## Workflow Overview

```
Collection → Processing → Training → Evaluation/Inference
```

1. **Collection**: Acquire CSI data from ESP32 via serial port
2. **Processing**: Convert raw samples into windowed features
3. **Training**: Train ML models on labeled features
4. **Deployment**: Run inference (batch or real-time)

## Extensibility

CSI Toolkit is built for modularity and can be easily extended:

- **Custom Features**: Add new feature extractors via registry system
- **Custom Models**: Integrate any ML model implementing the base interface
- **Custom Metrics**: Define domain-specific evaluation metrics
- **Custom Filters**: Add signal processing filters for visualization

The registry-based architecture allows extensions without modifying core code.

## Project Information

Developed by Paul Herrmann for his Seminar Paper at RWTH Aachen University.

For questions or issues, please contact: paul.herrmann@rwth-aachen.de
