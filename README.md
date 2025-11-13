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

### Labeled Data Collection

Collect labeled CSI data for machine learning by pressing number keys during collection:

**Keyboard Controls:**
- Keys **1-9**: Set current class label (1-9)
- Key **0**: Return to unlabeled state (background)
- Labels are recorded in real-time with each CSI sample

```bash
# Start collection with labeling support (automatic)
python -m csi_toolkit collect

# During collection, press number keys to label activities:
# Press 1 when performing Activity A
# Press 2 when performing Activity B
# Press 0 to mark unlabeled/background periods
```

**Workflow Example:**
1. Start data collection: `python -m csi_toolkit collect`
2. Wait for baseline data (label stays at 0)
3. Begin Activity 1, press key `1`
4. Perform activity for several seconds
5. Return to baseline, press key `0`
6. Begin Activity 2, press key `2`
7. Stop collection with Ctrl+C

The CSV file will include a `label` column with the class label (0-9) for each sample.

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

# Process labeled data (includes labels, filters transitions)
python -m csi_toolkit process input.csv output.csv --labeled

# Adjust transition buffer (default: 1 window before/after)
python -m csi_toolkit process input.csv output.csv --labeled --transition-buffer 2
```

#### Labeled Mode Processing

When processing labeled data collected with keyboard input:

**Enable Labeled Mode:**
```bash
python -m csi_toolkit process labeled_data.csv features.csv --labeled
```

**Behavior:**
- Reads label from each sample in the CSV
- Adds `label` column to output features
- **Discards transition windows**: Windows where ANY sample has a different label
- **Applies buffer**: Also discards N windows before/after each transition (default: 1)

**Transition Window Handling:**

Windows are discarded if they contain label transitions, plus a configurable buffer:

```
Window:    0  1  2  3  4  5  6  7  8  9
Label:     1  1  1  2  2  2  2  3  3  3
                    ^              ^
                transition      transition

With --transition-buffer 1 (default):
Discarded: windows 2, 3, 4 and windows 6, 7, 8
Output:    windows 0, 1, 5, 9 (with labels 1, 1, 2, 3)
```

This ensures clean, homogeneous windows for training machine learning models.

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

**Standard Mode:**
```csv
window_id,start_seq,end_seq,mean_amp,std_amp,max_amp,min_amp,mean_last3,std_last3,mean_last10
9,900,999,45.2,3.1,52.3,38.1,45.0,1.9,44.8
10,1000,1099,46.1,2.9,51.8,39.0,45.5,0.7,45.1
```

**Labeled Mode (--labeled):**
```csv
window_id,start_seq,end_seq,label,mean_amp,std_amp,max_amp,min_amp,mean_last3,std_last3,mean_last10
9,900,999,1,45.2,3.1,52.3,38.1,45.0,1.9,44.8
10,1000,1099,1,46.1,2.9,51.8,39.0,45.5,0.7,45.1
```

**Columns:**
- `window_id`: Index of the window
- `start_seq`: Sequence number of first sample in window
- `end_seq`: Sequence number of last sample in window
- `label`: Class label (0-9) when using --labeled flag
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

## Machine Learning

CSI Toolkit includes a complete machine learning pipeline for activity recognition using CSI data. The ML functionality is modular, allowing custom models and metrics to be easily integrated.

### Installation

ML functionality requires additional dependencies. Install with:

```bash
pip install -e ".[ml]"
```

This installs:
- **scikit-learn**: For MLP models and evaluation metrics
- Optimized for Apple Silicon through Accelerate framework

### ML Workflow Overview

The complete workflow from data collection to deployment:

```bash
# 1. Collect labeled data with keyboard input
python -m csi_toolkit collect
# (Press keys 0-9 during collection to label activities)

# 2. Extract features with labeled mode
python -m csi_toolkit process data/current.csv features/labeled_features.csv --labeled

# 3. Train a model
python -m csi_toolkit train features/labeled_features.csv

# 4. Evaluate model performance
python -m csi_toolkit evaluate --dataset features/labeled_features.csv --model-dir models/model_20250113_143022

# 5. Run inference on new data
python -m csi_toolkit inference --dataset features/new_data.csv --model-dir models/model_20250113_143022
```

### Training

Train a machine learning model on labeled feature data:

```bash
# Basic training with defaults (MLP model, 70/15/15 split)
python -m csi_toolkit train features/labeled_features.csv

# Specify model type
python -m csi_toolkit train features/labeled_features.csv --model mlp

# Custom output directory
python -m csi_toolkit train features/labeled_features.csv --output-dir models/my-model

# Custom train/val/test split
python -m csi_toolkit train features.csv --train-split 0.8 --val-split 0.1 --test-split 0.1

# Override hyperparameters
python -m csi_toolkit train features.csv --params hidden_layer_sizes=(200,100),max_iter=1000

# List available model types
python -m csi_toolkit train --list-models
```

**What happens during training:**
1. Loads features from CSV (automatically detects feature columns)
2. Validates that `label` column exists
3. Splits data into train/validation/test sets (stratified)
4. Trains the model
5. Creates timestamped directory (e.g., `models/model_20250113_143022/`)
6. Saves model, metadata, and training log

### Inference

Generate predictions on new (unlabeled) data:

```bash
# Basic inference
python -m csi_toolkit inference --dataset features/new_data.csv --model-dir models/model_20250113_143022

# Specify output file
python -m csi_toolkit inference \
  --dataset features/new_data.csv \
  --model-dir models/model_20250113_143022 \
  --output predictions.csv

# Include class probabilities
python -m csi_toolkit inference \
  --dataset features/new_data.csv \
  --model-dir models/model_20250113_143022 \
  --probabilities
```

**Output Format:**

Standard output (predictions only):
```csv
window_id,start_seq,end_seq,predicted_label
0,0,99,1
1,100,199,1
2,200,299,2
```

With `--probabilities`:
```csv
window_id,start_seq,end_seq,predicted_label,prob_class_1,prob_class_2,prob_class_3
0,0,99,1,0.92,0.05,0.03
1,100,199,1,0.87,0.10,0.03
2,200,299,2,0.05,0.89,0.06
```

### Evaluation

Compute performance metrics on labeled test data:

```bash
# Evaluate with all metrics
python -m csi_toolkit evaluate --dataset features/test.csv --model-dir models/model_20250113_143022

# Compute specific metrics
python -m csi_toolkit evaluate \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022 \
  --metrics accuracy,f1_macro,f1_per_class

# Custom output paths
python -m csi_toolkit evaluate \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022 \
  --output-json my_eval.json \
  --output-txt my_eval.txt

# List available metrics
python -m csi_toolkit evaluate --list-metrics
```

**Available Metrics:**
- `accuracy`: Overall classification accuracy
- `precision_macro`: Macro-averaged precision
- `precision_micro`: Micro-averaged precision
- `recall_macro`: Macro-averaged recall
- `recall_micro`: Micro-averaged recall
- `f1_macro`: Macro-averaged F1 score
- `f1_micro`: Micro-averaged F1 score
- `precision_per_class`: Precision for each class individually
- `recall_per_class`: Recall for each class individually
- `f1_per_class`: F1 score for each class individually
- `classification_report`: Comprehensive sklearn report
- `confusion_matrix`: Confusion matrix with class labels

**Evaluation outputs:**

JSON (`evaluation_20250113_143022.json`):
```json
{
  "accuracy": 0.9235,
  "f1_macro": 0.9180,
  "precision_macro": 0.9201,
  "recall_macro": 0.9162,
  "f1_per_class": {
    "1": 0.95,
    "2": 0.91,
    "3": 0.89
  }
}
```

Text report (`evaluation_20250113_143022.txt`):
```
CSI Toolkit Model Evaluation Report
==================================================

Model: mlp
Model Directory: models/model_20250113_143022
Evaluation Date: 2025-01-13T14:30:22

Model Info:
  Classes: [1, 2, 3]
  Features: 7

Metrics:
  accuracy: 0.9235
  f1_macro: 0.9180
  ...
```

### Model Directory Structure

Each trained model creates a directory containing:

```
models/model_20250113_143022/
├── model.pkl                    # Trained model (pickle format)
├── metadata.json                # Model configuration and info
├── training_log.txt             # Human-readable training summary
├── predictions_20250113_150000.csv  # Inference outputs (if run)
└── evaluation_20250113_160000.json  # Evaluation results (if run)
```

**metadata.json** structure:
```json
{
  "model_type": "mlp",
  "features": ["mean_amp", "std_amp", "max_amp", "min_amp", "mean_last3", "std_last3", "mean_last10"],
  "n_features": 7,
  "n_classes": 3,
  "class_names": [1, 2, 3],
  "training_date": "2025-01-13T14:30:22.123456",
  "hyperparameters": {
    "hidden_layer_sizes": [100, 50],
    "max_iter": 500,
    "learning_rate": "adaptive",
    "solver": "adam",
    "random_state": 42
  },
  "model_specific_params": {
    "n_layers": 3,
    "n_iter": 245,
    "loss": 0.0823
  },
  "splits": {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
  },
  "random_seed": 42,
  "val_accuracy": 0.9235
}
```

This metadata ensures you can always reproduce training and understand what features the model expects.

### Custom Models

To implement a custom model, inherit from `BaseModel` and register it:

```python
# In src/csi_toolkit/ml/models/custom_models.py

import numpy as np
from .base import BaseModel
from .registry import registry

@registry.register(
    'my_model',
    description='My custom activity recognition model',
    default_params={
        'param1': 10,
        'param2': 0.01,
    }
)
class MyCustomModel(BaseModel):
    """Custom model implementation."""

    def __init__(self, param1=10, param2=0.01, **kwargs):
        super().__init__(param1=param1, param2=param2, **kwargs)
        # Initialize your model here
        self.internal_model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MyCustomModel':
        """Train the model."""
        # Store metadata
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Train your model
        # self.internal_model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self._validate_fitted()
        self._validate_features(X)

        # Generate predictions
        # return self.internal_model.predict(X)
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probabilities."""
        self._validate_fitted()
        self._validate_features(X)

        # Return probabilities or None if not supported
        # return self.internal_model.predict_proba(X)
        return None

    def get_model_specific_params(self) -> dict:
        """Get model-specific parameters for metadata."""
        if not self.is_fitted:
            return {}

        return {
            'param1': self.hyperparameters.get('param1'),
            'param2': self.hyperparameters.get('param2'),
            # Add any learned parameters
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        import pickle
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            'internal_model': self.internal_model,
            'hyperparameters': self.hyperparameters,
            'n_features_': self.n_features_,
            'n_classes_': self.n_classes_,
            'classes_': self.classes_,
            'is_fitted': self.is_fitted,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        import pickle
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.internal_model = model_data['internal_model']
        self.hyperparameters = model_data['hyperparameters']
        self.n_features_ = model_data['n_features_']
        self.n_classes_ = model_data['n_classes_']
        self.classes_ = model_data['classes_']
        self.is_fitted = model_data['is_fitted']
```

**Register your model** in `ml/models/__init__.py`:

```python
from . import custom_models  # Trigger registration
```

**Use your model:**

```bash
python -m csi_toolkit train features.csv --model my_model --params param1=20,param2=0.005
```

### Custom Metrics

To add custom evaluation metrics, register them with the metric registry:

```python
# In src/csi_toolkit/ml/metrics/custom_metrics.py

import numpy as np
from .registry import registry

@registry.register('my_metric', description='My custom evaluation metric')
def my_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute a custom metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Metric value (float or dict for per-class metrics)
    """
    # Implement your metric calculation
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    return float(correct / total)

# For metrics that need probabilities instead of labels
@registry.register(
    'my_proba_metric',
    description='Metric requiring probability predictions',
    requires_proba=True
)
def my_proba_metric(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute a metric using probability predictions.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities (n_samples, n_classes)

    Returns:
        Metric value
    """
    # Implement metric using probabilities
    pass
```

**Register your metrics** in `ml/metrics/__init__.py`:

```python
from . import custom_metrics  # Trigger registration
```

**Use your metrics:**

```bash
python -m csi_toolkit evaluate --dataset test.csv --model-dir models/model_X --metrics my_metric,accuracy
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
