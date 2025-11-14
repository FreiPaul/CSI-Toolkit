# Data Format Reference

Complete specification of data formats used in CSI Toolkit.

## Raw CSI Data CSV

### Schema

CSV file produced by the collection module.

| Column | Type | Description |
|--------|------|-------------|
| type | string | Packet type (always "CSI_DATA") |
| seq | int | Sequence number (incrementing) |
| mac | string | MAC address of transmitter |
| rssi | int | Received Signal Strength Indicator (dBm) |
| rate | int | Data rate |
| noise_floor | int | Noise floor measurement |
| fft_gain | int | FFT gain value |
| agc_gain | int | Automatic Gain Control value |
| channel | int | WiFi channel number |
| local_timestamp | string | Local collection timestamp (YYYY-MM-DD HH:MM:SS.mmm) |
| sig_len | int | Signal length |
| rx_state | int | Receiver state |
| len | int | Data length |
| first_word | int | First data word |
| data | string | Raw Q,I values (JSON array) |
| amplitudes | string | Calculated amplitudes (JSON array of 64 floats) |
| label | int | Class label 0-9 (0 = unlabeled) |
| predicted_label | string | Model prediction (if live inference enabled) |

### Example Row

```csv
type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data,amplitudes,label
CSI_DATA,0,1a:00:00:00:00:00,-45,11,0,0,0,6,2025-01-13 14:30:22.123,128,0,384,0,"[[1,2],[3,4],...]","[45.2,46.1,...]",1
```

### Notes

- `amplitudes` array always has 64 values (one per subcarrier)
- `label` column added during collection with keyboard input
- `predicted_label` column only present if live inference is enabled

## Feature CSV

### Schema

CSV file produced by the processing module.

| Column | Type | Description |
|--------|------|-------------|
| window_id | int | Window index (0-based) |
| start_seq | int | Sequence number of first sample in window |
| end_seq | int | Sequence number of last sample in window |
| label | int | Class label (only in labeled mode) |
| [features] | float | One column per extracted feature |

### Example (Standard Mode)

```csv
window_id,start_seq,end_seq,mean_amp,std_amp,max_amp,min_amp,mean_last3,std_last3,mean_last10
9,900,999,45.2,3.1,52.3,38.1,45.0,1.9,44.8
10,1000,1099,46.1,2.9,51.8,39.0,45.5,0.7,45.1
```

### Example (Labeled Mode)

```csv
window_id,start_seq,end_seq,label,mean_amp,std_amp,max_amp,min_amp,mean_last3,std_last3,mean_last10
9,900,999,1,45.2,3.1,52.3,38.1,45.0,1.9,44.8
10,1000,1099,1,46.1,2.9,51.8,39.0,45.5,0.7,45.1
```

### Notes

- Feature columns are dynamic based on selected features
- `label` column required for training
- Edge windows may be skipped if features need context

## Prediction CSV

### Schema

CSV file produced by inference module.

| Column | Type | Description |
|--------|------|-------------|
| window_id | int | Window index |
| start_seq | int | First sample sequence number |
| end_seq | int | Last sample sequence number |
| predicted_label | string/int | Predicted class label |
| prob_class_[X] | float | Probability for class X (if --probabilities flag used) |

### Example (Without Probabilities)

```csv
window_id,start_seq,end_seq,predicted_label
0,0,99,1
1,100,199,1
2,200,299,2
```

### Example (With Probabilities)

```csv
window_id,start_seq,end_seq,predicted_label,prob_class_1,prob_class_2,prob_class_3
0,0,99,1,0.92,0.05,0.03
1,100,199,1,0.87,0.10,0.03
2,200,299,2,0.05,0.89,0.06
```

## Model Metadata JSON

### Schema

JSON file storing model configuration and training information.

```json
{
  "model_type": "string",
  "features": ["list", "of", "feature", "names"],
  "n_features": "int",
  "n_classes": "int",
  "class_names": ["list", "of", "class", "labels"],
  "training_date": "ISO 8601 timestamp",
  "hyperparameters": {
    "param1": "value1",
    "param2": "value2"
  },
  "model_specific_params": {
    "n_layers": "int",
    "n_iter": "int",
    "loss": "float"
  },
  "splits": {
    "train": "float",
    "val": "float",
    "test": "float"
  },
  "random_seed": "int",
  "val_accuracy": "float"
}
```

### Example

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

### Field Descriptions

- `model_type`: Registered model name
- `features`: Ordered list of features (must match during inference)
- `n_features`: Number of features
- `n_classes`: Number of unique classes
- `class_names`: Actual class labels
- `training_date`: When model was trained
- `hyperparameters`: User-configurable parameters
- `model_specific_params`: Model-specific learned parameters
- `splits`: Train/val/test split ratios
- `random_seed`: Random seed for reproducibility
- `val_accuracy`: Validation accuracy achieved

## Evaluation Results JSON

### Schema

JSON file storing evaluation metrics.

```json
{
  "metric_name": "float or dict",
  "confusion_matrix": [[...], [...]]
}
```

### Example

```json
{
  "accuracy": 0.9235,
  "precision_macro": 0.9201,
  "recall_macro": 0.9162,
  "f1_macro": 0.9180,
  "f1_per_class": {
    "1": 0.95,
    "2": 0.91,
    "3": 0.89
  },
  "confusion_matrix": [
    [45, 2, 1],
    [1, 43, 2],
    [0, 3, 45]
  ]
}
```

### Notes

- Scalar metrics stored as floats
- Per-class metrics stored as dictionaries
- Confusion matrix as 2D array

## Training Log Text

### Format

Human-readable text file with training summary.

```
CSI Toolkit Model Training Log
==================================================

Training Date: 2025-01-13T14:30:22
Model Type: mlp

Dataset Info:
  Total samples: 1000
  Train samples: 700
  Val samples: 150
  Test samples: 150
  Features: 7
  Classes: 3

Hyperparameters:
  hidden_layer_sizes: [100, 50]
  max_iter: 500
  learning_rate: adaptive
  solver: adam
  random_state: 42

Training Results:
  Validation Accuracy: 0.9235
  Training completed in 245 iterations
  Final loss: 0.0823

Model saved to: models/model_20250113_143022/
```

## Environment Configuration

### .env File Format

```env
# Serial Port Settings
SERIAL_PORT=/dev/cu.usbmodem1101
BAUDRATE=921600

# CSV Output Settings
FLUSH_INTERVAL=1
```

### Fields

- `SERIAL_PORT`: Serial device path (platform-specific)
- `BAUDRATE`: Communication speed (typically 921600)
- `FLUSH_INTERVAL`: How often to flush CSV writes (seconds)

## CSISample Data Class

### Python Structure

```python
@dataclass
class CSISample:
    seq: int
    mac: str
    rssi: int
    rate: int
    noise_floor: int
    fft_gain: int
    agc_gain: int
    channel: int
    local_timestamp: str
    sig_len: int
    rx_state: int
    len: int
    first_word: int
    data: str
    amplitudes: List[float]
    label: Optional[int] = None
```

### Notes

- Constructed from CSV rows
- `amplitudes` parsed from JSON string to list of floats
- `label` only present in labeled mode

## WindowData Data Class

### Python Structure

```python
@dataclass
class WindowData:
    window_id: int
    start_seq: int
    end_seq: int
    samples: List[CSISample]
```

### Notes

- Represents a window of consecutive samples
- Used internally in feature extraction
- `samples` length = window size (e.g., 100)

## Next Steps

- [Data Collection Guide](../user-guide/data-collection.md)
- [Feature Extraction Guide](../user-guide/feature-extraction.md)
- [Architecture Overview](../developer-guide/architecture.md)
