# Architecture Overview

System design and module structure of CSI Toolkit.

## Design Principles

1. **Modularity**: Independent modules with clear interfaces
2. **Extensibility**: Registry-based systems for features, models, and metrics
3. **Separation of Concerns**: Distinct modules for collection, processing, and ML
4. **Reusability**: Shared utilities and data structures across modules

## Module Structure

```
src/csi_toolkit/
├── core/               # Core utilities and constants
├── collection/         # Data collection from serial devices
├── visualization/      # Real-time plotting
├── processing/         # Signal processing and feature extraction
├── ml/                 # Machine learning pipeline
├── io/                 # Input/output utilities
└── main.py            # CLI entry point
```

## Core Module

Provides fundamental utilities used across the toolkit.

### Components

- `constants.py`: Global constants (CSV headers, defaults)
- `parser.py`: CSI data parsing functions
- `exceptions.py`: Custom exception classes

### Purpose

Ensures consistent data formats and parsing logic throughout the system.

## Collection Module

Handles real-time data acquisition from ESP32 devices via serial port.

### Components

- `serial_collector.py`: SerialCollector class
- `config.py`: Configuration management with .env support

### Data Flow

```
ESP32 → Serial Port → SerialCollector → CSV Writer
                            ↓
                     Live Inference (optional)
                            ↓
                       Predictions
```

### Key Features

- Non-blocking serial communication
- Real-time keyboard labeling
- Configurable flush intervals
- Optional live inference integration

## Visualization Module

Real-time plotting with filtering capabilities.

### Components

- `live_plotter.py`: LivePlotter class with matplotlib backend
- `filters.py`: Signal filtering (moving average, Butterworth)

### Data Flow

```
CSV File (local or SSH) → CSV Reader → Filters → Matplotlib Plot
```

## Processing Module

Feature extraction pipeline with windowing and aggregation.

### Components

```
processing/
├── amplitude.py           # Amplitude calculations
├── windowing.py          # Window creation, CSISample/WindowData classes
├── feature_extractor.py  # Main extraction pipeline
└── features/             # Modular feature system
    ├── registry.py       # Feature registration
    ├── basic.py          # Basic features (mean, std, max, min)
    ├── temporal.py       # Multi-window features
    └── utils.py          # Helper functions
```

### Data Flow

```
CSV → CSISamples → Windows → Feature Extraction → Features CSV
```

### Windowing Strategy

- Non-overlapping windows
- Configurable window size
- Context windows for temporal features
- Transition filtering in labeled mode

### Registry System

Features register themselves using decorators:

```python
@registry.register('feature_name', n_prev=2, description='...')
def feature_func(current, prev, next):
    # Implementation
    pass
```

This allows dynamic feature discovery and selection.

## Machine Learning Module

Complete ML pipeline from training to deployment.

### Components

```
ml/
├── models/               # Model implementations
│   ├── base.py          # BaseModel interface
│   ├── registry.py      # Model registration
│   └── sklearn_models.py # scikit-learn models (MLP)
├── metrics/             # Evaluation metrics
│   ├── registry.py      # Metric registration
│   └── classification_metrics.py # Standard metrics
├── training/            # Training pipeline
│   └── trainer.py       # ModelTrainer class
├── inference/           # Prediction pipeline
│   ├── predictor.py     # Batch inference
│   ├── evaluator.py     # Model evaluation
│   └── live_predictor.py # Live inference handler
└── utils.py             # ML utilities
```

### Data Flow

#### Training

```
Features CSV → Train/Val/Test Split → Model Training → Saved Model
```

#### Inference

```
Features CSV → Model Loading → Predictions → Results CSV
```

#### Live Inference

```
Serial Data → Window Buffer → Feature Extraction → Model → Predictions
                                                              ↓
                                                        CSV + Display
```

### Registry System

Models and metrics use similar registry patterns:

```python
# Models
@registry.register('model_name', default_params={...})
class CustomModel(BaseModel):
    # Implementation
    pass

# Metrics
@registry.register('metric_name', requires_proba=False)
def custom_metric(y_true, y_pred):
    # Implementation
    pass
```

## I/O Module

Handles file reading and writing with remote support.

### Components

- `csv_writer.py`: CSVWriter with flush control
- `csv_reader.py`: Local CSV reading with tail support
- `ssh_reader.py`: Remote file access via SSH

### Features

- Automatic flushing for data safety
- Live file tailing for real-time monitoring
- SSH integration for remote data access

## Data Structures

### CSISample

Represents a single CSI measurement:

```python
@dataclass
class CSISample:
    seq: int                    # Sequence number
    mac: str                    # MAC address
    rssi: int                   # Signal strength
    amplitudes: List[float]     # 64 subcarrier amplitudes
    label: Optional[int]        # Class label (labeled mode)
    # ... other fields
```

### WindowData

Represents a window of samples:

```python
@dataclass
class WindowData:
    window_id: int              # Window index
    start_seq: int              # First sample sequence
    end_seq: int                # Last sample sequence
    samples: List[CSISample]    # All samples in window
```

## CLI Architecture

The main CLI provides unified access to all functionality:

```
main.py → Argument Parser → Command Handlers
                                ↓
                    ┌───────────┼───────────┐
                    ↓           ↓           ↓
                collect     process      train/eval/infer
```

Each command has its own handler function with validation and error handling.

## Extension Points

The toolkit is designed for easy extension at multiple levels:

### 1. Custom Features

Add to `processing/features/`:

```python
@registry.register('my_feature')
def my_feature(current, prev, next):
    return compute_value()
```

### 2. Custom Models

Add to `ml/models/`:

```python
@registry.register('my_model')
class MyModel(BaseModel):
    def fit(self, X, y): ...
    def predict(self, X): ...
```

### 3. Custom Metrics

Add to `ml/metrics/`:

```python
@registry.register('my_metric')
def my_metric(y_true, y_pred):
    return compute_score()
```

### 4. Custom Filters

Add to `visualization/filters.py`:

```python
def my_filter(data, **params):
    return filtered_data
```

## Error Handling

Consistent error handling across modules:

1. **Validation**: Input validation at entry points
2. **Exceptions**: Custom exceptions in `core/exceptions.py`
3. **Graceful Degradation**: Fallback behavior when possible
4. **User Feedback**: Clear error messages with actionable advice

## Performance Considerations

### Collection

- Buffered serial reading
- Configurable flush intervals
- Asynchronous keyboard handling

### Processing

- NumPy vectorization for feature calculations
- Batch processing of windows
- Memory-efficient CSV streaming

### ML

- scikit-learn optimizations (Accelerate on macOS)
- Batch prediction for efficiency
- Lazy model loading

## Testing Strategy

Each module has testable components:

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test module interactions
3. **End-to-End Tests**: Test complete workflows

## Future Extensibility

The architecture supports:

- Additional data sources (network, file streaming)
- More ML frameworks (TensorFlow, PyTorch)
- Custom visualization backends
- Distributed processing
- Model versioning and tracking

## Next Steps

- [Adding Custom Features](adding-features.md)
- [Adding Custom Models](adding-models.md)
- [Adding Custom Metrics](adding-metrics.md)
- [Data Format Reference](../reference/data-formats.md)
