# Data Collection Guide

Comprehensive guide for collecting CSI data from ESP32 devices.

## Basic Collection

### Starting Collection

```bash
python -m csi_toolkit collect
```

This starts serial data collection with default settings from `.env` file.

### Configuration

Create a `.env` file in the project root:

```env
SERIAL_PORT=/dev/cu.usbmodem1101
BAUDRATE=921600
FLUSH_INTERVAL=1
```

Parameters:
- `SERIAL_PORT`: Serial device path
- `BAUDRATE`: Communication speed (typically 921600)
- `FLUSH_INTERVAL`: How often to flush CSV writes (seconds)

### Command-Line Options

Override `.env` settings via command-line flags:

```bash
# Custom serial port
python -m csi_toolkit collect --port /dev/ttyUSB0

# Custom baudrate
python -m csi_toolkit collect --baudrate 115200

# Custom output directory
python -m csi_toolkit collect --output-dir mydata

# Custom flush interval (in seconds)
python -m csi_toolkit collect --flush 5

# Enable debug output
python -m csi_toolkit collect --debug
```

## Labeled Data Collection

For machine learning applications, collect data with real-time keyboard labeling.

### Keyboard Controls

During collection, press number keys to label activities:

- **Keys 0-9**: Set current class label
  - `0`: Unlabeled/background
  - `1-9`: Activity classes
- **Ctrl+C**: Stop collection gracefully

### Labeling Workflow

Example workflow for collecting two activities:

```bash
python -m csi_toolkit collect
```

1. Collection starts, label defaults to `0` (unlabeled)
2. Wait a few seconds for baseline data
3. Press `1` before starting Activity A
4. Perform Activity A for 10-20 seconds
5. Press `0` to return to background
6. Wait a few seconds
7. Press `2` before starting Activity B
8. Perform Activity B for 10-20 seconds
9. Press `0` to return to background
10. Press `Ctrl+C` to stop

The CSV file includes a `label` column recording the label at each sample.

### Best Practices

1. **Baseline Data**: Always collect unlabeled background data (label 0)
2. **Activity Duration**: Collect 10-20 seconds per activity instance
3. **Multiple Instances**: Repeat each activity multiple times (5-10 instances)
4. **Clean Transitions**: Press label key before starting activity, not during
5. **Consistent Timing**: Wait a few seconds between activities

## Live Inference During Collection

Collect data with real-time model predictions.

### Prerequisites

1. Trained model available
2. Model window size matches collection window size

### Usage

```bash
python -m csi_toolkit collect --live-inference --model-dir models/model_20250113_143022
```

### Options

```bash
# Custom window size (must match training)
python -m csi_toolkit collect \
  --live-inference \
  --model-dir models/model_20250113_143022 \
  --window-size 100
```

### Behavior

During live inference collection:

1. **Warmup Period**: If features require previous windows (e.g., `mean_last10`), initial windows show "Waiting for history..."
   - Example: Features requiring 9 previous windows start predicting at window 9
2. **Real-time Display**: Predictions appear in the CLI every completed window
   - Format: `Packets: 1500, Errors: 0 | Prediction: 2 (0.87)`
3. **CSV Output**: Both `label` and `predicted_label` columns are saved
   - `label`: User-provided label (keyboard input)
   - `predicted_label`: Model's prediction

### Use Cases

- **Model Validation**: Compare predictions with manual labels in real-time
- **Live Monitoring**: Deploy trained model for real-time activity recognition
- **Debugging**: Verify model performance during data collection

## Output Files

Collection creates CSV files in the output directory:

```
data/
└── csi_log_20250113_143022.csv
```

Filename format: `csi_log_YYYYMMDD_HHMMSS.csv`

See [Data Formats Reference](../reference/data-formats.md) for CSV schema details.

## Troubleshooting

### Serial Port Not Found

Error: `Failed to open serial port`

Solutions:
1. Check device is connected: `ls /dev/cu.*` (macOS) or `ls /dev/ttyUSB*` (Linux)
2. Verify port in `.env` matches actual device
3. Check permissions (Linux): Add user to `dialout` group
4. Close other applications using the port

### No Data Appearing

Collection starts but no packets:

1. Check ESP32 is running CSI firmware
2. Verify baudrate matches ESP32 configuration (usually 921600)
3. Enable debug mode: `python -m csi_toolkit collect --debug`

### High Error Count

Many parsing errors during collection:

1. Reduce baudrate if experiencing data corruption
2. Check USB cable quality
3. Verify ESP32 firmware is sending correct format
4. Use `--debug` to inspect malformed packets

### Keyboard Labeling Not Working

macOS accessibility warning:

The first run may show: "This process is not trusted! Input event monitoring will not be possible..."

This is normal. Keyboard labeling still works through terminal input.

### Live Inference Errors

Predictions show "Unknown":

1. Verify model directory exists and contains `model.pkl` and `metadata.json`
2. Check window size matches training: `--window-size 100`
3. Wait for warmup period if features need previous windows
4. Check model was trained on same feature set

## Performance Tips

### Flush Interval

- Lower flush interval (1-2 seconds): More frequent disk writes, safer but slightly slower
- Higher flush interval (5-10 seconds): Faster collection, risk of data loss on crash

### Buffer Management

Collection automatically manages buffers for:
- Serial input buffer
- CSV write buffer
- Live inference window buffer

No manual tuning required.

## Next Steps

- [Visualization Guide](visualization.md)
- [Feature Extraction Guide](feature-extraction.md)
- [Machine Learning Pipeline](machine-learning.md)
