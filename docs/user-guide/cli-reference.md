# CLI Reference

Complete reference for all CSI Toolkit commands and options.

## Global Commands

### info

Display package information:

```bash
python -m csi_toolkit info
```

Shows version, modules, and installation details.

### help

```bash
python -m csi_toolkit --help
python -m csi_toolkit <command> --help
```

## collect

Collect CSI data from serial port.

### Synopsis

```bash
python -m csi_toolkit collect [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | string | From .env | Serial port path |
| `--baudrate` | int | From .env | Serial baudrate |
| `--output-dir` | string | `data` | Output directory |
| `--flush` | int | From .env | CSV flush interval (seconds) |
| `--debug` | flag | False | Enable debug output |
| `--live-inference` | flag | False | Enable live predictions |
| `--model-dir` | string | None | Model directory for live inference |
| `--window-size` | int | 100 | Window size for live inference |

### Examples

```bash
# Basic collection
python -m csi_toolkit collect

# Custom port and baudrate
python -m csi_toolkit collect --port /dev/ttyUSB0 --baudrate 115200

# With live inference
python -m csi_toolkit collect --live-inference --model-dir models/model_20250113_143022

# Debug mode
python -m csi_toolkit collect --debug
```

### Configuration File

`.env` file in project root:

```env
SERIAL_PORT=/dev/cu.usbmodem1101
BAUDRATE=921600
FLUSH_INTERVAL=1
```

## plot

Real-time visualization of CSI data.

### Synopsis

```bash
python -m csi_toolkit plot FILE [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `FILE` | CSV file path (local or SSH: `user@host:/path`) |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--subcarrier` | int | 0 | Subcarrier index to plot (0-63) |
| `--limit` | int | None | Limit displayed points |
| `--moving-avg` | int | None | Moving average window size |
| `--fs` | float | None | Sampling frequency (Hz) for Butterworth |
| `--fc` | float | None | Cutoff frequency (Hz) for Butterworth |
| `--order` | int | None | Butterworth filter order |

### Examples

```bash
# Basic plot
python -m csi_toolkit plot data/csi_log_20250113_120000.csv

# Specific subcarrier
python -m csi_toolkit plot data/current.csv --subcarrier 15

# With filtering
python -m csi_toolkit plot data/current.csv --moving-avg 5 --limit 1000

# Butterworth filter
python -m csi_toolkit plot data/current.csv --fs 100 --fc 2.0 --order 4

# Remote file
python -m csi_toolkit plot user@host:/path/to/file.csv
```

## plot-data

Generate static plots from processed feature data.

### Synopsis

```bash
python -m csi_toolkit plot-data FILE [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `FILE` | Feature CSV file path (from `process` command) |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--display` | flag | False | Display plots interactively after saving |
| `-p, --plots` | string | All applicable | Comma-separated list of plots to generate |
| `--list-plots` | flag | False | List available plot types and exit |

### Examples

```bash
# Generate all applicable plots
python -m csi_toolkit plot-data output/features.csv

# Display plots interactively
python -m csi_toolkit plot-data output/features.csv --display

# Generate specific plots
python -m csi_toolkit plot-data output/features.csv -p class_distribution,amplitude_over_windows

# List available plots
python -m csi_toolkit plot-data --list-plots
```

### Available Plots

| Plot | Description | Required Columns |
|------|-------------|------------------|
| `class_distribution` | Pie chart of class labels | `label` |
| `amplitude_over_windows` | Mean/std amplitude over windows | `mean_amp`, `std_amp` |

### Output

Plots are saved as PNG files in the same directory as the input CSV:
- `{filename}_class_distribution.png`
- `{filename}_amplitude_windows.png`

## process

Extract features from raw CSI data.

### Synopsis

```bash
python -m csi_toolkit process INPUT OUTPUT [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT` | Input CSV file path |
| `OUTPUT` | Output features CSV path |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--window-size` | int | 100 | Samples per window |
| `--features` | string | All | Comma-separated feature names |
| `--labeled` | flag | False | Process labeled data |
| `--transition-buffer` | int | 1 | Windows to discard around transitions |
| `--split` | int | None | Train/test split percentage (e.g., 70 for 70% train, 30% test). Stratified by label. Outputs `<output>-train.csv` and `<output>-test.csv`. Requires `--labeled`. |
| `--list-features` | flag | False | List available features and exit |

### Examples

```bash
# Basic processing
python -m csi_toolkit process input.csv output.csv

# Custom window size
python -m csi_toolkit process input.csv output.csv --window-size 50

# Labeled data
python -m csi_toolkit process labeled.csv features.csv --labeled

# Specific features
python -m csi_toolkit process input.csv output.csv --features mean_amp,std_amp,max_amp

# List features
python -m csi_toolkit process --list-features

# Labeled with custom buffer
python -m csi_toolkit process labeled.csv features.csv --labeled --transition-buffer 2

# Stratified train/test split (70% train, 30% test)
# Creates features-train.csv and features-test.csv
python -m csi_toolkit process labeled.csv features.csv --labeled --split 70
```

## train

Train machine learning model.

### Synopsis

```bash
python -m csi_toolkit train DATASET [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `DATASET` | Features CSV file path (with labels) |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | string | `mlp` | Model type |
| `--output-dir` | string | Auto | Output directory for model |
| `--train-split` | float | 0.7 | Training set fraction |
| `--val-split` | float | 0.15 | Validation set fraction |
| `--test-split` | float | 0.15 | Test set fraction |
| `--params` | string | None | Hyperparameters (key=value,key=value) |
| `--list-models` | flag | False | List available models and exit |

### Examples

```bash
# Basic training
python -m csi_toolkit train features/labeled_features.csv

# Specific model
python -m csi_toolkit train features.csv --model mlp

# Custom splits
python -m csi_toolkit train features.csv --train-split 0.8 --val-split 0.1 --test-split 0.1

# Custom hyperparameters
python -m csi_toolkit train features.csv --params hidden_layer_sizes=(200,100),max_iter=1000

# List models
python -m csi_toolkit train --list-models

# Custom output directory
python -m csi_toolkit train features.csv --output-dir models/experiment1
```

## inference

Run predictions on new data.

### Synopsis

```bash
python -m csi_toolkit inference --dataset DATASET --model-dir MODEL_DIR [OPTIONS]
```

### Required Options

| Option | Type | Description |
|--------|------|-------------|
| `--dataset` | string | Features CSV file path |
| `--model-dir` | string | Model directory path |

### Optional Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | string | Auto | Output predictions CSV path |
| `--probabilities` | flag | False | Include class probabilities |

### Examples

```bash
# Basic inference
python -m csi_toolkit inference \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022

# Custom output
python -m csi_toolkit inference \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022 \
  --output predictions.csv

# With probabilities
python -m csi_toolkit inference \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022 \
  --probabilities
```

## evaluate

Evaluate model performance.

### Synopsis

```bash
python -m csi_toolkit evaluate --dataset DATASET --model-dir MODEL_DIR [OPTIONS]
```

### Required Options

| Option | Type | Description |
|--------|------|-------------|
| `--dataset` | string | Labeled features CSV path |
| `--model-dir` | string | Model directory path |

### Optional Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--metrics` | string | All | Comma-separated metric names |
| `--output-json` | string | Auto | JSON output path |
| `--output-txt` | string | Auto | Text report output path |
| `--list-metrics` | flag | False | List available metrics and exit |

### Examples

```bash
# Basic evaluation
python -m csi_toolkit evaluate \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022

# Specific metrics
python -m csi_toolkit evaluate \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022 \
  --metrics accuracy,f1_macro,confusion_matrix

# Custom output files
python -m csi_toolkit evaluate \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022 \
  --output-json my_eval.json \
  --output-txt my_eval.txt

# List metrics
python -m csi_toolkit evaluate --list-metrics
```

## Exit Codes

All commands return:
- `0` on success
- `1` on error

## Environment Variables

The following environment variables can be used:

| Variable | Description |
|----------|-------------|
| `SERIAL_PORT` | Default serial port |
| `BAUDRATE` | Default baudrate |
| `FLUSH_INTERVAL` | Default CSV flush interval |
| `PYTHONPATH` | Include `src` for development |

## Common Patterns

### Complete Workflow

```bash
# 1. Collect
python -m csi_toolkit collect

# 2. Process
python -m csi_toolkit process data/current.csv features/train.csv --labeled

# 3. Train
python -m csi_toolkit train features/train.csv

# 4. Evaluate
python -m csi_toolkit evaluate --dataset features/train.csv --model-dir models/model_X

# 5. Inference
python -m csi_toolkit inference --dataset features/test.csv --model-dir models/model_X
```

### Live Monitoring

```bash
# Terminal 1: Collect with live inference
python -m csi_toolkit collect --live-inference --model-dir models/model_X

# Terminal 2: Visualize simultaneously
python -m csi_toolkit plot data/current.csv --limit 1000
```

### Batch Processing

```bash
# Process multiple files
for file in data/*.csv; do
  python -m csi_toolkit process "$file" "features/$(basename $file)" --labeled
done
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Data Collection Guide](data-collection.md)
- [Machine Learning Guide](machine-learning.md)
