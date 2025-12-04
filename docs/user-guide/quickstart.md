# Quick Start Guide

This guide will walk you through a complete workflow from data collection to model evaluation.

## Quick Start with Docker (No Installation Required)

If you just want to explore CSI data processing using the included sample dataset (`data/csi-60k.csv`), Docker is the fastest way to get started:

```bash
# Clone and build
git clone https://github.com/FreiPaul/CSI-Collector
cd "CSI-Collector"
docker build -t csi-toolkit:latest .

# Process the sample dataset (60k CSI samples)
docker compose run --rm csi-toolkit process data/csi-60k.csv output/features.csv --labeled

# Train a model on the extracted features
docker compose run --rm csi-toolkit train output/features.csv

# Evaluate the trained model
docker compose run --rm csi-toolkit evaluate --dataset output/features.csv --model-dir models/<your_model>

# Interactive shell for exploration
docker compose run --rm csi-dev
```

This gives you a fully working ML environment with TensorFlow, PyTorch, and scikit-learn without any local Python setup.

---

## Prerequisites (Native Installation)

1. CSI Toolkit installed ([Installation Guide](installation.md))
2. ESP32 device connected via USB (for data collection)
3. ML dependencies installed for training: `pip install -e ".[ml]"`

## Complete Workflow

### 1. Collect Labeled Data

Start collecting CSI data with keyboard labeling:

```bash
python -m csi_toolkit collect
```

During collection:
- Press `1` when performing Activity A
- Press `2` when performing Activity B
- Press `0` for background/unlabeled periods
- Press `Ctrl+C` to stop

The data is saved to `data/csi_log_YYYYMMDD_HHMMSS.csv`.

### 2. Visualize Raw Data (Optional)

Check that your data looks good:

```bash
python -m csi_toolkit plot data/csi_log_20250113_120000.csv
```

This opens a real-time plot showing CSI amplitudes.

### 3. Extract Features

Convert raw CSI samples into windowed features:

```bash
python -m csi_toolkit process data/csi_log_20250113_120000.csv features/labeled_features.csv --labeled
```

This creates non-overlapping windows of 100 samples each and extracts features like mean, std, max, min amplitudes.

### 4. Train a Model

Train a classifier on the extracted features:

```bash
python -m csi_toolkit train features/labeled_features.csv
```

The model is saved to `models/model_YYYYMMDD_HHMMSS/` with metadata and training logs.

### 5. Evaluate the Model

Assess model performance on test data:

```bash
python -m csi_toolkit evaluate --dataset features/labeled_features.csv --model-dir models/model_20250113_143022
```

This outputs evaluation metrics (accuracy, F1 score, etc.) in JSON and text format.

### 6. Run Live Inference

Collect new data with real-time predictions:

```bash
python -m csi_toolkit collect --live-inference --model-dir models/model_20250113_143022
```

You will see predicted labels appear in real-time as windows are completed.

## Understanding the Data Flow

```
Raw CSI Data (serial) → CSV File → Windowed Features → Trained Model → Predictions
```

1. **Collection**: ESP32 sends CSI packets over serial → saved to CSV
2. **Processing**: CSV samples → windows of 100 samples → feature extraction
3. **Training**: Features + labels → ML model
4. **Inference**: New features → model → predicted labels

## Common Commands Reference

| Task | Command |
|------|---------|
| Collect data | `python -m csi_toolkit collect` |
| Visualize data | `python -m csi_toolkit plot <file.csv>` |
| Extract features | `python -m csi_toolkit process <input.csv> <output.csv> --labeled` |
| Train model | `python -m csi_toolkit train <features.csv>` |
| Evaluate model | `python -m csi_toolkit evaluate --dataset <test.csv> --model-dir <dir>` |
| Run inference | `python -m csi_toolkit inference --dataset <features.csv> --model-dir <dir>` |
| Live inference | `python -m csi_toolkit collect --live-inference --model-dir <dir>` |

## Next Steps

- [Data Collection Details](data-collection.md)
- [Feature Extraction Details](feature-extraction.md)
- [Machine Learning Pipeline](machine-learning.md)
- [Complete CLI Reference](cli-reference.md)

## Troubleshooting

### No data appearing during collection

- Check serial port connection
- Verify `.env` file has correct `SERIAL_PORT`
- Try listing ports: `ls /dev/cu.*` (macOS) or `ls /dev/ttyUSB*` (Linux)

### Import errors

- Ensure virtual environment is activated
- For ML functionality: `pip install -e ".[ml]"`

### Label transitions in features

When processing labeled data, windows containing label transitions are automatically discarded to ensure clean training data. This is expected behavior.
