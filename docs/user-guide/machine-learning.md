# Machine Learning Guide

Complete guide for training and deploying ML models for CSI-based activity recognition.

## Installation

ML functionality requires additional dependencies:

```bash
pip install -e ".[ml]"
```

This installs scikit-learn and related packages.

## Workflow Overview

```bash
# 1. Collect labeled data
python -m csi_toolkit collect

# 2. Extract features
python -m csi_toolkit process data/labeled.csv features/train.csv --labeled

# 3. Train model
python -m csi_toolkit train features/train.csv

# 4. Evaluate model
python -m csi_toolkit evaluate --dataset features/train.csv --model-dir models/model_X

# 5. Run inference (offline)
python -m csi_toolkit inference --dataset features/test.csv --model-dir models/model_X

# 6. Run inference (live during collection)
python -m csi_toolkit collect --live-inference --model-dir models/model_X
```

## Training

### Basic Training

Train with default settings (MLP model, 70/15/15 split):

```bash
python -m csi_toolkit train features/labeled_features.csv
```

### Model Selection

List available models:

```bash
python -m csi_toolkit train --list-models
```

Specify model type:

```bash
python -m csi_toolkit train features.csv --model mlp
```

Currently available: `mlp` (Multi-Layer Perceptron)

### Custom Output Directory

```bash
python -m csi_toolkit train features.csv --output-dir models/my-experiment
```

Default: `models/model_YYYYMMDD_HHMMSS/`

### Train/Val/Test Split

```bash
python -m csi_toolkit train features.csv \
  --train-split 0.8 \
  --val-split 0.1 \
  --test-split 0.1
```

Defaults: 70% train, 15% validation, 15% test

Splits are stratified (maintain class distribution).

### Hyperparameter Override

```bash
python -m csi_toolkit train features.csv \
  --params hidden_layer_sizes=(200,100),max_iter=1000,learning_rate=adaptive
```

Format: `key=value` pairs separated by commas.

### Training Process

During training:

1. Load features from CSV
2. Validate `label` column exists
3. Split data (stratified by label)
4. Train model on training set
5. Validate on validation set
6. Report final metrics on test set
7. Save model + metadata to timestamped directory

### Training Output

Each training run creates a directory:

```
models/model_20250113_143022/
├── model.pkl              # Trained model
├── metadata.json          # Model configuration
└── training_log.txt       # Training summary
```

## Inference

Generate predictions on new unlabeled data.

### Basic Inference

```bash
python -m csi_toolkit inference \
  --dataset features/new_data.csv \
  --model-dir models/model_20250113_143022
```

### Custom Output File

```bash
python -m csi_toolkit inference \
  --dataset features/new_data.csv \
  --model-dir models/model_20250113_143022 \
  --output predictions.csv
```

Default: `predictions_YYYYMMDD_HHMMSS.csv` in model directory

### Include Probabilities

```bash
python -m csi_toolkit inference \
  --dataset features/new_data.csv \
  --model-dir models/model_20250113_143022 \
  --probabilities
```

### Output Format

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

## Evaluation

Compute performance metrics on labeled test data.

### Basic Evaluation

```bash
python -m csi_toolkit evaluate \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022
```

### Specific Metrics

List available metrics:

```bash
python -m csi_toolkit evaluate --list-metrics
```

Compute specific metrics:

```bash
python -m csi_toolkit evaluate \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022 \
  --metrics accuracy,f1_macro,f1_per_class
```

### Custom Output Files

```bash
python -m csi_toolkit evaluate \
  --dataset features/test.csv \
  --model-dir models/model_20250113_143022 \
  --output-json my_eval.json \
  --output-txt my_eval.txt
```

### Available Metrics

Global metrics:
- `accuracy`: Overall classification accuracy
- `precision_macro`: Macro-averaged precision
- `precision_micro`: Micro-averaged precision
- `recall_macro`: Macro-averaged recall
- `recall_micro`: Micro-averaged recall
- `f1_macro`: Macro-averaged F1 score
- `f1_micro`: Micro-averaged F1 score

Per-class metrics:
- `precision_per_class`: Precision for each class
- `recall_per_class`: Recall for each class
- `f1_per_class`: F1 score for each class

Reports:
- `classification_report`: Comprehensive sklearn report
- `confusion_matrix`: Confusion matrix with class labels

### Evaluation Output

JSON format (`evaluation_YYYYMMDD_HHMMSS.json`):

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
  },
  "confusion_matrix": [[45, 2, 1], [1, 43, 2], [0, 3, 45]]
}
```

Text format (`evaluation_YYYYMMDD_HHMMSS.txt`):

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
  precision_macro: 0.9201
  ...
```

## Live Inference

Run real-time predictions during data collection.

### Basic Usage

```bash
python -m csi_toolkit collect \
  --live-inference \
  --model-dir models/model_20250113_143022
```

### With Custom Window Size

```bash
python -m csi_toolkit collect \
  --live-inference \
  --model-dir models/model_20250113_143022 \
  --window-size 100
```

Window size must match training window size.

### Behavior

During live inference:

1. **Warmup Period**: Features requiring previous windows show "Waiting for history..." until enough windows are collected
2. **Real-time Display**: Predictions appear every completed window
   ```
   Packets: 1500, Errors: 0 | Prediction: 2 (0.87)
   ```
3. **CSV Output**: Includes both `label` and `predicted_label` columns

### Use Cases

- Model validation: Compare predictions with manual labels
- Live monitoring: Real-time activity recognition
- Debugging: Verify model performance during collection

## Model Directory Structure

Each trained model creates:

```
models/model_20250113_143022/
├── model.pkl                        # Serialized model
├── metadata.json                    # Configuration
├── training_log.txt                 # Training summary
├── predictions_20250113_150000.csv  # Inference outputs (if run)
└── evaluation_20250113_160000.json  # Evaluation results (if run)
```

### metadata.json Structure

```json
{
  "model_type": "mlp",
  "features": ["mean_amp", "std_amp", "max_amp", "min_amp",
               "mean_last3", "std_last3", "mean_last10"],
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

This metadata ensures reproducibility and tracks which features the model expects.

## Best Practices

### Data Collection

1. Collect 10-20 seconds per activity instance
2. Collect multiple instances per class (5-10+)
3. Include baseline/background data (label 0)
4. Maintain consistent environment during collection

### Feature Selection

1. Start with basic features (mean, std, max, min)
2. Add temporal features if needed
3. Use same features for training and inference
4. Avoid features with many NaNs or missing values

### Training

1. Use stratified splits to maintain class balance
2. Start with default hyperparameters
3. Tune hyperparameters if performance is poor
4. Save models with descriptive directory names

### Evaluation

1. Always evaluate on held-out test set
2. Check per-class metrics (not just global accuracy)
3. Inspect confusion matrix for systematic errors
4. Validate with live inference before deployment

## Troubleshooting

### Training Errors

Error: `Label column not found`

Solution: Use `--labeled` flag when extracting features

Error: `Insufficient samples for split`

Solution: Collect more data or adjust split ratios

### Inference Errors

Error: `Feature mismatch`

Solution: Use same features for inference as training (check metadata.json)

Error: `Model file not found`

Solution: Verify model directory exists and contains `model.pkl`

### Evaluation Errors

Error: `No labels in dataset`

Solution: Use labeled test data with `label` column

### Live Inference Errors

Predictions show "Unknown":

1. Check window size matches training
2. Wait for warmup period (context windows)
3. Verify model directory and files exist
4. Check features match training features

## Performance Tips

### Model Selection

MLP (Multi-Layer Perceptron):
- Fast training and inference
- Good for moderate-sized datasets
- Handles non-linear patterns well

### Hyperparameter Tuning

For MLP:
- Increase `hidden_layer_sizes` for more complex patterns
- Increase `max_iter` if training doesn't converge
- Use `learning_rate=adaptive` for automatic learning rate adjustment

### Feature Engineering

- More features ≠ better performance
- Start simple, add complexity if needed
- Remove highly correlated features
- Consider domain knowledge when selecting features

## Next Steps

- [Adding Custom Models](../developer-guide/adding-models.md)
- [Adding Custom Metrics](../developer-guide/adding-metrics.md)
- [CLI Reference](cli-reference.md)
