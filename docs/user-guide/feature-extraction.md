# Feature Extraction Guide

Convert raw CSI data into windowed features for machine learning.

## Basic Usage

Extract all features with default settings:

```bash
python -m csi_toolkit process input.csv output.csv
```

This creates non-overlapping windows of 100 samples and extracts all registered features.

## Window Size

Customize the window size:

```bash
# 50 samples per window
python -m csi_toolkit process input.csv output.csv --window-size 50

# 200 samples per window
python -m csi_toolkit process input.csv output.csv --window-size 200
```

Considerations:
- Smaller windows: Higher temporal resolution, less data per window
- Larger windows: More stable statistics, lower temporal resolution
- Must match window size used during model training

## Feature Selection

### List Available Features

```bash
python -m csi_toolkit process --list-features
```

### Extract Specific Features

```bash
python -m csi_toolkit process input.csv output.csv --features mean_amp,std_amp,max_amp
```

Separate feature names with commas (no spaces).

## Labeled Mode

Process labeled data collected with keyboard input:

```bash
python -m csi_toolkit process data/labeled_data.csv features/output.csv --labeled
```

Labeled mode behavior:

1. Reads `label` column from input CSV
2. Adds `label` column to output features
3. Discards transition windows (windows containing label changes)
4. Applies buffer around transitions

### Transition Window Handling

Windows where ANY sample has a different label are discarded:

```
Window:    0  1  2  3  4  5  6  7  8  9
Label:     1  1  1  2  2  2  2  3  3  3
                    ^              ^
                transition      transition

With --transition-buffer 1 (default):
Discarded: windows 2, 3, 4 and windows 6, 7, 8
Output:    windows 0, 1, 5, 9 (with labels 1, 1, 2, 3)
```

This ensures clean, homogeneous windows for training.

### Transition Buffer

Adjust the buffer size:

```bash
# No buffer (only discard transition windows themselves)
python -m csi_toolkit process input.csv output.csv --labeled --transition-buffer 0

# Larger buffer (discard 2 windows before/after transitions)
python -m csi_toolkit process input.csv output.csv --labeled --transition-buffer 2
```

Default: 1 window before and after each transition.

## Available Features

### Basic Features

Calculated from a single window:

- `mean_amp`: Mean amplitude across all subcarriers
- `std_amp`: Standard deviation of amplitude
- `max_amp`: Maximum amplitude
- `min_amp`: Minimum amplitude

### Temporal Features

Calculated from multiple windows (require context):

- `mean_last3`: Mean amplitude over last 3 windows (needs 2 previous windows)
- `std_last3`: Std deviation over last 3 windows (needs 2 previous windows)
- `mean_last10`: Mean amplitude over last 10 windows (needs 9 previous windows)

## Windowing Behavior

### Non-Overlapping Windows

Data is split into non-overlapping windows:

```
Samples:  [0-99] [100-199] [200-299] [300-399] ...
Windows:    W0      W1        W2        W3     ...
```

No overlap ensures statistical independence between windows.

### Aggregation Across Subcarriers

For each sample:
1. Calculate mean amplitude across all 64 subcarriers
2. This provides robustness to single-subcarrier fluctuations

For each window:
1. Apply aggregation function (mean, std, max, min) to per-sample means
2. Result is a single feature value per window

### Edge Window Handling

Features requiring N previous windows cannot be calculated for the first N windows:

Example with `mean_last10` (needs 9 previous windows):

```
Input:  1500 samples → 15 windows (window size = 100)
Output: Only windows 9-14 (6 windows)
        Windows 0-8 are skipped due to insufficient context
```

Similarly, features requiring next windows skip the last N windows.

## Output Format

### Standard Mode

```csv
window_id,start_seq,end_seq,mean_amp,std_amp,max_amp,min_amp,mean_last3,std_last3,mean_last10
9,900,999,45.2,3.1,52.3,38.1,45.0,1.9,44.8
10,1000,1099,46.1,2.9,51.8,39.0,45.5,0.7,45.1
```

Columns:
- `window_id`: Window index
- `start_seq`: First sample sequence number
- `end_seq`: Last sample sequence number
- Feature columns: One per registered feature

### Labeled Mode

```csv
window_id,start_seq,end_seq,label,mean_amp,std_amp,max_amp,min_amp,mean_last3,std_last3,mean_last10
9,900,999,1,45.2,3.1,52.3,38.1,45.0,1.9,44.8
10,1000,1099,1,46.1,2.9,51.8,39.0,45.5,0.7,45.1
```

Additional column:
- `label`: Class label (0-9) from labeled collection

## Workflow Integration

### For Training

```bash
# 1. Collect labeled data
python -m csi_toolkit collect

# 2. Extract features in labeled mode
python -m csi_toolkit process data/labeled.csv features/training.csv --labeled

# 3. Train model
python -m csi_toolkit train features/training.csv
```

### For Inference

```bash
# 1. Collect unlabeled data
python -m csi_toolkit collect

# 2. Extract features (no --labeled flag)
python -m csi_toolkit process data/new_data.csv features/test.csv

# 3. Run inference
python -m csi_toolkit inference --dataset features/test.csv --model-dir models/model_X
```

## Performance

Processing speed depends on:
- Number of features (more features = slower)
- Window size (larger windows = fewer windows to process)
- Input file size

Typical performance: ~10,000-50,000 samples/second

## Troubleshooting

### Insufficient Windows Error

Error: `Insufficient windows: need at least X for requested features, but only have Y`

Solution:
- Collect more data (longer collection time)
- Reduce window size: `--window-size 50`
- Remove features requiring many context windows

### Too Many Discarded Windows

Warning: Large number of transition windows discarded in labeled mode

Solutions:
- Reduce transition buffer: `--transition-buffer 0`
- Collect longer activity periods (10-20 seconds each)
- Ensure cleaner label transitions during collection

### Missing Features

Error: Feature `xyz` not found

Solutions:
- List available features: `python -m csi_toolkit process --list-features`
- Check feature name spelling
- Ensure custom features are registered (see [Adding Features](../developer-guide/adding-features.md))

### Empty Output

No features in output file:

1. Check input file has sufficient samples
2. Verify features are registered
3. Check edge windows are not consuming all data

## Best Practices

1. **Consistent Window Size**: Use the same window size for training and inference
2. **Feature Selection**: Start with basic features, add temporal features if needed
3. **Labeled Collection**: Collect 10-20 seconds per activity, avoid quick transitions
4. **Transition Buffer**: Default (1) works well for most cases
5. **Validation**: Check output CSV has expected number of windows and features

## Next Steps

- [Machine Learning Guide](machine-learning.md)
- [Adding Custom Features](../developer-guide/adding-features.md)
- [Data Format Reference](../reference/data-formats.md)
