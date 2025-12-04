# Visualization Guide

CSI Toolkit provides two visualization modes:
- **Real-time plotting** (`plot`): Live visualization of raw CSI data as it's collected
- **Static plot generation** (`plot-data`): Generate publication-ready plots from processed feature data

## Basic Visualization

Plot a local CSV file:

```bash
python -m csi_toolkit plot data/csi_log_20250113_120000.csv
```

This opens a matplotlib window with two plots:
- Top: Mean amplitude across all subcarriers
- Bottom: Single subcarrier amplitude (default: subcarrier 0)

## Subcarrier Selection

View a specific subcarrier:

```bash
python -m csi_toolkit plot data/current.csv --subcarrier 15
```

Valid subcarrier indices: 0-63

## Display Limits

Limit the number of displayed points:

```bash
# Show last 1000 points
python -m csi_toolkit plot data/current.csv --limit 1000

# Show last 500 points
python -m csi_toolkit plot data/current.csv --limit 500
```

This is useful for:
- Reducing plot clutter
- Improving rendering performance
- Focusing on recent data

## Filtering Options

### Moving Average Filter

Apply moving average smoothing:

```bash
# 5-point moving average
python -m csi_toolkit plot data/current.csv --moving-avg 5

# 10-point moving average
python -m csi_toolkit plot data/current.csv --moving-avg 10
```

### Butterworth Filter

Apply Butterworth lowpass filter:

```bash
# Sampling frequency 100 Hz, cutoff 2 Hz, 4th order
python -m csi_toolkit plot data/current.csv --fs 100 --fc 2.0 --order 4

# More aggressive filtering
python -m csi_toolkit plot data/current.csv --fs 100 --fc 1.0 --order 6
```

Parameters:
- `--fs`: Sampling frequency in Hz
- `--fc`: Cutoff frequency in Hz
- `--order`: Filter order (higher = sharper cutoff)

### Combining Filters

You can apply both filters simultaneously:

```bash
python -m csi_toolkit plot data/current.csv --moving-avg 5 --fs 100 --fc 2.0 --order 4
```

## Remote File Access

Plot files on remote servers via SSH:

```bash
python -m csi_toolkit plot user@hostname:/path/to/file.csv
```

Requirements:
- SSH access configured
- SSH key authentication (no password prompt during plotting)

Example:

```bash
python -m csi_toolkit plot pi@raspberrypi:/home/pi/data/current.csv
```

## Plot Behavior

The plotter continuously updates as new data is added to the CSV file, making it suitable for monitoring live collection:

```bash
# Terminal 1: Collect data
python -m csi_toolkit collect

# Terminal 2: Plot in real-time
python -m csi_toolkit plot data/current.csv --limit 1000
```

The plot refreshes automatically when the CSV file is updated.

## Keyboard Controls

While the plot window is open:

- Close window or `Ctrl+C` in terminal to stop plotting

## Troubleshooting

### Plot Not Updating

If the plot is not updating when new data is collected:

1. Check that the CSV file path is correct
2. Verify the collection is writing to the expected file
3. Close and restart the plot

### Performance Issues

If plotting is slow or laggy:

1. Use `--limit` to reduce displayed points
2. Close other applications
3. Reduce filter complexity (lower order Butterworth filter)

### SSH Connection Errors

If remote plotting fails:

1. Test SSH access manually: `ssh user@host`
2. Verify SSH key authentication is configured
3. Check file path is correct on remote system
4. Ensure CSV file exists before plotting

### Empty or No Data

If plot shows no data:

1. Check CSV file contains data rows (not just header)
2. Verify amplitude values are being calculated
3. Check CSV format matches expected schema

## Advanced Usage

### Monitoring Specific Patterns

When monitoring for specific signal patterns:

```bash
# Tight focus on recent data with aggressive filtering
python -m csi_toolkit plot data/current.csv \
  --limit 200 \
  --fs 100 \
  --fc 1.0 \
  --order 6
```

### Comparing Multiple Subcarriers

To compare different subcarriers, run multiple plot windows:

```bash
# Terminal 1
python -m csi_toolkit plot data/current.csv --subcarrier 0

# Terminal 2
python -m csi_toolkit plot data/current.csv --subcarrier 32
```

---

## Static Plot Generation

The `plot-data` command generates static plots from processed feature CSV files. Unlike the real-time `plot` command, this creates publication-ready PNG images.

### Basic Usage

```bash
# Generate all applicable plots
python -m csi_toolkit plot-data output/features.csv

# Display plots interactively (in addition to saving)
python -m csi_toolkit plot-data output/features.csv --display

# Generate specific plots only
python -m csi_toolkit plot-data output/features.csv -p class_distribution

# List available plot types
python -m csi_toolkit plot-data --list-plots
```

### Available Plots

| Plot | Description | Condition |
|------|-------------|-----------|
| `class_distribution` | Pie chart of class label distribution | Requires `label` column |
| `amplitude_over_windows` | Line plot of mean_amp and std_amp over window_id | Requires `mean_amp` and `std_amp` columns |

### Output

Plots are saved to the same directory as the input CSV file:
- `features_class_distribution.png`
- `features_amplitude_windows.png`

### Plot Features

**Class Distribution Pie Chart:**
- Shows percentage breakdown of each class
- Legend with window counts per class
- Automatically sorted by class label

**Amplitude Over Windows:**
- Two-panel plot: mean amplitude (top) and standard deviation (bottom)
- Background shading indicates class regions (if labeled data)
- Y-axis limited to 1st-95th percentile to handle outliers
- Black line for visibility against colored backgrounds

### Adding Custom Plots

The plot system uses a registry pattern. To add a custom plot, create a function in `src/csi_toolkit/visualization/plots/feature_plots.py`:

```python
from .registry import registry

@registry.register(
    name="my_custom_plot",
    condition=lambda df: "my_column" in df.columns,
    description="Description of my plot",
    output_suffix="_my_plot.png",
)
def plot_my_custom(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    # ... create your plot ...
    return fig
```

The `condition` function determines when the plot is applicable based on the DataFrame columns.

## Next Steps

- [Data Collection Guide](data-collection.md)
- [Feature Extraction Guide](feature-extraction.md)
- [Data Format Reference](../reference/data-formats.md)
