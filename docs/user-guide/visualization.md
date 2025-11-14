# Visualization Guide

Real-time CSI data visualization with filtering options.

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

## Next Steps

- [Data Collection Guide](data-collection.md)
- [Feature Extraction Guide](feature-extraction.md)
- [Data Format Reference](../reference/data-formats.md)
