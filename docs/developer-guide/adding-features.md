# Adding Custom Features

Guide for implementing custom features in the CSI Toolkit.

## Overview

The CSI Toolkit uses a registry-based feature system that allows easy addition of custom features without modifying core code.

## Feature Structure

Features are functions that take windowed CSI samples and return computed values.

### Feature Function Signature

```python
def feature_function(
    current_samples: List[CSISample],
    prev_samples: List[List[CSISample]],
    next_samples: List[List[CSISample]]
) -> float:
    """
    Compute feature value.

    Args:
        current_samples: Samples in current window
        prev_samples: List of previous windows (each is List[CSISample])
        next_samples: List of next windows (each is List[CSISample])

    Returns:
        Computed feature value (float)
    """
    pass
```

## Basic Feature Example

Create a new file `src/csi_toolkit/processing/features/custom.py`:

```python
"""Custom features."""

import numpy as np
from .registry import registry
from .utils import get_sample_mean_amplitudes


@registry.register('range_amp', description='Amplitude range (max - min)')
def range_amp(current_samples, prev_samples, next_samples):
    """Calculate amplitude range in current window."""
    # Get per-sample mean amplitudes
    sample_means = get_sample_mean_amplitudes(current_samples)

    # Compute range
    return float(np.max(sample_means) - np.min(sample_means))


@registry.register('median_amp', description='Median amplitude')
def median_amp(current_samples, prev_samples, next_samples):
    """Calculate median amplitude in current window."""
    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(np.median(sample_means))
```

## Registering Features

Add your module to `src/csi_toolkit/processing/features/__init__.py`:

```python
"""Feature extraction functions."""

from . import basic
from . import temporal
from . import custom  # Add this line

__all__ = ['registry']
```

## Context-Aware Features

Features can require previous or next windows for temporal context.

### Using Previous Windows

```python
@registry.register(
    'mean_last5',
    n_prev=4,  # Requires 4 previous windows
    description='Mean amplitude over last 5 windows'
)
def mean_last5(current_samples, prev_samples, next_samples):
    """Calculate mean amplitude across 5 windows (current + 4 previous)."""
    # Combine all samples
    all_samples = []

    # Add previous windows
    for prev_window in prev_samples:
        all_samples.extend(prev_window)

    # Add current window
    all_samples.extend(current_samples)

    # Calculate mean
    sample_means = get_sample_mean_amplitudes(all_samples)
    return float(np.mean(sample_means))
```

### Using Next Windows

```python
@registry.register(
    'mean_next3',
    n_next=3,  # Requires 3 next windows
    description='Mean amplitude over next 3 windows'
)
def mean_next3(current_samples, prev_samples, next_samples):
    """Calculate mean amplitude across next 3 windows."""
    all_samples = []

    for next_window in next_samples:
        all_samples.extend(next_window)

    sample_means = get_sample_mean_amplitudes(all_samples)
    return float(np.mean(sample_means))
```

### Using Both

```python
@registry.register(
    'mean_centered',
    n_prev=2,
    n_next=2,
    description='Mean amplitude over 5-window centered window'
)
def mean_centered(current_samples, prev_samples, next_samples):
    """Calculate mean amplitude across centered window."""
    all_samples = []

    # Previous windows
    for prev_window in prev_samples:
        all_samples.extend(prev_window)

    # Current window
    all_samples.extend(current_samples)

    # Next windows
    for next_window in next_samples:
        all_samples.extend(next_window)

    sample_means = get_sample_mean_amplitudes(all_samples)
    return float(np.mean(sample_means))
```

## Utility Functions

The toolkit provides helper functions for common operations.

### get_sample_mean_amplitudes

Calculate mean amplitude across subcarriers for each sample:

```python
from .utils import get_sample_mean_amplitudes

sample_means = get_sample_mean_amplitudes(current_samples)
# Returns: np.ndarray of shape (n_samples,)
```

### Direct Amplitude Access

Access raw amplitude data:

```python
# Each sample has amplitudes attribute (list of 64 values)
for sample in current_samples:
    amplitudes = sample.amplitudes  # List[float]
    # Process individual subcarrier amplitudes
```

## Registration Options

The `@registry.register()` decorator accepts several parameters:

```python
@registry.register(
    name='feature_name',          # Required: Feature name
    description='Description',    # Optional: Human-readable description
    n_prev_windows=0,             # Optional: Number of previous windows needed
    n_next_windows=0,             # Optional: Number of next windows needed
)
```

Alternative shorter syntax:

```python
@registry.register('feature_name')  # Uses function name and defaults
```

## Advanced Examples

### Statistical Features

```python
@registry.register('skewness_amp', description='Amplitude skewness')
def skewness_amp(current_samples, prev_samples, next_samples):
    """Calculate skewness of amplitude distribution."""
    from scipy.stats import skew

    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(skew(sample_means))


@registry.register('kurtosis_amp', description='Amplitude kurtosis')
def kurtosis_amp(current_samples, prev_samples, next_samples):
    """Calculate kurtosis of amplitude distribution."""
    from scipy.stats import kurtosis

    sample_means = get_sample_mean_amplitudes(current_samples)
    return float(kurtosis(sample_means))
```

### Derivative Features

```python
@registry.register(
    'amp_derivative',
    n_prev=1,
    description='Amplitude change from previous window'
)
def amp_derivative(current_samples, prev_samples, next_samples):
    """Calculate mean amplitude change from previous window."""
    current_mean = np.mean(get_sample_mean_amplitudes(current_samples))

    # Get previous window mean
    prev_window = prev_samples[-1]  # Last previous window
    prev_mean = np.mean(get_sample_mean_amplitudes(prev_window))

    return float(current_mean - prev_mean)
```

### Subcarrier-Specific Features

```python
@registry.register('sc0_mean', description='Mean amplitude of subcarrier 0')
def subcarrier_0_mean(current_samples, prev_samples, next_samples):
    """Mean amplitude of specific subcarrier."""
    # Extract subcarrier 0 amplitudes
    sc0_values = [sample.amplitudes[0] for sample in current_samples]
    return float(np.mean(sc0_values))
```

## Testing Features

After implementing features:

1. **List Features**: Verify registration

```bash
python -m csi_toolkit process --list-features
```

2. **Extract Features**: Test on data

```bash
python -m csi_toolkit process \
  data/test.csv \
  features/test_output.csv \
  --features your_feature_name
```

3. **Inspect Output**: Check CSV contains expected values

```bash
head features/test_output.csv
```

## Best Practices

1. **Return Type**: Always return `float`, not int or numpy types
2. **Handle Edge Cases**: Check for empty sample lists
3. **Error Handling**: Use try-except for robustness
4. **Descriptive Names**: Use clear, descriptive feature names
5. **Documentation**: Add docstrings explaining computation
6. **Performance**: Avoid unnecessary loops, use NumPy operations
7. **Validation**: Test on real data before using in production

## Error Handling Example

```python
@registry.register('robust_mean', description='Mean amplitude with error handling')
def robust_mean(current_samples, prev_samples, next_samples):
    """Calculate mean amplitude with robust error handling."""
    try:
        if not current_samples:
            return 0.0

        sample_means = get_sample_mean_amplitudes(current_samples)

        if len(sample_means) == 0:
            return 0.0

        return float(np.mean(sample_means))
    except Exception:
        # Return default value on error
        return 0.0
```

## Next Steps

- Review existing features in `src/csi_toolkit/processing/features/basic.py`
- Study temporal features in `src/csi_toolkit/processing/features/temporal.py`
- See [Architecture Guide](architecture.md) for system overview
- Check [Data Formats Reference](../reference/data-formats.md) for sample structure
