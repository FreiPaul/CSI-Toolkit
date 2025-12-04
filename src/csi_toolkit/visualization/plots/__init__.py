"""Plot registry and implementations for feature data visualization."""

from .registry import PlotConfig, PlotRegistry, registry

# Import plot implementations to trigger registration
from . import feature_plots

__all__ = ["PlotConfig", "PlotRegistry", "registry"]
