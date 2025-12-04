"""Plot registry for feature data visualization."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class PlotConfig:
    """Configuration for a registered plot."""

    name: str
    func: Callable[[pd.DataFrame], plt.Figure]
    condition: Callable[[pd.DataFrame], bool]
    description: str
    output_suffix: str


class PlotRegistry:
    """Registry for plot functions with conditions."""

    def __init__(self):
        self._plots: Dict[str, PlotConfig] = {}

    def register(
        self,
        name: str,
        condition: Callable[[pd.DataFrame], bool],
        description: str = "",
        output_suffix: str = ".png",
    ):
        """Decorator to register a plot function.

        Args:
            name: Unique name for the plot
            condition: Function that checks if plot can be generated for given DataFrame
            description: Human-readable description of the plot
            output_suffix: Suffix for output filename (e.g., '_class_distribution.png')
        """

        def decorator(func: Callable[[pd.DataFrame], plt.Figure]):
            if name in self._plots:
                raise ValueError(f"Plot '{name}' is already registered")

            self._plots[name] = PlotConfig(
                name=name,
                func=func,
                condition=condition,
                description=description,
                output_suffix=output_suffix,
            )
            return func

        return decorator

    def get(self, name: str) -> PlotConfig:
        """Get a plot configuration by name."""
        if name not in self._plots:
            available = ", ".join(self.list_names())
            raise ValueError(f"Plot '{name}' not found. Available plots: {available}")
        return self._plots[name]

    def get_all(self) -> List[PlotConfig]:
        """Get all registered plots."""
        return list(self._plots.values())

    def list_names(self) -> List[str]:
        """List all registered plot names."""
        return list(self._plots.keys())

    def get_applicable(self, df: pd.DataFrame) -> List[PlotConfig]:
        """Get plots whose conditions are satisfied for the given DataFrame."""
        applicable = []
        for config in self._plots.values():
            try:
                if config.condition(df):
                    applicable.append(config)
            except Exception:
                # If condition check fails, skip this plot
                pass
        return applicable

    def get_by_names(self, names: List[str]) -> List[PlotConfig]:
        """Get specific plots by name with validation."""
        configs = []
        for name in names:
            configs.append(self.get(name))
        return configs

    def list_plots(self) -> str:
        """Get a formatted string listing all plots."""
        lines = ["Available plots:"]
        for config in self._plots.values():
            lines.append(f"  {config.name:25s} - {config.description}")
        return "\n".join(lines)


# Global registry instance
registry = PlotRegistry()
