"""Data plotter for generating static plots from feature CSV files."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .plots import registry


class DataPlotter:
    """Generate static plots from processed feature CSV files."""

    def __init__(self, input_csv: str, display: bool = False):
        """Initialize the data plotter.

        Args:
            input_csv: Path to the input feature CSV file
            display: Whether to display plots interactively
        """
        self.input_path = Path(input_csv)
        self.output_dir = self.input_path.parent
        self.display = display

        # Load the data
        self.df = pd.read_csv(input_csv)

    def generate_plots(self, plot_names: Optional[List[str]] = None) -> List[Path]:
        """Generate plots and save them to the output directory.

        Args:
            plot_names: List of specific plot names to generate.
                       If None, generates all applicable plots.

        Returns:
            List of paths to generated plot files
        """
        # Determine which plots to generate
        if plot_names:
            # Get specific plots by name
            plots_to_generate = registry.get_by_names(plot_names)
            # Filter to only applicable plots
            plots_to_generate = [
                p for p in plots_to_generate if p.condition(self.df)
            ]
            if len(plots_to_generate) < len(plot_names):
                skipped = set(plot_names) - {p.name for p in plots_to_generate}
                for name in skipped:
                    print(f"Skipping '{name}': conditions not met for this dataset")
        else:
            # Get all applicable plots
            plots_to_generate = registry.get_applicable(self.df)

        if not plots_to_generate:
            print("No applicable plots found for this dataset.")
            print("Use --list-plots to see available plot types.")
            return []

        # Generate plots
        output_paths = []
        figures = []

        for plot_config in plots_to_generate:
            output_path = self.output_dir / f"{self.input_path.stem}{plot_config.output_suffix}"

            try:
                fig = plot_config.func(self.df)
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"Saved: {output_path}")
                output_paths.append(output_path)

                if self.display:
                    figures.append(fig)
                else:
                    plt.close(fig)

            except Exception as e:
                print(f"Error generating '{plot_config.name}': {e}")
                continue

        # Display all figures at once if requested
        if self.display and figures:
            plt.show()
            for fig in figures:
                plt.close(fig)

        return output_paths
