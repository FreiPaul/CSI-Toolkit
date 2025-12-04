"""Feature-based plot implementations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .registry import registry


@registry.register(
    name="class_distribution",
    condition=lambda df: "label" in df.columns,
    description="Pie chart showing distribution of class labels",
    output_suffix="_class_distribution.png",
)
def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    """Generate a pie chart of class label distribution."""
    label_counts = df["label"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.Set3.colors[: len(label_counts)]
    wedges, texts, autotexts = ax.pie(
        label_counts.values,
        labels=[f"Class {label}" for label in label_counts.index],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )

    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")

    # Add legend with counts
    legend_labels = [
        f"Class {label}: {count} windows"
        for label, count in zip(label_counts.index, label_counts.values)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Classes",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    plt.tight_layout()
    return fig


@registry.register(
    name="amplitude_over_windows",
    condition=lambda df: "mean_amp" in df.columns and "std_amp" in df.columns,
    description="Line plot of mean_amp and std_amp over window_id",
    output_suffix="_amplitude_windows.png",
)
def plot_amplitude_over_windows(df: pd.DataFrame) -> plt.Figure:
    """Generate a line plot of amplitude features over window IDs."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Use window_id if available, otherwise use index
    if "window_id" in df.columns:
        x = df["window_id"]
        x_label = "Window ID"
    else:
        x = df.index
        x_label = "Window Index"

    # Calculate y-axis limits using 95th percentile
    mean_amp_p95 = np.percentile(df["mean_amp"], 95)
    std_amp_p95 = np.percentile(df["std_amp"], 95)
    mean_amp_p1 = np.percentile(df["mean_amp"], 1)
    std_amp_p1 = np.percentile(df["std_amp"], 1)

    # Add background coloring by class label if available
    if "label" in df.columns:
        labels = sorted(df["label"].unique())
        colors = plt.cm.Set1.colors
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(labels)}

        # Find contiguous regions of same label
        x_values = x.values if hasattr(x, 'values') else x
        label_values = df["label"].values

        # Add background spans for each contiguous region
        i = 0
        legend_handles = {}
        while i < len(label_values):
            current_label = label_values[i]
            start_x = x_values[i]
            # Find end of contiguous region
            j = i
            while j < len(label_values) and label_values[j] == current_label:
                j += 1
            end_x = x_values[j - 1]

            # Add small padding to make regions touch
            x_padding = (x_values[1] - x_values[0]) / 2 if len(x_values) > 1 else 0.5

            for ax in [ax1, ax2]:
                span = ax.axvspan(
                    start_x - x_padding,
                    end_x + x_padding,
                    alpha=0.2,
                    color=color_map[current_label],
                    label=f"Class {current_label}" if current_label not in legend_handles else None,
                )
                if current_label not in legend_handles:
                    legend_handles[current_label] = span

            i = j

    # Plot mean amplitude
    ax1.plot(x, df["mean_amp"], "k-", linewidth=0.8, alpha=0.9)
    ax1.set_ylabel("Mean Amplitude", fontsize=10)
    ax1.set_title("Mean Amplitude Over Windows", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(top=mean_amp_p95 * 1.05, bottom=mean_amp_p1 * 0.95)

    # Add legend for class colors
    if "label" in df.columns:
        ax1.legend(loc="upper right", fontsize=8)

    # Plot std amplitude
    ax2.plot(x, df["std_amp"], "k-", linewidth=0.8, alpha=0.9)
    ax2.set_xlabel(x_label, fontsize=10)
    ax2.set_ylabel("Std Amplitude", fontsize=10)
    ax2.set_title("Standard Deviation of Amplitude Over Windows", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(top=std_amp_p95 * 1.05, bottom=std_amp_p1 * 0.95)

    plt.tight_layout()
    return fig
