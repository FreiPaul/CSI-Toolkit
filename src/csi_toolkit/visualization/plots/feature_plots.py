"""Feature-based plot implementations."""

import matplotlib.pyplot as plt
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

    # Plot mean amplitude
    ax1.plot(x, df["mean_amp"], "b-", linewidth=0.8, alpha=0.8)
    ax1.set_ylabel("Mean Amplitude", fontsize=10)
    ax1.set_title("Mean Amplitude Over Windows", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Add label coloring if available
    if "label" in df.columns:
        labels = df["label"].unique()
        colors = plt.cm.Set1.colors
        for i, label in enumerate(sorted(labels)):
            mask = df["label"] == label
            ax1.scatter(
                x[mask],
                df.loc[mask, "mean_amp"],
                c=[colors[i % len(colors)]],
                s=10,
                alpha=0.5,
                label=f"Class {label}",
            )
        ax1.legend(loc="upper right", fontsize=8)

    # Plot std amplitude
    ax2.plot(x, df["std_amp"], "r-", linewidth=0.8, alpha=0.8)
    ax2.set_xlabel(x_label, fontsize=10)
    ax2.set_ylabel("Std Amplitude", fontsize=10)
    ax2.set_title("Standard Deviation of Amplitude Over Windows", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Add label coloring if available
    if "label" in df.columns:
        for i, label in enumerate(sorted(labels)):
            mask = df["label"] == label
            ax2.scatter(
                x[mask],
                df.loc[mask, "std_amp"],
                c=[colors[i % len(colors)]],
                s=10,
                alpha=0.5,
                label=f"Class {label}",
            )
        ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig
