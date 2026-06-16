"""Module for loading cross-validation training histories and generating

comparative visualization plots for loss, accuracy, and final metric distributions.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Directory path configurations
RESULTS_DIR = Path("results")
HISTORY_DIR = RESULTS_DIR / "histories"
PLOTS_DIR = RESULTS_DIR / "plots"
SUMMARY_CSV = RESULTS_DIR / "mnist_folds_summary.csv"


def load_histories(model_prefix):
    """Load and stack all NPZ history files matching a specific model prefix.

    Parameters:
    model_prefix (str): Prefix identifier of the model (e.g., 'nnc' or 'svc').

    Returns:
    tuple: Stacked numpy arrays containing (losses, accuracies).
    """
    # Find and sort all validation fold history records
    files = sorted(HISTORY_DIR.glob(f"{model_prefix}_fold_*.npz"))
    if not files:
        raise FileNotFoundError(f"No histories found for prefix: {model_prefix}")

    losses = []
    accuracies = []

    # Load individual matrices from each fold file
    for file in files:
        data = np.load(file)
        losses.append(data["loss"])
        accuracies.append(data["accuracy"])

    # Vertically stack rows to create a unified (folds, epochs) matrix
    return np.vstack(losses), np.vstack(accuracies)


def plot_metric(metric_name, nnc_values, svc_values, output_path):
    """Generate and save an epoch-wise line plot comparing average metrics.

    Parameters:
    metric_name (str): Name of the evaluation metric ('loss' or 'accuracy').
    nnc_values (numpy.ndarray): NNC metrics matrix.
    svc_values (numpy.ndarray): SVC metrics matrix.
    output_path (Path): Destination image save path.
    """
    # Set up epoch coordinate arrays matching the recorded timeline length
    epochs_nnc = np.arange(1, nnc_values.shape[1] + 1)
    epochs_svc = np.arange(1, svc_values.shape[1] + 1)

    # Compute average trajectories across all cross-validation folds
    nnc_mean = np.nanmean(nnc_values, axis=0)
    svc_mean = np.nanmean(svc_values, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_nnc, nnc_mean, label="NNC / MLP")
    plt.plot(epochs_svc, svc_mean, label="SVC One-vs-One")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"MNIST Comparative {metric_name.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_final_accuracy_boxplot(output_path):
    """Generate and save a boxplot showing final accuracy distribution across folds.

    Parameters:
    output_path (Path): Destination image save path.
    """
    if not SUMMARY_CSV.exists():
        return

    # Load the unified benchmark summaries dataframe
    df = pd.read_csv(SUMMARY_CSV)
    models = sorted(df["model"].unique())

    # Isolate target performance metric vectors for each individual model type
    data = [
        df[df["model"] == model]["final_accuracy"].values for model in models
    ]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=models)
    plt.ylabel("Final Accuracy")
    plt.title("Final Accuracy Distribution Across Folds")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    """Main execution sequence to load raw data logs and render performance plots."""
    # Ensure target directories exist before saving figures
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load historical runs records for both model frameworks
    nnc_loss, nnc_accuracy = load_histories("nnc")
    svc_loss, svc_accuracy = load_histories("svc")

    # Generate evaluation comparison trajectories and variance distribution boxplots
    plot_metric("loss", nnc_loss, svc_loss, PLOTS_DIR / "loss_comparison.png")
    plot_metric(
        "accuracy",
        nnc_accuracy,
        svc_accuracy,
        PLOTS_DIR / "accuracy_comparison.png",
    )
    plot_final_accuracy_boxplot(PLOTS_DIR / "final_accuracy_boxplot.png")

    print(f"Plots saved in: {PLOTS_DIR}")

