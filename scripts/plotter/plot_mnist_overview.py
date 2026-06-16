"""Module for exploratory data analysis of the MNIST dataset, generating

a comprehensive overview figure that includes class distributions, random
samples, and average class representations.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from src.tools import load_flatten_mnist


def plot_mnist_overview(
    X, y, output_path="results/plots/mnist_overview.png", random_state=42
):
    """Generate a single composite figure showing:

    1. Class distribution bar chart.
    2. A random image sample for each class.
    3. The computed mean image for each class.

    Parameters:
    X (array-like): Input feature dataset (flattened or 3D images).
    y (array-like): Target class labels.
    output_path (str): Destination path to save the generated figure.
    random_state (int): Seed for reproducible random sampling.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rng = np.random.default_rng(random_state)

    X = np.asarray(X)
    y = np.asarray(y)

    # If flattened as (n_samples, 784), reshape to 28x28 matrix images
    if X.ndim == 2 and X.shape[1] == 784:
        X_images = X.reshape(-1, 28, 28)
    elif X.ndim == 3:
        X_images = X
    else:
        raise ValueError(
            f"Unrecognized shape for X. Received shape: {X.shape}"
        )

    classes = np.arange(10)
    class_counts = np.array([np.sum(y == c) for c in classes])

    # Extract a random sample per class
    sample_images = []
    for c in classes:
        class_indices = np.where(y == c)[0]

        if len(class_indices) == 0:
            raise ValueError(f"No samples found for class {c}")

        selected_idx = rng.choice(class_indices)
        sample_images.append(X_images[selected_idx])

    # Compute the mean image per class
    mean_images = []
    for c in classes:
        class_images = X_images[y == c]
        mean_image = np.mean(class_images, axis=0)
        mean_images.append(mean_image)

    # Create combined composite figure layout
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=10,
        height_ratios=[2.2, 1.4, 1.4],
        hspace=0.45,
        wspace=0.05,
    )

    # 1. Class Distribution Bar Chart
    ax_bar = fig.add_subplot(gs[0, :])
    ax_bar.bar(classes, class_counts)
    ax_bar.set_title("MNIST Class Distribution", fontsize=15)
    ax_bar.set_xlabel("Class / Digit")
    ax_bar.set_ylabel("Number of Instances")
    ax_bar.set_xticks(classes)
    ax_bar.grid(axis="y", alpha=0.3)

    for c, count in zip(classes, class_counts):
        ax_bar.text(
            c, count, str(count), ha="center", va="bottom", fontsize=9
        )

    # 2. Random Sample per Class
    for i, c in enumerate(classes):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(sample_images[i], cmap="gray")
        ax.set_title(f"Sample {c}", fontsize=10)
        ax.axis("off")

    # 3. Mean Image per Class
    for i, c in enumerate(classes):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(mean_images[i], cmap="gray")
        ax.set_title(f"Mean {c}", fontsize=10)
        ax.axis("off")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved at: {output_path}")


if __name__ == "__main__":
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_flatten_mnist(
        base_path="data/raw/"
    )

    # Utilize training partition for baseline visualization exploration
    plot_mnist_overview(
        X=X_train,
        y=y_train,
        output_path="results/plots/mnist_overview.png",
        random_state=42,
    )