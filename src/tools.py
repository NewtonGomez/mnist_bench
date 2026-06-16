import mlx.core as mx
import numpy as np
import subprocess
import logging


def load_flatten_mnist(base_path="data/raw/"):
    """
    Loads the MNIST .npy files, normalizes them to the [0, 1] range,
    and flattens the image matrices into 1D vectors.
    """
    # Load from disk (using numpy as a bridge)
    x_train_np = np.load(f"{base_path}train_images.npy")
    y_train_np = np.load(f"{base_path}train_labels.npy")
    x_test_np = np.load(f"{base_path}test_images.npy")
    y_test_np = np.load(f"{base_path}test_labels.npy")

    # Flatten (reshape) and normalize
    # From (N, 28, 28) we go to (N, 784)
    x_train_flat = x_train_np.reshape(x_train_np.shape[0], -1) / 255.0
    x_test_flat = x_test_np.reshape(x_test_np.shape[0], -1) / 255.0

    # Convert labels for SVM (from [0-9] to binary format {-1, 1}
    # if necessary, or keep them if you will do a One-vs-Rest scheme)

    # Transfer to MLX unified memory
    return (
        np.array(x_train_flat),
        np.array(y_train_np),
        np.array(x_test_flat),
        np.array(y_test_np)
    )
