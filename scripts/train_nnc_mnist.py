"""Script for training a Multilayer Perceptron (MLP) on the MNIST dataset

incorporating dropout regularization and external process monitoring callbacks.
"""

import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

# Add the project root to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.models.mlp import MLP
from src.tools import load_flatten_mnist

try:
    from argos.callbacks import ArgosGuard
except Exception:  # pragma: no cover
    ArgosGuard = None

if __name__ == "__main__":
    # Hyperparameters and training configurations
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MAX_SAMPLES = 15000
    BATCH_SIZE = 128

    print("Loading and normalizing the MNIST dataset...")
    X_train, y_train, X_test, y_test = load_flatten_mnist(
        base_path="data/raw/"
    )

    # Subsample the dataset up to the defined safe limit
    X_train = X_train[:MAX_SAMPLES]
    y_train = y_train[:MAX_SAMPLES]

    print(f"Training samples: {X_train.shape}")
    print(f"Testing samples: {X_test.shape}")

    print("\nStarting MLP training...")

    # Initialize the webhook training monitor guard
    argos = ArgosGuard(prefix="MLP_MNIST", check_memory=True)

    # Initialize the deep neural network architecture
    model = MLP(
        in_features=784,
        hidden_dims=[512, 256, 128, 64],
        num_classes=10,
        dropout_prob=0.5,
    )

    # Train the network using mini-batch gradient descent
    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        eval_every=10,
        verbose=True,
    )
    print(history.keys())

    # Perform inference over the unseen validation subset
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nTraining completed.")
    print(f"Final test accuracy: {accuracy * 100:.2f}%")

    # Send final summary status tracking notification
    argos.on_process_end(f"Test Accuracy: {accuracy * 100:.2f}%")