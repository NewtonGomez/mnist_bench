"""Script to train and evaluate a One-vs-One Support Vector Classifier (SVC)

on a subsampled MNIST dataset with automated metrics telemetry tracking.
"""

import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

# Add the project root to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.models.svm import OneVsOneSVC
from src.tools import load_flatten_mnist

try:
    from argos.callbacks import ArgosGuard
except Exception:  # pragma: no cover
    ArgosGuard = None

if __name__ == "__main__":
    # Hyperparameters and experiment constraints
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MAX_SAMPLES = 15000
    EVAL_EVERY = 1

    print("Loading and normalizing the MNIST dataset...")
    X_train, y_train, X_test, y_test = load_flatten_mnist(
        base_path="data/raw/"
    )

    print(f"Reducing training data to {MAX_SAMPLES} random samples...")
    X_train_np = np.asarray(X_train, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.int32)

    # Draw a reproducible stratified random subset of the training dataset
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_train_np))
    selected = indices[:MAX_SAMPLES]

    X_train_sub = X_train_np[selected]
    y_train_sub = y_train_np[selected]

    # Initialize the webhook telemetry monitoring guard
    argos = ArgosGuard(prefix="SVC_OVO_MNIST", check_memory=True)

    print("\nStarting One-vs-One SVC training...")

    # Initialize the multi-class ensemble of support vector machines
    model = OneVsOneSVC(
        kernel="rbf",
        gamma=0.01,
        C=1.0,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
    )

    # Train the model ensemble over the subsampled pairwise combinations
    history = model.fit(
        X=X_train_sub,
        y=y_train_sub,
        X_val=X_test,
        y_val=y_test,
        epochs=EPOCHS,
        eval_every=EVAL_EVERY,
        verbose=True,
    )

    # Evaluate final generalization accuracy across the full test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nTraining completed.")
    print(f"Final test accuracy: {accuracy * 100:.2f}%")
    print(f"History keys: {history.keys()}")

    # Dispatch final verification summary metrics to the monitoring interface
    argos.on_process_end(f"Test Accuracy: {accuracy * 100:.2f}%")
