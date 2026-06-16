"""Script for executing a stratified cross-validation benchmark comparing

a Neural Network Classifier (MLP) and a One-vs-One Support Vector Classifier
on the MNIST dataset.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.mlp import MLP
from src.models.svm import OneVsOneSVC
from src.tools import load_flatten_mnist

try:
    from argos.callbacks import ArgosGuard
except Exception:  # pragma: no cover
    ArgosGuard = None


# EXPERIMENTAL CONFIGURATION
SEED = 42
FOLDS = 10
MAX_SAMPLES = 15000

EPOCHS_NNC = 100
EPOCHS_SVC = 100
EVAL_EVERY = 1

BATCH_SIZE = 128
LEARNING_RATE_NNC = 0.001
LEARNING_RATE_SVC = 0.001

RESULTS_DIR = Path("results")
HISTORY_DIR = RESULTS_DIR / "histories"
SUMMARY_CSV = RESULTS_DIR / "mnist_folds_summary.csv"
CONFIG_JSON = RESULTS_DIR / "mnist_folds_config.json"


NNC_CONFIG = {
    "in_features": 784,
    "hidden_dims": [512, 256, 128, 64],
    "num_classes": 10,
    "dropout_prob": 0.2,
}

SVC_CONFIG = {
    "kernel": "rbf",
    "gamma": 0.01,
    "C": 1.0,
    "lr": LEARNING_RATE_SVC,
    "epochs": EPOCHS_SVC,
}


def ensure_dirs():
    """Ensure that the results and histories output directories exist."""
    RESULTS_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def stratified_subsample(X, y, max_samples, seed):
    """Take a stratified subsample so all classes are equally represented."""
    if max_samples is None or max_samples >= len(X):
        return X, y

    indices = np.arange(len(y))
    sampled_idx, _ = train_test_split(
        indices,
        train_size=max_samples,
        random_state=seed,
        stratify=y,
    )
    sampled_idx = np.sort(sampled_idx)
    return X[sampled_idx], y[sampled_idx]


def save_history(model_name, fold_idx, history):
    """Save the training loss and accuracy history to an NPZ file."""
    output_path = HISTORY_DIR / f"{model_name}_fold_{fold_idx:02d}.npz"
    np.savez(
        output_path,
        loss=np.asarray(history["loss"], dtype=np.float32),
        accuracy=np.asarray(history["accuracy"], dtype=np.float32),
    )
    return output_path


def save_config():
    """Save the current experimental configuration tracking parameters to a JSON file."""
    config = {
        "seed": SEED,
        "folds": FOLDS,
        "max_samples": MAX_SAMPLES,
        "epochs_nnc": EPOCHS_NNC,
        "epochs_svc": EPOCHS_SVC,
        "eval_every": EVAL_EVERY,
        "batch_size": BATCH_SIZE,
        "learning_rate_nnc": LEARNING_RATE_NNC,
        "learning_rate_svc": LEARNING_RATE_SVC,
        "nnc_config": NNC_CONFIG,
        "svc_config": SVC_CONFIG,
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    """Run the stratified cross-validation pipeline for both models."""
    ensure_dirs()
    save_config()

    np.random.seed(SEED)

    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_flatten_mnist(base_path="data/raw/")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    X, y = stratified_subsample(X_train, y_train, MAX_SAMPLES, SEED)

    print(f"Dataset for folds: X={X.shape}, y={y.shape}")
    print(f"Folds: {FOLDS}")

    skf = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    argos = (
        ArgosGuard(prefix="MNIST_FOLDS", check_memory=True)
        if ArgosGuard
        else None
    )

    rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print("\n" + "=" * 70)
        print(f"FOLD {fold_idx}/{FOLDS}")
        print("=" * 70)

        X_fold_train = X[train_idx]
        y_fold_train = y[train_idx]
        X_fold_val = X[val_idx]
        y_fold_val = y[val_idx]

        # Neural Network Classifier
        print("\nTraining NNC / MLP...")
        start_time = time.perf_counter()

        nnc = MLP(**NNC_CONFIG)
        nnc_history = nnc.fit(
            X_train=X_fold_train,
            y_train=y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            epochs=EPOCHS_NNC,
            learning_rate=LEARNING_RATE_NNC,
            batch_size=BATCH_SIZE,
            eval_every=EVAL_EVERY,
            verbose=True,
        )

        nnc_seconds = time.perf_counter() - start_time
        nnc_pred = nnc.predict(X_fold_val)
        nnc_acc = accuracy_score(y_fold_val, nnc_pred)
        nnc_hist_path = save_history("nnc", fold_idx, nnc_history)

        rows.append(
            {
                "fold": fold_idx,
                "model": "nnc",
                "final_accuracy": nnc_acc,
                "best_accuracy": np.nanmax(nnc_history["accuracy"]),
                "final_loss": nnc_history["loss"][-1],
                "seconds": nnc_seconds,
                "history_file": str(nnc_hist_path),
            }
        )

        # Support Vector Classifier One-vs-One
        print("\nTraining SVC One-vs-One...")
        start_time = time.perf_counter()

        svc = OneVsOneSVC(**SVC_CONFIG)
        svc_history = svc.fit(
            X=X_fold_train,
            y=y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            epochs=EPOCHS_SVC,
            eval_every=EVAL_EVERY,
            verbose=True,
        )

        svc_seconds = time.perf_counter() - start_time
        svc_pred = svc.predict(X_fold_val)
        svc_acc = accuracy_score(y_fold_val, svc_pred)
        svc_hist_path = save_history("svc", fold_idx, svc_history)

        rows.append(
            {
                "fold": fold_idx,
                "model": "svc_ovo",
                "final_accuracy": svc_acc,
                "best_accuracy": np.nanmax(svc_history["accuracy"]),
                "final_loss": svc_history["loss"][-1],
                "seconds": svc_seconds,
                "history_file": str(svc_hist_path),
            }
        )

        pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

        if argos:
            argos.send_heartbeat(
                f"Fold {fold_idx}/{FOLDS} completed | "
                f"NNC acc={nnc_acc:.4f} | SVC acc={svc_acc:.4f}"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        summary.groupby("model")["final_accuracy"].agg(
            ["mean", "std", "min", "max"]
        )
    )

    if argos:
        argos.on_process_end("MNIST folds benchmark completed.")

