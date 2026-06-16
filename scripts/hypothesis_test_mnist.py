"""Script for executing a paired hypothesis testing pipeline (Paired t-test,

Wilcoxon signed-rank test, Cohen's d effect size, and bootstrap confidence intervals)
comparing cross-validation histories from NNC and SVC models on MNIST.
"""

import glob
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# CONFIGURATION
HISTORIES_DIR = "results/histories"
OUTPUT_DIR = "results/statistics"

NNC_PATTERN = "nnc_fold_*.npz"
SVC_PATTERN = "svc_fold_*.npz"

ALPHA = 0.05


# UTILITIES
def load_history_file(path):
    """Load an NPZ file containing optimization training histories.

    Expected matrices:
        - loss
        - accuracy
    """
    data = np.load(path)
    history = {"loss": data["loss"], "accuracy": data["accuracy"]}
    return history


def extract_fold_number(path):
    """Extract the specific partition fold index from the file name string."""
    filename = os.path.basename(path)
    number = filename.split("_fold_")[-1].replace(".npz", "")
    return int(number)


def load_histories_by_model(histories_dir, pattern):
    """Load all cross-validation file history runs and sort them sequentially by fold."""
    files = glob.glob(os.path.join(histories_dir, pattern))
    records = []

    for file_path in files:
        fold = extract_fold_number(file_path)
        history = load_history_file(file_path)

        records.append(
            {
                "fold": fold,
                "path": file_path,
                "history": history,
                "final_loss": float(history["loss"][-1]),
                "final_accuracy": float(history["accuracy"][-1]),
            }
        )

    # Ensure records are aligned sequentially by fold index
    records = sorted(records, key=lambda x: x["fold"])
    return records


def paired_cohens_d(x, y):
    """Compute the paired sample variant of Cohen's d effect size metric."""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)


def bootstrap_ci_mean_diff(x, y, n_bootstrap=10000, ci=95, random_state=42):
    """Compute a bootstrap empirical confidence interval for the paired mean difference."""
    rng = np.random.default_rng(random_state)
    diff = x - y
    boot_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(diff, size=len(diff), replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - ((100 - ci) / 2))
    return lower, upper


# HYPOTHESIS TESTING
if __name__ == "__main__":
    """Execute the primary paired statistical testing suite over aligned CV folds."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nnc_records = load_histories_by_model(HISTORIES_DIR, NNC_PATTERN)
    svc_records = load_histories_by_model(HISTORIES_DIR, SVC_PATTERN)

    if len(nnc_records) == 0:
        raise FileNotFoundError(
            f"No NNC model histories found in {HISTORIES_DIR}"
        )

    if len(svc_records) == 0:
        raise FileNotFoundError(
            f"No SVC model histories found in {HISTORIES_DIR}"
        )

    nnc_by_fold = {r["fold"]: r for r in nnc_records}
    svc_by_fold = {r["fold"]: r for r in svc_records}

    # Identify matching cross-validation partitions across both frameworks
    common_folds = sorted(set(nnc_by_fold.keys()) & set(svc_by_fold.keys()))

    if len(common_folds) < 2:
        raise ValueError(
            "At least 2 common folds are required to run paired testing."
        )

    rows = []
    for fold in common_folds:
        nnc_acc = nnc_by_fold[fold]["final_accuracy"]
        svc_acc = svc_by_fold[fold]["final_accuracy"]

        nnc_loss = nnc_by_fold[fold]["final_loss"]
        svc_loss = svc_by_fold[fold]["final_loss"]

        rows.append(
            {
                "fold": fold,
                "nnc_final_accuracy": nnc_acc,
                "svc_final_accuracy": svc_acc,
                "accuracy_diff_nnc_minus_svc": nnc_acc - svc_acc,
                "nnc_final_loss": nnc_loss,
                "svc_final_loss": svc_loss,
                "loss_diff_nnc_minus_svc": nnc_loss - svc_loss,
            }
        )

    df = pd.DataFrame(rows)

    nnc_acc = df["nnc_final_accuracy"].to_numpy()
    svc_acc = df["svc_final_accuracy"].to_numpy()
    diff_acc = nnc_acc - svc_acc

    # Main Hypothesis Formulation
    # H0: mean(NNC_accuracy - SVC_accuracy) <= 0
    # H1: mean(NNC_accuracy - SVC_accuracy) > 0
    # H1 posits that the base Neural Network architecture outperforms the SVC.
    t_stat, p_value_ttest = stats.ttest_rel(
        nnc_acc, svc_acc, alternative="greater"
    )

    # Complementary non-parametric test
    try:
        wilcoxon_stat, p_value_wilcoxon = stats.wilcoxon(
            nnc_acc, svc_acc, alternative="greater"
        )
    except ValueError:
        wilcoxon_stat, p_value_wilcoxon = np.nan, np.nan

    # Calculate standardized effect size and bootstrap empirical boundaries
    effect_size = paired_cohens_d(nnc_acc, svc_acc)
    ci_lower, ci_upper = bootstrap_ci_mean_diff(nnc_acc, svc_acc)

    summary = {
        "n_folds": int(len(common_folds)),
        "alpha": ALPHA,
        "hypothesis": {
            "H0": "mean(NNC accuracy - SVC accuracy) <= 0",
            "H1": "mean(NNC accuracy - SVC accuracy) > 0",
        },
        "nnc_mean_accuracy": float(np.mean(nnc_acc)),
        "svc_mean_accuracy": float(np.mean(svc_acc)),
        "mean_accuracy_difference_nnc_minus_svc": float(np.mean(diff_acc)),
        "std_accuracy_difference": float(np.std(diff_acc, ddof=1)),
        "median_accuracy_difference": float(np.median(diff_acc)),
        "paired_t_test": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value_ttest),
            "reject_H0": bool(p_value_ttest < ALPHA),
        },
        "wilcoxon_signed_rank_test": {
            "statistic": (
                None if np.isnan(wilcoxon_stat) else float(wilcoxon_stat)
            ),
            "p_value": (
                None if np.isnan(p_value_wilcoxon) else float(p_value_wilcoxon)
            ),
            "reject_H0": (
                None
                if np.isnan(p_value_wilcoxon)
                else bool(p_value_wilcoxon < ALPHA)
            ),
        },
        "paired_cohens_d": float(effect_size),
        "bootstrap_95_ci_mean_accuracy_difference": {
            "lower": float(ci_lower),
            "upper": float(ci_upper),
        },
    }

    # Save outputs
    csv_path = os.path.join(OUTPUT_DIR, "mnist_hypothesis_test_by_fold.csv")
    json_path = os.path.join(OUTPUT_DIR, "mnist_hypothesis_test_summary.json")
    txt_path = os.path.join(OUTPUT_DIR, "mnist_hypothesis_test_report.txt")

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MNIST MODEL COMPARISON HYPOTHESIS TEST\n")
        f.write("=" * 60 + "\n\n")

        f.write("Hypothesis\n")
        f.write("-" * 60 + "\n")
        f.write("H0: mean(NNC accuracy - SVC accuracy) <= 0\n")
        f.write("H1: mean(NNC accuracy - SVC accuracy) > 0\n\n")

        f.write("Descriptive results\n")
        f.write("-" * 60 + "\n")
        f.write(f"Number of paired folds: {summary['n_folds']}\n")
        f.write(f"NNC mean accuracy: {summary['nnc_mean_accuracy']:.6f}\n")
        f.write(f"SVC mean accuracy: {summary['svc_mean_accuracy']:.6f}\n")
        f.write(
            "Mean difference NNC - SVC: "
            f"{summary['mean_accuracy_difference_nnc_minus_svc']:.6f}\n"
        )
        f.write(
            "Median difference NNC - SVC: "
            f"{summary['median_accuracy_difference']:.6f}\n"
        )
        f.write(
            "Std difference: "
            f"{summary['std_accuracy_difference']:.6f}\n\n"
        )

        f.write("Paired t-test\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"t statistic: {summary['paired_t_test']['t_statistic']:.6f}\n"
        )
        f.write(f"p-value: {summary['paired_t_test']['p_value']:.6f}\n")
        f.write(f"Reject H0: {summary['paired_t_test']['reject_H0']}\n\n")

        f.write("Wilcoxon signed-rank test\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"statistic: {summary['wilcoxon_signed_rank_test']['statistic']}\n"
        )
        f.write(
            f"p-value: {summary['wilcoxon_signed_rank_test']['p_value']}\n"
        )
        f.write(
            f"Reject H0: {summary['wilcoxon_signed_rank_test']['reject_H0']}\n\n"
        )

        f.write("Effect size\n")
        f.write("-" * 60 + "\n")
        f.write(f"Paired Cohen's d: {summary['paired_cohens_d']:.6f}\n\n")

        f.write("Bootstrap 95% CI\n")
        f.write("-" * 60 + "\n")
        f.write(f"Lower: {ci_lower:.6f}\n")
        f.write(f"Upper: {ci_upper:.6f}\n\n")

        if p_value_ttest < ALPHA:
            f.write(
                "Conclusion\n"
                "-" * 60 + "\n"
                "The paired t-test rejects H0 at alpha = 0.05. "
                "There is statistical evidence that NNC obtains higher "
                "accuracy than SVC on the evaluated folds.\n"
            )
        else:
            f.write(
                "Conclusion\n"
                "-" * 60 + "\n"
                "The paired t-test does not reject H0 at alpha = 0.05. "
                "There is not enough statistical evidence to conclude that "
                "NNC outperforms SVC on the evaluated folds.\n"
            )

    print("\nHypothesis test completed.")
    print(f"Paired folds: {len(common_folds)}")
    print(f"NNC mean accuracy: {np.mean(nnc_acc):.6f}")
    print(f"SVC mean accuracy: {np.mean(svc_acc):.6f}")
    print(f"Mean difference NNC - SVC: {np.mean(diff_acc):.6f}")
    print(f"Paired t-test p-value: {p_value_ttest:.6f}")
    print(f"Wilcoxon p-value: {p_value_wilcoxon}")
    print(f"Paired Cohen's d: {effect_size:.6f}")
    print(f"Bootstrap 95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")

    if p_value_ttest < ALPHA:
        print("\nDecision: Reject H0.")
        print(
            "Conclusion: Evidence supports that NNC performs better than SVC."
        )
    else:
        print("\nDecision: Do not reject H0.")
        print(
            "Conclusion: Evidence is insufficient to claim NNC is better than SVC."
        )

    print("\nSaved files:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {txt_path}")

