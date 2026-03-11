#!/usr/bin/env python3
"""
Comparison: Standard Scoring with Ordinal Metrics.

Shows that ordinal evaluation metrics reveal additional insights
even when using standard conformal prediction scoring.

Key insight: Same predictions, different interpretations.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.ordinal_metric import (
    SplitConformalClassifier,
    compute_ordinal_metrics,
    generate_ordinal_data,
    get_class_names,
)


def plot_error_breakdown(
    y_test: np.ndarray,
    pred_sets: np.ndarray,
    classes: np.ndarray,
    save_path: str | None = None,
):
    """
    Break down errors by ordinal severity.

    ESSENTIAL: Shows why ordinal evaluation matters.
    """
    n_samples = len(y_test)

    correct = 0
    near_miss = 0
    far_miss = 0
    empty = 0

    for i in range(n_samples):
        included = np.where(pred_sets[i])[0]
        true_idx = np.where(classes == y_test[i])[0][0]

        if len(included) == 0:
            empty += 1
        elif true_idx in included:
            correct += 1
        else:
            min_dist = min(abs(true_idx - inc) for inc in included)
            if min_dist == 1:
                near_miss += 1
            else:
                far_miss += 1

    _fig, ax = plt.subplots(figsize=(8, 6))

    categories = [
        "Correct\n(truth in set)",
        "Near Miss\n(adjacent)",
        "Far Miss\n(distant)",
        "Empty Set",
    ]
    counts = [correct, near_miss, far_miss, empty]
    colors = ["green", "orange", "red", "gray"]

    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor="black")

    ax.set_ylabel("Number of Samples")
    ax.set_title("Error Severity Breakdown (Ordinal-Aware)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    total = sum(counts)
    for bar, count in zip(bars, counts, strict=True):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"n={count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Run comparison demo."""
    ALPHA = 0.1
    N_CLASSES = 5
    N_SAMPLES = 2000
    CLASS_NAMES = get_class_names(N_CLASSES, ordinal=True)

    print("=" * 60)
    print("Standard Scoring + Ordinal Metrics")
    print("=" * 60)
    print()
    print(f"Target coverage: {(1 - ALPHA) * 100}%")
    print(f"Classes: {CLASS_NAMES}")
    print()

    # Generate data
    X, y = generate_ordinal_data(n_samples=N_SAMPLES, n_classes=N_CLASSES)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Train conformal predictor
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    conformal = SplitConformalClassifier(base_model, alpha=ALPHA)
    conformal.fit(X_train, y_train, X_cal, y_cal)
    prediction_sets = conformal.predict_set(X_test)

    assert conformal.classes_ is not None
    metrics = compute_ordinal_metrics(prediction_sets, y_test, conformal.classes_)

    # Print metrics
    print("Standard Scoring Results:")
    print("-" * 40)
    print(f"Coverage:        {metrics['coverage'] * 100:.2f}%")
    print(f"Avg Set Size:    {metrics['avg_set_size']:.2f} classes")
    print(f"Contiguity Rate: {metrics['contiguity_rate'] * 100:.2f}%")
    print(f"Weighted Error:  {metrics['weighted_error_rate']:.4f}")
    print("=" * 60)

    # Generate essential figure
    print("\nGenerating essential figure...")

    plot_error_breakdown(
        y_test,
        prediction_sets,
        conformal.classes_,
        save_path="figures/error_breakdown.png",
    )

    print("Done! Figure saved:")
    print("  - figures/error_breakdown.png (error severity)")


if __name__ == "__main__":
    main()
