#!/usr/bin/env python3
"""
Conformal Prediction - Standard Evaluation.

Demonstrates split conformal prediction with standard evaluation metrics:
coverage and prediction set size.

For ordinal-aware evaluation, run: python -m src.ordinal_metric
For scoring comparison, run: python -m src.compare_all
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.ordinal_metric import SplitConformalClassifier, compute_ordinal_metrics


def generate_data(
    n_samples: int = 1000,
    n_classes: int = 5,
    n_features: int = 10,
    random_state: int = 42,
    ordinal: bool = False,
):
    """
    Generate synthetic classification data.

    Uses same generator as ordinal_metric.py for fair comparison.
    """
    rng = np.random.RandomState(random_state)

    if ordinal:
        X_list = []
        y_list = []

        weights = np.exp(-0.3 * np.arange(n_classes))
        weights = weights / weights.sum()
        samples_per_class = (weights * n_samples).astype(int)
        samples_per_class[-1] = n_samples - samples_per_class.sum()

        for class_idx in range(n_classes):
            center = np.zeros(n_features)
            center[0] = class_idx * 2.0
            center[1] = class_idx * 1.5

            scale = 1.2 + class_idx * 0.2

            class_samples = rng.normal(
                loc=center,
                scale=scale,
                size=(samples_per_class[class_idx], n_features),
            )

            X_list.append(class_samples)
            y_list.extend([class_idx] * samples_per_class[class_idx])

        X = np.vstack(X_list)
        y = np.array(y_list)

        n_noisy = int(len(y) * 0.05)
        noisy_idx = rng.choice(len(y), n_noisy, replace=False)
        for idx in noisy_idx:
            new_label = rng.randint(n_classes)
            y[idx] = new_label

        shuffle_idx = rng.permutation(len(y))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        return X, y
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=8,
            n_redundant=2,
            n_classes=n_classes,
            n_clusters_per_class=2,
            random_state=random_state,
        )
        return X, y


def get_class_names(n_classes: int) -> list[str]:
    """Get human-readable class names."""
    return [f"Class {i}" for i in range(n_classes)]


def plot_coverage_vs_set_size(
    y_test: np.ndarray,
    pred_sets: np.ndarray,
    classes: np.ndarray,
    alpha: float,
    save_path: str | None = None,
):
    """
    Plot coverage guarantee and set size distribution.

    ESSENTIAL: Shows the fundamental CP guarantee.
    """
    n_samples = len(y_test)

    coverage = np.mean(
        [pred_sets[i, np.where(classes == y_test[i])[0][0]] for i in range(n_samples)]
    )

    set_sizes = np.sum(pred_sets, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    target = 1 - alpha
    fig.suptitle(
        f"Conformal Prediction Results (alpha={alpha}, Target={target:.0%})",
        fontsize=14,
        fontweight="bold",
    )

    # Coverage
    ax1 = axes[0]
    colors = ["green" if coverage >= target else "red"]
    bars = ax1.bar(
        ["Empirical", "Target"],
        [coverage, target],
        color=[*colors, "gray"],
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_ylabel("Coverage")
    ax1.set_title("Coverage Guarantee", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=target, color="gray", linestyle="--", linewidth=2)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, [coverage, target], strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Set size distribution
    ax2 = axes[1]
    unique_sizes, counts = np.unique(set_sizes, return_counts=True)
    ax2.bar(unique_sizes, counts, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Prediction Set Size (classes)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Set Size Distribution", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    for size, count in zip(unique_sizes, counts, strict=True):
        ax2.text(size, count + 5, f"n={count}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Run standard conformal prediction demo."""
    ALPHA = 0.1
    N_CLASSES = 5
    N_SAMPLES = 2000
    CLASS_NAMES = get_class_names(N_CLASSES)

    print("=" * 60)
    print("Conformal Prediction - Standard Evaluation")
    print("=" * 60)
    print()
    print(f"Target coverage: {(1 - ALPHA) * 100}%")
    print(f"Classes: {CLASS_NAMES}")
    print()

    # Generate ordinal data (same as other scripts)
    X, y = generate_data(n_samples=N_SAMPLES, n_classes=N_CLASSES, ordinal=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Training: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")
    print()

    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    conformal = SplitConformalClassifier(base_model, alpha=ALPHA)
    conformal.fit(X_train, y_train, X_cal, y_cal)
    prediction_sets = conformal.predict_set(X_test)

    assert conformal.classes_ is not None

    coverage = np.mean(
        [
            prediction_sets[i, np.where(conformal.classes_ == y_test[i])[0][0]]
            for i in range(len(y_test))
        ]
    )
    avg_set_size = np.mean(np.sum(prediction_sets, axis=1))

    print("Standard Metrics:")
    print("-" * 40)
    print(f"Coverage:     {coverage * 100:.2f}%")
    print(f"Avg Set Size: {avg_set_size:.2f} classes")
    print()

    # Also show ordinal metrics
    ordinal_metrics = compute_ordinal_metrics(
        prediction_sets, y_test, conformal.classes_
    )
    print("Ordinal Metrics (reference):")
    print("-" * 40)
    print(f"Contiguity Rate: {ordinal_metrics['contiguity_rate'] * 100:.2f}%")
    print(f"Weighted Error:  {ordinal_metrics['weighted_error_rate']:.4f}")
    print("=" * 60)

    # Generate essential figure
    print("\nGenerating essential figure...")

    plot_coverage_vs_set_size(
        y_test,
        prediction_sets,
        conformal.classes_,
        ALPHA,
        save_path="figures/standard_coverage.png",
    )

    print("Done! Figure saved:")
    print("  - figures/standard_coverage.png (coverage guarantee)")



if __name__ == "__main__":
    main()
