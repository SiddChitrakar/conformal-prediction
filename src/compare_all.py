#!/usr/bin/env python3
"""
Comprehensive Comparison: Standard vs. Ordinal Scoring.

Compares two approaches:
1. Standard conformal (standard scoring)
2. Ordinal conformal (ordinal-aware scoring)

Key question: Does ordinal-aware scoring improve prediction sets?
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.ordinal_metric import (
    compute_ordinal_metrics,
    generate_ordinal_data,
    get_class_names,
)
from src.ordinal_score import (
    OrdinalConformalClassifier,
    StandardConformalClassifier,
)


def run_all_methods(X_train, y_train, X_cal, y_cal, X_test, y_test, alpha=0.1):
    """Run both methods and return prediction sets."""
    # Standard scoring
    std_model = RandomForestClassifier(n_estimators=100, random_state=42)
    std_conformal = StandardConformalClassifier(std_model, alpha=alpha)
    std_conformal.fit(X_train, y_train, X_cal, y_cal)
    std_sets = std_conformal.predict_set(X_test)

    # Ordinal scoring
    ord_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ord_conformal = OrdinalConformalClassifier(ord_model, alpha=alpha)
    ord_conformal.fit(X_train, y_train, X_cal, y_cal)
    ord_sets = ord_conformal.predict_set(X_test)

    return std_sets, ord_sets, std_conformal, ord_conformal


def compute_all_metrics(std_sets, ord_sets, y_test, classes):
    """Compute metrics for both methods."""
    std_metrics = compute_ordinal_metrics(std_sets, y_test, classes)
    ord_metrics = compute_ordinal_metrics(ord_sets, y_test, classes)
    return std_metrics, ord_metrics


def plot_coverage_vs_alpha(
    X_train, y_train, X_cal, y_cal, X_test, y_test, classes, save_path=None
):
    """
    Compare coverage curves for standard vs ordinal scoring.

    ESSENTIAL: Shows the fundamental CP guarantee.
    """
    alphas = np.arange(0.02, 0.51, 0.02)
    std_coverages = []
    ord_coverages = []
    std_sizes = []
    ord_sizes = []

    for alpha in alphas:
        std_model = RandomForestClassifier(n_estimators=100, random_state=42)
        std_conformal = StandardConformalClassifier(std_model, alpha=alpha)
        std_conformal.fit(X_train, y_train, X_cal, y_cal)
        std_sets = std_conformal.predict_set(X_test)

        ord_model = RandomForestClassifier(n_estimators=100, random_state=42)
        ord_conformal = OrdinalConformalClassifier(ord_model, alpha=alpha)
        ord_conformal.fit(X_train, y_train, X_cal, y_cal)
        ord_sets = ord_conformal.predict_set(X_test)

        std_metrics = compute_ordinal_metrics(std_sets, y_test, classes)
        ord_metrics = compute_ordinal_metrics(ord_sets, y_test, classes)

        std_coverages.append(std_metrics["coverage"])
        ord_coverages.append(ord_metrics["coverage"])
        std_sizes.append(std_metrics["avg_set_size"])
        ord_sizes.append(ord_metrics["avg_set_size"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Coverage and Efficiency vs. Target Alpha", fontsize=14, fontweight="bold"
    )

    # Coverage curves
    ax1 = axes[0]
    ax1.plot(alphas, std_coverages, "b-", linewidth=2, label="Standard Scoring")
    ax1.plot(alphas, ord_coverages, "orange", linewidth=2, label="Ordinal Scoring")
    ax1.plot(alphas, 1 - np.array(alphas), "r--", linewidth=2, label="Target (1-alpha)")
    ax1.set_xlabel("Target Error Rate (alpha)")
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_title("Coverage Guarantee", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Set size curves
    ax2 = axes[1]
    ax2.plot(alphas, std_sizes, "b-", linewidth=2, label="Standard Scoring")
    ax2.plot(alphas, ord_sizes, "orange", linewidth=2, label="Ordinal Scoring")
    ax2.set_xlabel("Target Error Rate (alpha)")
    ax2.set_ylabel("Average Set Size")
    ax2.set_title("Prediction Set Efficiency", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_examples_comparison(
    X_test,
    y_test,
    std_conformal,
    ord_conformal,
    std_sets,
    ord_sets,
    class_names,
    save_path=None,
):
    """
    Show side-by-side prediction examples.

    ESSENTIAL: Shows what prediction sets actually look like.
    """
    n_examples = 6
    classes = std_conformal.classes_
    assert classes is not None

    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 4 * n_examples))
    fig.suptitle(
        "Standard vs. Ordinal Scoring: Example Predictions",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (std_ax, ord_ax) in enumerate(zip(axes[:, 0], axes[:, 1], strict=True)):
        if idx >= n_examples:
            break

        probs_std = std_conformal.base_model.predict_proba(X_test[idx : idx + 1])[0]
        probs_ord = ord_conformal.base_model.predict_proba(X_test[idx : idx + 1])[0]
        true_idx = np.where(classes == y_test[idx])[0][0]

        std_included = np.where(std_sets[idx])[0]
        ord_included = np.where(ord_sets[idx])[0]

        # Standard
        colors = ["red" if i == true_idx else "steelblue" for i in range(len(classes))]
        bars_std = std_ax.bar(range(len(classes)), probs_std, color=colors, alpha=0.7)
        for i in std_included:
            bars_std[i].set_edgecolor("green")
            bars_std[i].set_linewidth(3)
        std_ax.set_xticks(range(len(classes)))
        std_ax.set_xticklabels(class_names, rotation=45, ha="right")
        std_ax.set_title(f"Standard: Set = {[class_names[i] for i in std_included]}")
        std_ax.set_ylim(0, max(probs_std) * 1.2)

        # Ordinal
        bars_ord = ord_ax.bar(range(len(classes)), probs_ord, color=colors, alpha=0.7)
        for i in ord_included:
            bars_ord[i].set_edgecolor("green")
            bars_ord[i].set_linewidth(3)
        ord_ax.set_xticks(range(len(classes)))
        ord_ax.set_xticklabels(class_names, rotation=45, ha="right")
        ord_ax.set_title(f"Ordinal: Set = {[class_names[i] for i in ord_included]}")
        ord_ax.set_ylim(0, max(probs_ord) * 1.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_coverage_by_class(
    std_sets, ord_sets, y_test, classes, class_names, save_path=None
):
    """
    Coverage and contiguity by true class.

    INSIGHTFUL: Shows which stages are hardest to cover and if ordinal scoring helps.
    """
    n_classes = len(classes)

    std_coverage = []
    ord_coverage = []
    std_contiguity = []
    ord_contiguity = []
    counts = []

    for c in range(n_classes):
        class_mask = y_test == c
        n_in_class = np.sum(class_mask)
        counts.append(n_in_class)

        if n_in_class > 0:
            # Coverage
            std_cov = np.mean(
                [
                    std_sets[i, np.where(classes == y_test[i])[0][0]]
                    for i in range(len(y_test))
                    if y_test[i] == c
                ]
            )
            ord_cov = np.mean(
                [
                    ord_sets[i, np.where(classes == y_test[i])[0][0]]
                    for i in range(len(y_test))
                    if y_test[i] == c
                ]
            )
            std_coverage.append(std_cov)
            ord_coverage.append(ord_cov)

            # Contiguity
            std_contig = []
            ord_contig = []
            for i in range(len(y_test)):
                if y_test[i] == c:
                    std_inc = np.where(std_sets[i])[0]
                    ord_inc = np.where(ord_sets[i])[0]

                    if len(std_inc) > 0:
                        std_exp = set(range(min(std_inc), max(std_inc) + 1))
                        std_contig.append(1 if set(std_inc) == std_exp else 0)

                    if len(ord_inc) > 0:
                        ord_exp = set(range(min(ord_inc), max(ord_inc) + 1))
                        ord_contig.append(1 if set(ord_inc) == ord_exp else 0)

            std_contiguity.append(np.mean(std_contig) if std_contig else 0)
            ord_contiguity.append(np.mean(ord_contig) if ord_contig else 0)
        else:
            std_coverage.append(0)
            ord_coverage.append(0)
            std_contiguity.append(0)
            ord_contiguity.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Coverage and Contiguity by True Class", fontsize=14, fontweight="bold"
    )

    x = np.arange(n_classes)
    width = 0.35

    # Coverage by class
    ax1 = axes[0]
    ax1.bar(x - width / 2, std_coverage, width, label="Standard", color="steelblue")
    ax1.bar(x + width / 2, ord_coverage, width, label="Ordinal", color="darkorange")
    ax1.set_ylabel("Coverage")
    ax1.set_title("Coverage by True Class", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=15, ha="right")
    ax1.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="Target (90%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 1.05)

    # Contiguity by class
    ax2 = axes[1]
    ax2.bar(x - width / 2, std_contiguity, width, label="Standard", color="steelblue")
    ax2.bar(x + width / 2, ord_contiguity, width, label="Ordinal", color="darkorange")
    ax2.set_ylabel("Contiguity Rate")
    ax2.set_title("Contiguity by True Class", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=15, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_set_size_comparison(std_sets, ord_sets, y_test, classes, save_path=None):
    """
    Compare set size distributions.

    USEFUL: Shows efficiency differences.
    """
    std_sizes = np.sum(std_sets, axis=1)
    ord_sizes = np.sum(ord_sets, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Prediction Set Size Comparison", fontsize=14, fontweight="bold")

    # Histograms
    ax1 = axes[0]
    bins = np.arange(0.5, max(std_sizes.max(), ord_sizes.max()) + 1.5, 1)
    ax1.hist(
        std_sizes,
        bins=bins,
        alpha=0.6,
        label=f"Standard (μ={std_sizes.mean():.2f})",
        color="steelblue",
    )
    ax1.hist(
        ord_sizes,
        bins=bins,
        alpha=0.6,
        label=f"Ordinal (μ={ord_sizes.mean():.2f})",
        color="darkorange",
    )
    ax1.set_xlabel("Set Size (number of classes)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution Comparison", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Paired difference
    ax2 = axes[1]
    differences = ord_sizes - std_sizes
    ax2.hist(differences, bins=15, color="purple", alpha=0.7, edgecolor="black")
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Ordinal Set Size - Standard Set Size")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Paired Difference", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.text(
        0.02,
        0.98,
        f"Mean diff: {differences.mean():.3f}",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Run comprehensive comparison."""
    ALPHA = 0.1
    N_CLASSES = 5
    N_SAMPLES = 2000
    CLASS_NAMES = get_class_names(N_CLASSES, ordinal=True)

    print("=" * 70)
    print("Comprehensive Comparison: Standard vs. Ordinal Scoring")
    print("=" * 70)
    print()
    print(f"Target coverage: {(1 - ALPHA) * 100}%")
    print(f"Classes: {CLASS_NAMES}")
    print()

    # Generate data
    X, y = generate_ordinal_data(n_samples=N_SAMPLES, n_classes=N_CLASSES)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Training: {len(X_train)}, Calibration: {len(X_cal)}, Test: {len(X_test)}")
    print()

    # Run both methods
    print("Running Standard and Ordinal scoring methods...")
    std_sets, ord_sets, std_conformal, ord_conformal = run_all_methods(
        X_train, y_train, X_cal, y_cal, X_test, y_test, alpha=ALPHA
    )

    assert std_conformal.classes_ is not None
    std_metrics, ord_metrics = compute_all_metrics(
        std_sets, ord_sets, y_test, std_conformal.classes_
    )

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Standard':>15} {'Ordinal':>15} {'Difference':>15}")
    print("-" * 70)
    print(
        f"{'Coverage':<25} "
        f"{std_metrics['coverage'] * 100:>14.2f}% "
        f"{ord_metrics['coverage'] * 100:>14.2f}% "
        f"{(ord_metrics['coverage'] - std_metrics['coverage']) * 100:>14.2f}%"
    )
    print(
        f"{'Avg Set Size':<25} "
        f"{std_metrics['avg_set_size']:>15.2f} "
        f"{ord_metrics['avg_set_size']:>15.2f} "
        f"{ord_metrics['avg_set_size'] - std_metrics['avg_set_size']:>15.2f}"
    )
    contig_diff = (
        ord_metrics["contiguity_rate"] - std_metrics["contiguity_rate"]
    ) * 100
    print(
        f"{'Contiguity Rate':<25} "
        f"{std_metrics['contiguity_rate'] * 100:>14.2f}% "
        f"{ord_metrics['contiguity_rate'] * 100:>14.2f}% "
        f"{contig_diff:>14.2f}%"
    )
    weighted_diff = (
        ord_metrics["weighted_error_rate"] - std_metrics["weighted_error_rate"]
    )
    print(
        f"{'Weighted Error':<25} "
        f"{std_metrics['weighted_error_rate']:>15.4f} "
        f"{ord_metrics['weighted_error_rate']:>15.4f} "
        f"{weighted_diff:>15.4f}"
    )
    print("=" * 70)

    # Generate essential visualizations
    print("\nGenerating essential figures...")

    plot_coverage_vs_alpha(
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        std_conformal.classes_,
        save_path="figures/coverage_vs_alpha_compare.png",
    )

    plot_examples_comparison(
        X_test,
        y_test,
        std_conformal,
        ord_conformal,
        std_sets,
        ord_sets,
        CLASS_NAMES,
        save_path="figures/examples_comparison.png",
    )

    plot_coverage_by_class(
        std_sets,
        ord_sets,
        y_test,
        std_conformal.classes_,
        CLASS_NAMES,
        save_path="figures/coverage_by_class.png",
    )

    plot_set_size_comparison(
        std_sets,
        ord_sets,
        y_test,
        std_conformal.classes_,
        save_path="figures/set_size_compare.png",
    )

    print("Done! Essential figures saved:")
    print("  - figures/coverage_vs_alpha_compare.png (CP guarantee)")
    print("  - figures/examples_comparison.png (example predictions)")
    print("  - figures/coverage_by_class.png (by stage)")
    print("  - figures/set_size_compare.png (efficiency)")


if __name__ == "__main__":
    main()
