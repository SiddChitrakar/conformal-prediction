#!/usr/bin/env python3
"""
Ordinal Conformal Prediction Demo.

Demonstrates ordinal-aware evaluation metrics for conformal prediction sets.
Unlike standard classification, ordinal outcomes (e.g., cancer stages) have
natural ordering where some errors are worse than others.

Key insight: Evaluate prediction sets using ordinal metrics WITHOUT modifying
the conformal scoring function. This allows fair comparison between methods.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class SplitConformalClassifier:
    """
    Split conformal prediction for classification.

    Guarantees: P(Y ∈ C_hat(X)) ≥ 1 - alpha (marginally)
    where C_hat(X) is the prediction set and alpha is the error rate.
    """

    classes_: np.ndarray | None
    calibration_scores: np.ndarray | None

    def __init__(self, base_model, alpha: float = 0.1):
        self.base_model = base_model
        self.alpha = alpha
        self.classes_ = None
        self.calibration_scores = None

    def fit(self, X_train, y_train, X_cal, y_cal):
        self.base_model.fit(X_train, y_train)
        self.classes_ = self.base_model.classes_

        prob_cal = self.base_model.predict_proba(X_cal)
        self.calibration_scores = np.array(
            [
                1 - prob_cal[i, np.where(self.classes_ == y_cal[i])[0][0]]
                for i in range(len(y_cal))
            ]
        )

    def predict_set(self, X) -> np.ndarray:
        if self.calibration_scores is None:
            raise ValueError("Must call fit() first")

        prob_test = self.base_model.predict_proba(X)
        n_cal = len(self.calibration_scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_hat = np.quantile(self.calibration_scores, q_level, method="higher")

        threshold = 1 - q_hat
        return prob_test >= threshold


def generate_ordinal_data(
    n_samples: int = 1000, n_classes: int = 5, random_state: int = 42
):
    """
    Generate synthetic ordinal data with true ordinal structure.

    Simulates cancer staging (0-4) where adjacent stages have overlapping
    feature distributions, making them harder to distinguish than distant stages.
    """
    rng = np.random.RandomState(random_state)
    n_features = 10

    X_list = []
    y_list = []

    samples_per_class = n_samples // n_classes

    for class_idx in range(n_classes):
        # Class centers spaced along dimensions to create ordinal structure
        center = np.zeros(n_features)
        center[0] = class_idx * 2.0  # Primary ordinal dimension
        center[1] = class_idx * 1.5  # Secondary ordinal dimension

        # Add class-specific variation (overlap between adjacent classes)
        class_samples = rng.normal(
            loc=center,
            scale=1.2,
            size=(samples_per_class, n_features),
        )

        X_list.append(class_samples)
        y_list.extend([class_idx] * samples_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list)

    return X, y


def compute_ordinal_metrics(
    prediction_sets: np.ndarray, y_test: np.ndarray, classes: np.ndarray
):
    """
    Compute ordinal-aware evaluation metrics.

    These metrics evaluate prediction sets WITHOUT modifying the scoring.
    They measure how well the sets respect ordinal structure.

    Returns:
        coverage: Proportion where true class is in prediction set
        avg_set_size: Average number of classes in prediction set
        avg_ordinal_spread: Average (max - min) class index in set
        contiguity_rate: Proportion of sets that are contiguous intervals
        avg_max_gap: Average number of missing classes within the range
        weighted_error: Penalizes misses by distance to nearest included class
    """
    n_samples = len(y_test)

    # Standard metrics
    coverage = np.mean(
        [
            prediction_sets[i, np.where(classes == y_test[i])[0][0]]
            for i in range(n_samples)
        ]
    )
    avg_set_size = np.mean(np.sum(prediction_sets, axis=1))

    # Ordinal-specific metrics
    ordinal_spreads = []
    contiguous_count = 0
    max_gap_sizes = []

    for i in range(n_samples):
        included_indices = np.where(prediction_sets[i])[0]
        if len(included_indices) > 0:
            spread = max(included_indices) - min(included_indices)
            ordinal_spreads.append(spread)

            # Check for gaps (non-contiguity)
            expected = set(range(min(included_indices), max(included_indices) + 1))
            actual = set(included_indices)
            gaps = expected - actual
            max_gap_sizes.append(len(gaps))

            if len(gaps) == 0:
                contiguous_count += 1
        else:
            ordinal_spreads.append(0)
            max_gap_sizes.append(0)

    # Weighted error: penalize misses by distance to nearest included class
    weighted_errors = []
    for i in range(n_samples):
        true_idx = np.where(classes == y_test[i])[0][0]
        included_indices = np.where(prediction_sets[i])[0]

        if len(included_indices) == 0:
            weighted_errors.append(1.0)
        elif true_idx not in included_indices:
            min_distance = min(abs(true_idx - inc) for inc in included_indices)
            weighted_errors.append(min_distance / (len(classes) - 1))
        else:
            weighted_errors.append(0.0)

    return {
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "avg_ordinal_spread": np.mean(ordinal_spreads),
        "contiguity_rate": contiguous_count / n_samples,
        "avg_max_gap": np.mean(max_gap_sizes),
        "weighted_error_rate": np.mean(weighted_errors),
    }


def plot_prediction_sets_with_gaps(
    y_test: np.ndarray,
    prediction_sets: np.ndarray,
    classes: np.ndarray,
    stage_names: list[str],
    save_path: str | None = None,
):
    """
    Visualize prediction sets highlighting gaps (non-contiguous classes).

    Blue squares = predicted classes in the set
    Red squares = true class (when included)
    Orange X = gap (missing class that breaks contiguity)
    """
    n_samples = min(50, len(y_test))

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "Conformal Prediction Sets with Ordinal Gaps Highlighted",
        fontsize=14,
        fontweight="bold",
    )

    for i in range(n_samples):
        included = np.where(prediction_sets[i])[0]
        true_idx = np.where(classes == y_test[i])[0][0]

        if len(included) > 1:
            min_inc, max_inc = min(included), max(included)

            # Highlight the range
            ax.axhspan(min_inc - 0.4, max_inc + 0.4, alpha=0.15, color="gray")

            # Show included classes
            for j in included:
                color = "red" if j == true_idx else "steelblue"
                ax.scatter(
                    i, j, c=color, s=100, marker="s", alpha=0.8, edgecolors="white"
                )

            # Mark gaps
            for gap in range(min_inc + 1, max_inc):
                if gap not in included:
                    ax.scatter(i, gap, c="orange", s=150, marker="X", alpha=0.9)
        else:
            if len(included) == 1:
                j = included[0]
                color = "red" if j == true_idx else "steelblue"
                ax.scatter(i, j, c=color, s=100, marker="s", alpha=0.8)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(stage_names)
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Class")
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(-0.5, len(classes) - 0.5)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="steelblue", markersize=10
        ),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red", markersize=10),
        Line2D(
            [0], [0], marker="X", color="w", markerfacecolor="orange", markersize=12
        ),
    ]
    ax.legend(
        legend_elements,
        ["Predicted class", "True class", "Gap (missing class)"],
        loc="upper right",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_probability_distributions(
    X_test: np.ndarray,
    y_test: np.ndarray,
    conformal: SplitConformalClassifier,
    prediction_sets: np.ndarray,
    stage_names: list[str],
    save_path: str | None = None,
):
    """
    Show example probability distributions and resulting prediction sets.
    """
    n_examples = 6
    classes = conformal.classes_
    assert classes is not None

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Probability Distributions and Prediction Sets", fontsize=14, fontweight="bold"
    )
    axes = axes.flatten()

    for idx, ax in zip(range(n_examples), axes, strict=True):
        probs = conformal.base_model.predict_proba(X_test[idx : idx + 1])[0]
        true_idx = np.where(classes == y_test[idx])[0][0]
        included = np.where(prediction_sets[idx])[0]

        colors = ["red" if i == true_idx else "steelblue" for i in range(len(classes))]
        bars = ax.bar(
            range(len(classes)), probs, color=colors, alpha=0.7, edgecolor="black"
        )

        # Highlight included classes with green border
        for i in range(len(classes)):
            if i in included:
                bars[i].set_alpha(1.0)
                bars[i].set_edgecolor("green")
                bars[i].set_linewidth(3)

        ax.set_xlabel("Class")
        ax.set_ylabel("Predicted Probability")
        set_classes = [stage_names[i] for i in included]
        ax.set_title(
            f"Sample {idx + 1}\nTrue: {stage_names[true_idx]}\nSet: {set_classes}",
            fontsize=10,
        )
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(stage_names, rotation=45, ha="right")
        ax.set_ylim(0, max(probs) * 1.2)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_set_size_distribution(
    prediction_sets: np.ndarray, save_path: str | None = None
):
    """Plot distribution of prediction set sizes."""
    set_sizes = np.sum(prediction_sets, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Prediction Set Size Distribution", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    bins = np.arange(0.5, set_sizes.max() + 1.5, 1)
    ax1.hist(set_sizes, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Prediction Set Size (number of classes)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Set Sizes", fontsize=12)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    bp = ax2.boxplot(set_sizes, patch_artist=True, widths=0.4)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    ax2.set_ylabel("Prediction Set Size")
    ax2.set_title("Overall Distribution", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks([1])
    ax2.set_xticklabels(["All Samples"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ordinal_spread_by_class(
    y_test: np.ndarray,
    prediction_sets: np.ndarray,
    classes: np.ndarray,
    stage_names: list[str],
    save_path: str | None = None,
):
    """
    Plot ordinal spread of prediction sets grouped by true class.

    Shows how the "width" of prediction sets varies across the ordinal scale.
    """
    spreads_by_class: dict[int, list[float]] = {i: [] for i in range(len(classes))}

    for i in range(len(y_test)):
        true_idx = np.where(classes == y_test[i])[0][0]
        included = np.where(prediction_sets[i])[0]

        if len(included) > 0:
            spread = max(included) - min(included)
            spreads_by_class[true_idx].append(spread)

    _fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(stage_names))
    means: list[float] = [
        float(np.mean(spreads_by_class[i])) if spreads_by_class[i] else 0.0
        for i in range(len(classes))
    ]
    stds: list[float] = [
        float(np.std(spreads_by_class[i])) if spreads_by_class[i] else 0.0
        for i in range(len(classes))
    ]
    counts = [len(spreads_by_class[i]) for i in range(len(classes))]

    bars = ax.bar(x, means, yerr=stds, color="steelblue", alpha=0.7, capsize=5)

    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("True Class")
    ax.set_ylabel("Ordinal Spread (max - min class index)")
    ax.set_title("Prediction Set Spread by True Class", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_radar(metrics: dict, save_path: str | None = None):
    """Create a radar chart showing ordinal-aware metrics."""
    normalized = {
        "Coverage": metrics["coverage"],
        "Contiguity": metrics["contiguity_rate"],
        "1 - Weighted Error": 1 - metrics["weighted_error_rate"],
        "1 - Norm. Spread": 1 - metrics["avg_ordinal_spread"] / 4,
    }

    _fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    categories = list(normalized.keys())
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    values = list(normalized.values())
    values += values[:1]

    ax.plot(angles, values, "o-", linewidth=2, color="steelblue")
    ax.fill(angles, values, alpha=0.25, color="steelblue")

    # Polar plot methods - type: ignore for pyrefly
    ax.set_theta_offset(np.pi / 2)  # type: ignore[attr-defined]
    ax.set_theta_direction(-1)  # type: ignore[attr-defined]
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)  # type: ignore[attr-defined]
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])  # type: ignore[attr-defined]
    ax.set_ylim(0, 1)

    ax.set_title(
        "Ordinal-Aware Evaluation Metrics\n(Higher is better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Run the ordinal conformal prediction demo with visualizations."""
    # Configuration
    ALPHA = 0.1
    N_CLASSES = 5
    N_SAMPLES = 2000
    STAGE_NAMES = ["Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]

    # Generate ordinal data
    X, y = generate_ordinal_data(n_samples=N_SAMPLES, n_classes=N_CLASSES)

    # Split: 50% train, 25% calibrate, 25% test
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

    # Print metrics summary
    print("=" * 60)
    print("Ordinal Conformal Prediction - Evaluation Metrics")
    print("=" * 60)
    print(f"Target coverage: {(1 - ALPHA) * 100}%")
    print(f"Classes: {STAGE_NAMES}")
    print("-" * 60)
    print(f"Coverage:           {metrics['coverage'] * 100:.2f}%")
    print(f"Avg Set Size:       {metrics['avg_set_size']:.2f} classes")
    print(f"Avg Ordinal Spread: {metrics['avg_ordinal_spread']:.2f}")
    print(f"Contiguity Rate:    {metrics['contiguity_rate'] * 100:.2f}%")
    print(f"Avg Max Gap:        {metrics['avg_max_gap']:.2f}")
    print(f"Weighted Error:     {metrics['weighted_error_rate']:.4f}")
    print("=" * 60)

    # Generate visualizations
    print("\nGenerating visualizations in 'figures/' directory...")

    plot_prediction_sets_with_gaps(
        y_test,
        prediction_sets,
        conformal.classes_,
        STAGE_NAMES,
        save_path="figures/prediction_sets_with_gaps.png",
    )

    plot_probability_distributions(
        X_test,
        y_test,
        conformal,
        prediction_sets,
        STAGE_NAMES,
        save_path="figures/probability_distributions.png",
    )

    plot_set_size_distribution(
        prediction_sets,
        save_path="figures/set_size_distribution.png",
    )

    plot_ordinal_spread_by_class(
        y_test,
        prediction_sets,
        conformal.classes_,
        STAGE_NAMES,
        save_path="figures/ordinal_spread_by_class.png",
    )

    plot_metrics_radar(
        metrics,
        save_path="figures/metrics_radar.png",
    )

    print("Done! Figures saved:")
    print("  - figures/prediction_sets_with_gaps.png")
    print("  - figures/probability_distributions.png")
    print("  - figures/set_size_distribution.png")
    print("  - figures/ordinal_spread_by_class.png")
    print("  - figures/metrics_radar.png")


if __name__ == "__main__":
    main()
