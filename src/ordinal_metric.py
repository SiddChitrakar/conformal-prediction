#!/usr/bin/env python3
"""
Ordinal Conformal Prediction - Evaluation Metrics.

Demonstrates ordinal-aware evaluation metrics for conformal prediction sets.
Unlike standard classification, ordinal outcomes (e.g., cancer stages) have
natural ordering where some errors are worse than others.

Key insight: Evaluate prediction sets using ordinal metrics WITHOUT modifying
the conformal scoring function. This allows fair comparison between methods.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def generate_ordinal_data(
    n_samples: int = 1000,
    n_classes: int = 5,
    n_features: int = 10,
    random_state: int = 42,
):
    """
    Generate synthetic ordinal data with realistic characteristics.

    Features:
    - Class imbalance: More early-stage, fewer late-stage samples
    - Heterogeneous variance: Later stages more variable
    - Label noise: ~5% mislabeled samples
    - Ordinal structure: Adjacent classes closer in feature space
    """
    rng = np.random.RandomState(random_state)

    X_list = []
    y_list = []

    # Class imbalance: more early stages, fewer late stages
    weights = np.exp(-0.3 * np.arange(n_classes))
    weights = weights / weights.sum()
    samples_per_class = (weights * n_samples).astype(int)
    samples_per_class[-1] = n_samples - samples_per_class.sum()

    for class_idx in range(n_classes):
        center = np.zeros(n_features)
        center[0] = class_idx * 2.0
        center[1] = class_idx * 1.5

        # Heterogeneous variance: later stages more variable
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

    # Add label noise (~5%)
    n_noisy = int(len(y) * 0.05)
    noisy_idx = rng.choice(len(y), n_noisy, replace=False)
    for idx in noisy_idx:
        new_label = rng.randint(n_classes)
        y[idx] = new_label

    # Shuffle
    shuffle_idx = rng.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y


def get_class_names(n_classes: int, ordinal: bool = False) -> list[str]:
    """Get human-readable class names."""
    if ordinal:
        names = ["Normal"]
        names.extend([f"Stage {i}" for i in range(1, n_classes)])
        return names[:n_classes]
    else:
        return [f"Class {i}" for i in range(n_classes)]


class SplitConformalClassifier:
    """Standard split conformal prediction."""

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


def compute_ordinal_metrics(
    prediction_sets: np.ndarray, y_test: np.ndarray, classes: np.ndarray
):
    """Compute ordinal-aware evaluation metrics."""
    n_samples = len(y_test)

    coverage = np.mean(
        [
            prediction_sets[i, np.where(classes == y_test[i])[0][0]]
            for i in range(n_samples)
        ]
    )
    avg_set_size = np.mean(np.sum(prediction_sets, axis=1))

    ordinal_spreads = []
    contiguous_count = 0
    max_gap_sizes = []

    for i in range(n_samples):
        included_indices = np.where(prediction_sets[i])[0]
        if len(included_indices) > 0:
            spread = max(included_indices) - min(included_indices)
            ordinal_spreads.append(spread)

            expected = set(range(min(included_indices), max(included_indices) + 1))
            actual = set(included_indices)
            gaps = expected - actual
            max_gap_sizes.append(len(gaps))

            if len(gaps) == 0:
                contiguous_count += 1
        else:
            ordinal_spreads.append(0)
            max_gap_sizes.append(0)

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


def plot_prediction_examples(
    X_test, y_test, conformal, pred_sets, stage_names, save_path=None
):
    """
    Show a gallery of prediction sets categorized by quality.

    USEFUL: Shows what conformal prediction sets look like.
    """
    classes = conformal.classes_
    assert classes is not None

    # Categorize examples
    perfect, good, uncertain, miss = [], [], [], []

    for i in range(len(y_test)):
        true_idx = np.where(classes == y_test[i])[0][0]
        included = np.where(pred_sets[i])[0]
        set_size = len(included)

        if true_idx in included:
            if set_size == 1:
                perfect.append(i)
            elif set_size <= 2:
                good.append(i)
            else:
                uncertain.append(i)
        else:
            miss.append(i)

    examples = {
        "Perfect (n=1, correct)": perfect[:3],
        "Good (n<=2, correct)": good[:3],
        "Uncertain (n>2, correct)": uncertain[:3],
        "Miss (incorrect)": miss[:3],
    }

    fig, axes = plt.subplots(4, 3, figsize=(14, 10))
    fig.suptitle("Prediction Set Examples by Quality", fontsize=14, fontweight="bold")

    for row, (category, indices) in enumerate(examples.items()):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            probs = conformal.base_model.predict_proba(X_test[idx : idx + 1])[0]
            true_idx = np.where(classes == y_test[idx])[0][0]
            included = np.where(pred_sets[idx])[0]

            colors = [
                "red" if i == true_idx else "steelblue" for i in range(len(classes))
            ]
            bars = ax.bar(
                range(len(classes)), probs, color=colors, alpha=0.7, edgecolor="black"
            )

            for i in range(len(classes)):
                if i in included:
                    bars[i].set_alpha(1.0)
                    bars[i].set_edgecolor("green")
                    bars[i].set_linewidth(3)

            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(stage_names, rotation=45, ha="right", fontsize=9)
            ax.set_ylim(0, max(probs) * 1.3 if max(probs) > 0 else 1)
            set_classes = [stage_names[i] for i in included]
            ax.set_title(
                f"True: {stage_names[true_idx]}\nSet: {set_classes}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3, axis="y")

            if col == 0:
                ax.set_ylabel(category, fontsize=10, fontweight="bold")
            else:
                ax.set_yticklabels([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Run the ordinal conformal prediction demo."""
    ALPHA = 0.1
    N_CLASSES = 5
    N_SAMPLES = 2000
    STAGE_NAMES = get_class_names(N_CLASSES, ordinal=True)

    print("=" * 60)
    print("Ordinal Conformal Prediction - Evaluation Metrics")
    print("=" * 60)
    print()
    print(f"Target coverage: {(1 - ALPHA) * 100}%")
    print(f"Classes: {STAGE_NAMES}")
    print()

    X, y = generate_ordinal_data(n_samples=N_SAMPLES, n_classes=N_CLASSES)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    conformal = SplitConformalClassifier(base_model, alpha=ALPHA)
    conformal.fit(X_train, y_train, X_cal, y_cal)
    prediction_sets = conformal.predict_set(X_test)

    assert conformal.classes_ is not None
    metrics = compute_ordinal_metrics(prediction_sets, y_test, conformal.classes_)

    print("Ordinal-Aware Evaluation Metrics:")
    print("-" * 60)
    print(f"Coverage:           {metrics['coverage'] * 100:.2f}%")
    print(f"Avg Set Size:       {metrics['avg_set_size']:.2f} classes")
    print(f"Avg Ordinal Spread: {metrics['avg_ordinal_spread']:.2f}")
    print(f"Contiguity Rate:    {metrics['contiguity_rate'] * 100:.2f}%")
    print(f"Avg Max Gap:        {metrics['avg_max_gap']:.2f}")
    print(f"Weighted Error:     {metrics['weighted_error_rate']:.4f}")
    print("=" * 60)

    # Generate essential figure
    print("\nGenerating essential figure...")

    plot_prediction_examples(
        X_test,
        y_test,
        conformal,
        prediction_sets,
        STAGE_NAMES,
        save_path="figures/prediction_examples.png",
    )

    print("Done! Figure saved:")
    print("  - figures/prediction_examples.png (quality gallery)")


if __name__ == "__main__":
    main()
