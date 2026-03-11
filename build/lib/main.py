#!/usr/bin/env python3
"""
Conformal Prediction - Basic Example.

This module demonstrates split conformal prediction for multi-class classification.
For an ordinal-aware demo with visualizations, run:
    python -m src.ordinal_demo
"""

import numpy as np
from sklearn.datasets import make_classification
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
        """
        Initialize the conformal classifier.

        Args:
            base_model: A classifier with predict_proba method
            alpha: Target error rate (1 - alpha = desired coverage)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.classes_ = None
        self.calibration_scores = None

    def fit(self, X_train, y_train, X_cal, y_cal):
        """
        Train the base model and compute calibration scores.

        Args:
            X_train: Training features
            y_train: Training labels
            X_cal: Calibration features (must be disjoint from training data)
            y_cal: Calibration labels
        """
        # Train the base model
        self.base_model.fit(X_train, y_train)
        self.classes_ = self.base_model.classes_

        # Get probabilities on calibration set
        prob_cal = self.base_model.predict_proba(X_cal)

        # Compute conformity scores: s_i = 1 - f_hat(x_i)[y_i]
        # Higher score = less conformant (worse prediction for true label)
        self.calibration_scores = np.array(
            [
                1 - prob_cal[i, np.where(self.classes_ == y_cal[i])[0][0]]
                for i in range(len(y_cal))
            ]
        )

    def predict_set(self, X) -> np.ndarray:
        """
        Generate prediction sets for new examples.

        Returns a boolean matrix where result[i, j] = True means
        class j is in the prediction set for example i.

        Args:
            X: Test features

        Returns:
            prediction_sets: boolean matrix of shape (n_samples, n_classes)
        """
        if self.calibration_scores is None:
            raise ValueError("Must call fit() first")

        # Get probabilities for test examples
        prob_test = self.base_model.predict_proba(X)

        # Compute the quantile threshold for coverage 1 - alpha
        n_cal = len(self.calibration_scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_hat = np.quantile(self.calibration_scores, q_level, method="higher")

        # Prediction set: {y: 1 - f_hat(x)[y] ≤ q_hat}
        # Equivalent to: {y: f_hat(x)[y] ≥ 1 - q_hat}
        threshold = 1 - q_hat
        prediction_sets = prob_test >= threshold

        return prediction_sets

    def predict(self, X):
        """
        Return the single most likely class (standard prediction).
        """
        return self.base_model.predict(X)


def generate_synthetic_data(n_samples: int = 1000, n_classes: int = 3):
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=42,
    )
    return X, y


def main():
    """Run the conformal prediction demonstration."""
    print("=" * 60)
    print("Conformal Prediction - Basic Example")
    print("=" * 60)
    print()
    print("For ordinal-aware demo with visualizations, run:")
    print("    python -m src.ordinal_demo")
    print()
    print("=" * 60)

    # Configuration
    ALPHA = 0.1
    N_CLASSES = 4
    N_SAMPLES = 2000

    print(f"Target coverage: {(1 - ALPHA) * 100}%")
    print(f"Number of classes: {N_CLASSES}")
    print()

    # Generate and split data
    X, y = generate_synthetic_data(n_samples=N_SAMPLES, n_classes=N_CLASSES)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Train and evaluate
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    conformal_classifier = SplitConformalClassifier(base_model, alpha=ALPHA)
    conformal_classifier.fit(X_train, y_train, X_cal, y_cal)
    prediction_sets = conformal_classifier.predict_set(X_test)

    # Evaluate coverage
    assert conformal_classifier.classes_ is not None
    coverage = np.mean(
        [
            prediction_sets[
                i, np.where(conformal_classifier.classes_ == y_test[i])[0][0]
            ]
            for i in range(len(y_test))
        ]
    )
    avg_set_size = np.mean(np.sum(prediction_sets, axis=1))

    print("Results:")
    print("-" * 60)
    print(f"Empirical coverage: {coverage * 100:.2f}%")
    print(f"Target coverage:    {(1 - ALPHA) * 100:.2f}%")
    print(f"Average set size:   {avg_set_size:.2f} classes")


if __name__ == "__main__":
    main()
