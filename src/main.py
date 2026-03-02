#!/usr/bin/env python3
"""
Conformal Prediction Toy Example for Classification.

This module demonstrates split conformal prediction for multi-class classification.
The key idea is to create prediction sets with guaranteed marginal coverage.
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
    print("Conformal Prediction Toy Example")
    print("=" * 60)
    print()

    # Configuration
    ALPHA = 0.1  # Target error rate (90% coverage)
    N_CLASSES = 4
    N_SAMPLES = 2000

    print(f"Target coverage: {(1 - ALPHA) * 100}%")
    print(f"Number of classes: {N_CLASSES}")
    print()

    # Generate data
    X, y = generate_synthetic_data(n_samples=N_SAMPLES, n_classes=N_CLASSES)

    # Split into proper training, calibration, and test sets
    # 50% train, 25% calibrate, 25% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Calibration set size: {len(X_cal)}")
    print(f"Test set size: {len(X_test)}")
    print()

    # Initialize conformal classifier with a base model
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    conformal_classifier = SplitConformalClassifier(base_model, alpha=ALPHA)

    # Train with split conformal
    conformal_classifier.fit(X_train, y_train, X_cal, y_cal)

    # Generate prediction sets for test data
    prediction_sets = conformal_classifier.predict_set(X_test)

    # Evaluate coverage
    # Coverage = proportion of test samples where true class is in prediction set
    coverage = np.mean(
        [
            prediction_sets[
                i, np.where(conformal_classifier.classes_ == y_test[i])[0][0]
            ]
            for i in range(len(y_test))
        ]
    )

    # Average prediction set size
    avg_set_size = np.mean(np.sum(prediction_sets, axis=1))

    print("Results:")
    print("-" * 60)
    print(f"Empirical coverage: {coverage * 100:.2f}%")
    print(f"Target coverage:    {(1 - ALPHA) * 100:.2f}%")
    print(f"Average prediction set size: {avg_set_size:.2f} classes")
    print()

    # Show example predictions
    print("Example predictions (first 5 test samples):")
    print("-" * 60)
    assert conformal_classifier.classes_ is not None
    for i in range(5):
        row = prediction_sets[i]
        pred_set = [
            int(cls)
            for cls, included in zip(conformal_classifier.classes_, row, strict=True)
            if included
        ]
        true_class = int(y_test[i])
        standard_pred = int(conformal_classifier.predict(X_test[i : i + 1])[0])

        print(f"Sample {i + 1}:")
        print(f"  True class:          {true_class}")
        print(f"  Standard prediction: {standard_pred}")
        print(f"  Prediction set:      {pred_set}")
        print(f"  Correctly covered:   {'Yes' if true_class in pred_set else 'No'}")
        print()


if __name__ == "__main__":
    main()
