#!/usr/bin/env python3
"""
Ordinal-Aware Conformal Prediction Scoring.

Implements conformal prediction with ordinal-aware nonconformity scores.
Unlike standard conformal prediction that treats all classes equally,
this method accounts for the ordinal structure of classes.

Key insight: The nonconformity score should penalize probability mass
on classes far from the true class more than classes close to it.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class OrdinalConformalClassifier:
    """
    Ordinal-aware split conformal prediction.

    Uses distance-weighted nonconformity scores:
        s(x, y) = Σ_y' |y' - y| * p(y'|x)

    This score is low when probability mass is concentrated near the true class,
    and high when mass is spread to distant classes.

    Prediction sets are constructed by including classes with low scores.
    """

    classes_: np.ndarray | None
    calibration_scores: np.ndarray | None
    class_distances_: np.ndarray | None

    def __init__(self, base_model, alpha: float = 0.1):
        self.base_model = base_model
        self.alpha = alpha
        self.classes_ = None
        self.calibration_scores = None
        self.class_distances_ = None

    def _compute_distance_matrix(self, n_classes: int) -> np.ndarray:
        """Compute distance matrix between ordinal classes."""
        class_indices = np.arange(n_classes)
        return np.abs(class_indices[:, np.newaxis] - class_indices[np.newaxis, :])

    def _compute_ordinal_score(self, probs: np.ndarray, true_class_idx: int) -> float:
        """
        Compute ordinal nonconformity score.

        Score = Σ_y' distance(y', y_true) * p(y'|x)

        Low score = probability mass concentrated near true class (good)
        High score = probability mass on distant classes (bad)
        """
        assert self.class_distances_ is not None
        distances = self.class_distances_[true_class_idx, :]
        return float(np.sum(distances * probs))

    def fit(self, X_train, y_train, X_cal, y_cal):
        """
        Fit the ordinal conformal predictor.

        Args:
            X_train, y_train: Training data for base model
            X_cal, y_cal: Calibration data for computing thresholds
        """
        self.base_model.fit(X_train, y_train)
        self.classes_ = self.base_model.classes_
        assert self.classes_ is not None
        n_classes = len(self.classes_)

        # Precompute distance matrix
        self.class_distances_ = self._compute_distance_matrix(n_classes)

        # Get probabilities on calibration set
        prob_cal = self.base_model.predict_proba(X_cal)

        # Compute ordinal nonconformity scores
        self.calibration_scores = np.array(
            [
                self._compute_ordinal_score(
                    prob_cal[i], np.where(self.classes_ == y_cal[i])[0][0]
                )
                for i in range(len(y_cal))
            ]
        )

    def predict_set(self, X) -> np.ndarray:
        """
        Generate prediction sets for test data.

        A class y is included if its ordinal score is below the threshold.
        """
        if self.calibration_scores is None:
            raise ValueError("Must call fit() first")

        assert self.classes_ is not None
        prob_test = self.base_model.predict_proba(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Compute quantile threshold
        n_cal = len(self.calibration_scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_hat = np.quantile(self.calibration_scores, q_level, method="higher")

        # For each test sample, compute score for each possible class
        prediction_sets = np.zeros((n_samples, n_classes), dtype=bool)

        for i in range(n_samples):
            for j in range(n_classes):
                score = self._compute_ordinal_score(prob_test[i], j)
                prediction_sets[i, j] = score <= q_hat

        return prediction_sets

    def predict(self, X):
        """Return standard point predictions."""
        return self.base_model.predict(X)


class StandardConformalClassifier:
    """
    Standard split conformal prediction (for comparison).

    Uses the standard nonconformity score:
        s(x, y) = 1 - p(y|x)

    This treats all classes equally, ignoring ordinal structure.
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

    def predict(self, X):
        return self.base_model.predict(X)


def compare_scoring_methods(
    X_train, y_train, X_cal, y_cal, X_test, y_test, alpha: float = 0.1
):
    """
    Compare standard vs ordinal scoring on the same data.

    Returns prediction sets from both methods.
    """
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
