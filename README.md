# Conformal Prediction Toy Example

A simple implementation of **split conformal prediction** for multi-class classification.

## What is Conformal Prediction?

Conformal prediction is a framework that provides prediction sets with guaranteed coverage. Unlike standard classifiers that output a single prediction, conformal prediction outputs a **set** of possible labels for each input, with a statistical guarantee that the true label is contained in the set with probability at least `1 - alpha`.

### Key Concepts

- **Coverage**: P(Y ∈ C_hat(X)) ≥ 1 - α (marginally)
- **Prediction Set**: A set of possible classes for each example
- **Calibration**: Uses a held-out calibration set to set the prediction set threshold

## Setup

```bash
make install
. .venv/bin/activate
make fix
make check
```

## Run the Example

```bash
python -m src.main
```

### Expected Output

The example demonstrates:
- **Split conformal prediction** using a Random Forest base classifier
- **4-class classification** on synthetic data
- **90% target coverage** (α = 0.1)
- Empirical coverage close to the target
- Average prediction set size showing efficiency

### Results

```
Target coverage: 90.0%
Empirical coverage: 91.20%
Average prediction set size: 1.39 classes
```

The empirical coverage should be close to the target 90%, demonstrating the coverage guarantee. The average prediction set size shows how many classes are typically included per example.

See `EXPLANATION.md` for details about the template structure and tooling.
