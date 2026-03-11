# Conformal Prediction for Ordinal Outcomes

Implementation of **split conformal prediction** with standard and ordinal-aware scoring methods, plus comprehensive evaluation metrics.

## Quick Start

```bash
make install
. .venv/bin/activate

# Run all demos
python -m src.main              # Standard evaluation
python -m src.compare           # Ordinal metrics breakdown
python -m src.compare_all       # Full scoring comparison
```

## Project Structure

| Script | Purpose | Key Figure |
|--------|---------|------------|
| `main.py` | Standard conformal prediction | Coverage guarantee |
| `ordinal_metric.py` | Ordinal evaluation metrics | Prediction examples gallery |
| `ordinal_score.py` | Ordinal-aware scoring implementation | — |
| `compare.py` | Standard + ordinal metrics | Error severity breakdown |
| `compare_all.py` | Standard vs. ordinal scoring | Coverage curves, by-class analysis |

## Essential Figures (6 total)

| Figure | Script | What It Shows |
|--------|--------|---------------|
| `standard_coverage.png` | `main.py` | **CP guarantee**: empirical coverage vs target (91% vs 90%) |
| `error_breakdown.png` | `compare.py` | **Why ordinal matters**: correct vs. near miss (7.8%) vs. far miss (1.3%) |
| `prediction_examples.png` | `ordinal_metric.py` | **What sets look like**: gallery by quality |
| `coverage_vs_alpha_compare.png` | `compare_all.py` | **CP guarantee across α**: standard vs. ordinal scoring |
| `coverage_by_class.png` | `compare_all.py` | **By-stage analysis**: coverage and contiguity for each cancer stage |
| `examples_comparison.png` | `compare_all.py` | **Side-by-side**: standard vs. ordinal predictions |
| `set_size_compare.png` | `compare_all.py` | **Efficiency**: set size distributions |

## Methods

### Standard Conformal Prediction
```
Score: s(x, y) = 1 - p(y|x)
```

### Ordinal Conformal Prediction
```
Score: s(x, y) = Σ_y' |y' - y| × p(y'|x)
```

Ordinal scoring penalizes probability mass on distant classes more than adjacent classes.

## Key Results

| Metric | Standard | Ordinal Scoring | Difference |
|--------|----------|-----------------|------------|
| Coverage | 90.89% | 90.67% | -0.22% |
| Avg Set Size | 1.72 | 1.82 | +6% |
| Contiguity Rate | 99.78% | 99.56% | -0.22% |
| Weighted Error | 0.0267 | 0.0294 | +0.0028 |

**Finding:** Ordinal scoring produces slightly larger sets but doesn't significantly improve ordinal metrics on this synthetic dataset. The base classifier already produces smooth probability distributions.

## Coverage by Stage

The `coverage_by_class.png` figure reveals:
- **Normal stage**: Highest coverage (~96%), best predicted
- **Stage 1-3**: Coverage near target (88-90%)
- **Stage 4**: No samples in test set (due to class imbalance)
- **Contiguity**: ~100% for all stages (both methods produce intervals)

This shows the method works consistently across stages, with slightly better performance on early/normal stages.

## Data

Synthetic ordinal data with realistic characteristics:
- Class imbalance (more early-stage, fewer late-stage)
- Heterogeneous variance (later stages more variable)
- 5% label noise
- Ordinal structure (adjacent classes closer in feature space)

**Note:** Results may differ on real ordinal datasets (medical staging, Likert scales, etc.).
