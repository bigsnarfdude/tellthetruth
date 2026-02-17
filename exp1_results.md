# Experiment 1: Truthfulness Probe Validation

*Generated: 2026-02-17 18:03:54*
*Runtime: 1408s*

## Setup

- **Model:** google/gemma-2-2b-it (26 layers)
- **Data:** TruthfulQA generation split (817 questions → 1634 paired samples)
- **Protocol:** 5-fold stratified CV, C ∈ {0.01, 0.1, 1.0, 10.0}
- **Bootstrap:** 1000 resamples for 95% CI

## 1.1 Full Layer Sweep

Best pooling: **last**

| Layer | AUROC (5-fold CV) | Best C |
|-------|-------------------|--------|
| 0 | 0.5286 | 0.01 |
| 1 | 0.5655 | 0.01 |
| 2 | 0.5801 | 1.0 |
| 3 | 0.5925 | 0.01 |
| 4 | 0.6043 | 0.01 |
| 5 | 0.6594 | 0.01 |
| 6 | 0.7107 | 0.01 |
| 7 | 0.7345 | 0.01 |
| 8 | 0.7522 | 0.01 |
| 9 | 0.7680 | 0.01 |
| 10 | 0.7775 | 0.01 |
| 11 | 0.8338 | 0.01 |
| 12 | 0.8432 | 0.01 |
| 13 | 0.8487 | 0.01 |
| 14 | 0.8746 | 0.01 |
| 15 | 0.8822 | 0.01 |
| 16 | 0.9009 | 0.01 | ← best
| 17 | 0.8897 | 0.01 |
| 18 | 0.8845 | 0.01 |
| 19 | 0.8811 | 0.01 |
| 20 | 0.8748 | 0.01 |
| 21 | 0.8685 | 0.01 |
| 22 | 0.8623 | 0.01 |
| 23 | 0.8575 | 0.01 |
| 24 | 0.8554 | 0.01 |
| 25 | 0.8510 | 0.01 |
| 26 | 0.8424 | 0.01 |

### Pooling comparison at layer 16

| Pooling | AUROC |
|---------|-------|
| mean | 0.7670 |
| first | 0.5132 |
| last | 0.9009 |

## 1.2 Final Probe (Test Set)

- **Layer:** 16
- **Pooling:** last
- **C:** 0.01

| Metric | Value |
|--------|-------|
| AUROC | 0.8768 [0.8358, 0.9103] |
| AUPRC | 0.8868 |
| Accuracy | 0.7737 |
| Precision | 0.7697 |
| Recall | 0.7791 |
| F1 | 0.7744 |
| Brier Score | 0.1528 |

### Classification Report

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| incorrect | 0.778 | 0.768 | 0.773 | 164.0 |
| correct | 0.770 | 0.779 | 0.774 | 163.0 |

### Calibration

| Bin Center | Mean Predicted | Mean Actual | Count |
|------------|---------------|-------------|-------|
| 0.05 | 0.032 | 0.074 | 81 |
| 0.15 | 0.150 | 0.276 | 29 |
| 0.25 | 0.254 | 0.360 | 25 |
| 0.35 | 0.351 | 0.462 | 13 |
| 0.45 | 0.456 | 0.500 | 14 |
| 0.55 | 0.553 | 0.500 | 14 |
| 0.65 | 0.648 | 0.333 | 15 |
| 0.75 | 0.758 | 0.600 | 15 |
| 0.85 | 0.854 | 0.793 | 29 |
| 0.95 | 0.975 | 0.902 | 92 |

## 1.3 Ablations

| Ablation | AUROC | Expected | Pass? |
|----------|-------|----------|-------|
| Random labels | 0.5052 ± 0.0250 | ~0.5 | YES |
| Shuffled activations | 0.4745 ± 0.0115 | ~0.5 | YES |
| BoW baseline | 0.3106 ± 0.0306 | < probe | YES |

### Pooling Strategy Comparison

| Pooling | AUROC (5-fold CV) |
|---------|-------------------|
| mean | 0.7670 ± 0.0220 |
| first | 0.5132 ± 0.0271 |
| last | 0.9009 ± 0.0048 |

## Interpretation

The truthfulness probe shows genuine signal in the activation space.
The probe (0.877) substantially outperforms BoW (0.311), confirming the signal is in the representation geometry, not surface vocabulary.
