# Experiment 3: Orthogonality Verification

*Generated: 2026-02-17 18:45:25*
*Runtime: 194s*

## Setup

- **Model:** google/gemma-2-2b-it (26 layers)
- **Truthfulness probe:** layer 16, C=0.01 (from Exp 1)
- **Deception probe:** layer 26, C=0.1 (from Exp 2)
- **Protocol:** Dev/test split upfront; probes trained on dev, evaluated on test

## 3.1 Cosine Similarity

- **cos(w_truth, w_decep) = -0.001204**
- **Angle: 89.9°**
- Truthfulness probe test AUROC: 0.8768
- Deception probe test AUROC: 0.9807

### Same-layer comparison

| Layer | cos(w_t, w_d) |
|-------|---------------|
| 16 | -0.004322 |
| 26 | -0.005003 |

**PASS:** Probes are near-orthogonal (cos=-0.0012 < 0.15 threshold).

## 3.2 Subspace Analysis (Principal Angles)

| k (PCA dims) | Mean Angle | Min Angle | Max Angle | Var Truth | Var Decep |
|-------------|------------|-----------|-----------|-----------|-----------|
| 1 | 89.8° | 89.8° | 89.8° | 0.065 | 0.387 |
| 2 | 88.2° | 87.5° | 89.0° | 0.126 | 0.544 |
| 5 | 87.9° | 85.4° | 89.6° | 0.221 | 0.707 |
| 10 | 86.6° | 82.2° | 89.7° | 0.296 | 0.791 |
| 20 | 85.1° | 78.0° | 90.0° | 0.384 | 0.844 |
| 50 | 81.8° | 69.8° | 89.9° | 0.524 | 0.904 |

### Cross-Subspace Capture (PCA-10)

- Deception variance captured by truthfulness PCA-10: -0.7021
- Truthfulness variance captured by deception PCA-10: -0.0098

## 3.3 Combined Probe

### Cross-Task AUROC

| Probe | Task | AUROC |
|-------|------|-------|
| Truthfulness | Truthfulness | 0.8768 |
| Deception | Deception | 0.9807 |
| Truthfulness | Deception | 0.7414 (cross) |
| Deception | Truthfulness | 0.5877 (cross) |

### Combined 2D Probe (detect any problem)

| Method | AUROC |
|--------|-------|
| Truth score only | 0.8167 |
| Decep score only | 0.7958 |
| **Combined [truth, decep]** | **0.8679** [0.8380, 0.8979] |

Combined probe weights: truth=8.1549, decep=4.5949

### Score Distributions

| Dataset | Truth Score (mean±std) | Decep Score (mean±std) |
|---------|----------------------|----------------------|
| Truthfulness data | 0.513±0.387 | 0.371±0.352 |
| Deception data | 0.800±0.171 | 0.509±0.477 |

## Interpretation

The truthfulness and deception probes are **near-orthogonal** (cos=-0.0012, angle=89.9°), consistent with the lambda_results finding of 99.6% orthogonal subspaces.

The combined probe (0.868) outperforms either individual probe (truth=0.817, decep=0.796), confirming additive value from orthogonal signals.
