# Experiment 7: Gemma-3-4B-IT Truthfulness Probe

*Generated: 2026-02-18 21:23:09*
*Runtime: 1417s*

## Hypothesis

Gemma-2-2B-it's truthfulness probe (Exp 1) scores 0.877 in-distribution but drops
to 0.592 on free-form generation (Exp 4). A larger model (Gemma-3-4B-IT) may encode
truthfulness more abstractly, producing a probe that generalizes across formats.

## Setup

- **Model:** google/gemma-3-4b-it (34 layers, hidden_dim=2560)
- **Precision:** bf16 (no quantization)
- **Data:** TruthfulQA generation split (817 questions → 1634 paired samples)
- **Protocol:** 5-fold stratified CV, C ∈ {0.01, 0.1, 1.0, 10.0}
- **Bootstrap:** 1000 resamples for 95% CI

## 7.1 Full Layer Sweep

Best pooling: **last**

| Layer | AUROC (5-fold CV) | Best C |
|-------|-------------------|--------|
| 0 | 0.5421 | 0.01 |
| 1 | 0.5599 | 0.1 |
| 2 | 0.5849 | 0.01 |
| 3 | 0.5976 | 0.01 |
| 4 | 0.6153 | 0.01 |
| 5 | 0.6644 | 0.01 |
| 6 | 0.6553 | 0.01 |
| 7 | 0.6635 | 0.01 |
| 8 | 0.7091 | 0.01 |
| 9 | 0.7510 | 0.01 |
| 10 | 0.7666 | 0.01 |
| 11 | 0.8050 | 0.01 |
| 12 | 0.7934 | 0.01 |
| 13 | 0.8287 | 0.01 |
| 14 | 0.8396 | 0.01 |
| 15 | 0.8600 | 0.01 |
| 16 | 0.8893 | 0.01 |
| 17 | 0.8914 | 0.01 |
| 18 | 0.8884 | 0.01 |
| 19 | 0.8915 | 0.01 |
| 20 | 0.8953 | 0.01 | ← best
| 21 | 0.8872 | 0.01 |
| 22 | 0.8887 | 0.01 |
| 23 | 0.8775 | 0.01 |
| 24 | 0.8750 | 0.01 |
| 25 | 0.8755 | 0.01 |
| 26 | 0.8724 | 0.01 |
| 27 | 0.8721 | 0.01 |
| 28 | 0.8704 | 0.01 |
| 29 | 0.8665 | 0.01 |
| 30 | 0.8635 | 0.01 |
| 31 | 0.8597 | 0.01 |
| 32 | 0.8599 | 0.01 |
| 33 | 0.8574 | 0.01 |
| 34 | 0.8435 | 0.01 |

### Pooling comparison at layer 20

| Pooling | AUROC |
|---------|-------|
| mean | 0.7971 |
| first | 0.5180 |
| last | 0.8953 |

## 7.2 Final Probe (Test Set — In-Distribution)

- **Layer:** 20
- **Pooling:** last
- **C:** 0.01

| Metric | Gemma-3-4B | Gemma-2-2B (Exp 1) |
|--------|-----------|-------------------|
| AUROC | 0.8558 [0.8118, 0.8933] | 0.8770 [0.836, 0.910] |
| AUPRC | 0.8707 | 0.887 |
| Accuracy | 0.7554 | 0.774 |
| F1 | 0.7561 | 0.774 |
| Brier Score | 0.1704 | 0.153 |

**In-distribution difference:** -0.0212

### Classification Report

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| incorrect | 0.759 | 0.750 | 0.755 | 164.0 |
| correct | 0.752 | 0.761 | 0.756 | 163.0 |

### Calibration

| Bin Center | Mean Predicted | Mean Actual | Count |
|------------|---------------|-------------|-------|
| 0.05 | 0.033 | 0.093 | 75 |
| 0.15 | 0.146 | 0.297 | 37 |
| 0.25 | 0.261 | 0.400 | 10 |
| 0.35 | 0.351 | 0.357 | 14 |
| 0.45 | 0.451 | 0.462 | 26 |
| 0.55 | 0.553 | 0.533 | 15 |
| 0.65 | 0.661 | 0.625 | 16 |
| 0.75 | 0.741 | 0.471 | 17 |
| 0.85 | 0.859 | 0.593 | 27 |
| 0.95 | 0.976 | 0.911 | 90 |

## 7.3 Ablations

| Ablation | AUROC | Expected | Pass? |
|----------|-------|----------|-------|
| Random labels | 0.4998 ± 0.0222 | ~0.5 | YES |
| Shuffled activations | 0.5086 ± 0.0137 | ~0.5 | YES |
| BoW baseline | 0.3768 ± 0.0311 | < probe | YES |

### Pooling Strategy Comparison

| Pooling | AUROC (5-fold CV) |
|---------|-------------------|
| mean | 0.7971 ± 0.0209 |
| first | 0.5151 ± 0.0386 |
| last | 0.8953 ± 0.0093 |

## 7.4 OOD Evaluation (Free-Form Claims from Exp 4)

| Metric | Gemma-3-4B | Gemma-2-2B (Exp 1/4) |
|--------|-----------|---------------------|
| In-distribution AUROC | 0.8558 | 0.8770 |
| OOD AUROC (free-form) | **0.5327** [0.4415, 0.6153] | **0.5920** |
| OOD gap (in-dist - OOD) | **0.3231** | **0.2850** |
| OOD Accuracy | 0.5238 | — |
| OOD F1 | 0.6617 | — |
| Flag rate | 12.7% | 95.7% |

- **Claims evaluated:** 189 (101 correct, 88 incorrect)
- **Free-form activation norms:** 32223.5 ± 3243.2

## 7.5 Gap Comparison

| Metric | Gemma-2-2B | Gemma-3-4B | Change |
|--------|-----------|-----------|--------|
| In-dist AUROC | 0.8770 | 0.8558 | -0.0212 |
| OOD AUROC | 0.5920 | 0.5327 | -0.0593 |
| OOD gap | 0.2850 | 0.3231 | +0.0381 |
| Best layer | L16 | L20 | — |
| Hidden dim | 2304 (Gemma 2) | 2560 (Gemma 3) | +256 |

**The bigger model is worse OOD** (-0.059 AUROC). More parameters may lead to more format-specific representations, not more abstract ones. The truthfulness probe problem is fundamentally about what the probe training data teaches, not model capacity.

## Interpretation

The truthfulness probe shows genuine signal in Gemma-3-4B-IT's activation space.
The probe (CV 0.895, test 0.856) substantially outperforms BoW (0.377), confirming the signal is geometric, not vocabulary.

OOD performance (0.533) is comparable to Gemma-2-2B (0.592). Model scale does not fix the format generalization problem for truthfulness probes.
