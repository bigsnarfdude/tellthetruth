# Experiment 2: Deception Probe Validation

*Generated: 2026-02-17 18:26:22*
*Runtime: 398s*

## Setup

- **Model:** google/gemma-2-2b-it (26 layers)
- **Positives:** aligned scratchpad samples
- **Negatives:** AF organisms (probe>0.95) + potential_faking
- **Samples/class:** 500
- **Pooling:** last-token (best from Exp 1)
- **Protocol:** 5-fold stratified CV, C ∈ {0.01, 0.1, 1.0, 10.0}

## 2.1 Layer Sweep

| Layer | AUROC (5-fold CV) | Best C |
|-------|-------------------|--------|
| 0 | 0.6839 | 0.01 |
| 1 | 0.9632 | 0.1 |
| 2 | 0.9800 | 0.1 |
| 3 | 0.9768 | 0.1 |
| 4 | 0.9703 | 0.1 |
| 5 | 0.9771 | 0.1 |
| 6 | 0.9781 | 0.1 |
| 7 | 0.9737 | 0.1 |
| 8 | 0.9915 | 10.0 |
| 9 | 0.9919 | 10.0 |
| 10 | 0.9885 | 0.1 |
| 11 | 0.9829 | 0.01 |
| 12 | 0.9887 | 0.1 |
| 13 | 0.9889 | 0.1 |
| 14 | 0.9895 | 0.1 |
| 15 | 0.9887 | 0.01 |
| 16 | 0.9881 | 0.01 |
| 17 | 0.9884 | 0.1 |
| 18 | 0.9855 | 0.01 |
| 19 | 0.9843 | 0.01 |
| 20 | 0.9878 | 0.01 |
| 21 | 0.9882 | 0.01 |
| 22 | 0.9886 | 0.01 |
| 23 | 0.9904 | 0.1 |
| 24 | 0.9914 | 0.1 |
| 25 | 0.9927 | 0.1 |
| 26 | 0.9935 | 0.1 | ← best

## 2.2 Final Probe

- **Layer:** 26
- **C:** 0.1

### Test Set (in-distribution)

| Metric | Value |
|--------|-------|
| AUROC | 0.9807 [0.9591, 0.9959] |
| AUPRC | 0.9860 |
| Accuracy | 0.9550 |
| Precision | 0.9505 |
| Recall | 0.9600 |
| F1 | 0.9552 |
| Brier Score | 0.0409 |

### Classification Report

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| deceptive | 0.960 | 0.950 | 0.955 | 100.0 |
| honest | 0.950 | 0.960 | 0.955 | 100.0 |

### gold_106 (held-out, NEVER trained on)

- **Samples:** 106 (51 AF, 55 aligned)

| Metric | Value |
|--------|-------|
| AUROC | 0.6627 [0.5607, 0.7644] |
| Accuracy | 0.5472 |

### False Positive Rate (clean factual text)

- **Samples:** 30 factual statements
- **Predicted honest:** 1/30
- **False positive rate:** 0.967
- **Mean P(honest):** 0.0604

### Calibration

| Bin Center | Mean Predicted | Mean Actual | Count |
|------------|---------------|-------------|-------|
| 0.05 | 0.008 | 0.045 | 88 |
| 0.15 | 0.153 | 0.000 | 6 |
| 0.25 | 0.266 | 0.000 | 2 |
| 0.35 | 0.318 | 0.000 | 1 |
| 0.45 | 0.447 | 0.000 | 2 |
| 0.55 | 0.502 | 0.000 | 1 |
| 0.65 | 0.694 | 0.000 | 1 |
| 0.75 | 0.760 | 0.800 | 5 |
| 0.85 | 0.808 | 1.000 | 1 |
| 0.95 | 0.996 | 0.978 | 93 |

## 2.3 Ablations

| Ablation | AUROC | Expected | Pass? |
|----------|-------|----------|-------|
| Random labels | 0.4860 ± 0.0421 | ~0.5 | YES |
| Shuffled activations | 0.5231 ± 0.0489 | ~0.5 | YES |
| BoW baseline | 0.9973 ± 0.0042 | < probe | WARN |
| No scratchpad tags | 0.9965 ± 0.0020 | ≈ probe | YES |
| Truncated 128 tokens | 0.9329 ± 0.0140 | Signal in first 128? | Early |

## Interpretation

**WARNING:** BoW baseline (0.997) is high — the deception signal may be partially vocabulary-based (scratchpad reasoning uses distinctive language).

gold_106 AUROC: 0.663 — weak OOD generalization.
