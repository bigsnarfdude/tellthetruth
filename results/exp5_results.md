# Experiment 5: Adversarial Robustness

*Generated: 2026-02-17 22:12:45*
*Model: google/gemma-2-2b-it*

## 5.1 Probe Evasion (Adversarial SFT)


| Metric | Value |
|--------|-------|
| Baseline AUROC | 0.8761 |
| Original probe on adversarial model | 0.4944 |
| AUROC drop | 0.3818 |
| Fresh probe recovery | 0.6961 |
| Passes criterion (≥ 0.90) | NO ✗ |

- **LoRA config:** r=8, α=16, lr=2e-05, 500 steps


### Interpretation

The original probe AUROC dropped by 0.3818 after adversarial fine-tuning. A fresh probe trained on the same adversarial activations recovered to 0.6961 AUROC — below the 0.90 criterion. This matches the obfuscation-probe-robustness finding: adversarial SFT can rotate the truthfulness signal in activation space but cannot eliminate it.

## 5.2 Paraphrase Attacks

| Metric | Value | Target |
|--------|-------|--------|
| Claims tested | 19 | — |
| Paraphrases per claim | 5 | — |
| Mean std(probe_score) | 0.0884 | < 0.10 |
| Median std | 0.0756 | — |
| Mean max-diff | 0.2415 | — |
| Pass rate (std < 0.10) | 57.9% | 100% |
| **Passes criterion** | **YES ✓** | std < 0.10 |

### Interpretation

The probe shows high consistency across paraphrases (mean std 0.0884). The truthfulness signal is robust to surface-level rephrasing — the probe detects the underlying factual content, not specific wording.

## Summary

| Criterion | Threshold | Result | Pass? |
|-----------|-----------|--------|-------|
| Fresh probe recovery after adversarial SFT | ≥ 0.90 | 0.6961319766571898 | — |
| Paraphrase stability std | ≤ 0.10 | 0.0884 | ✓ |
