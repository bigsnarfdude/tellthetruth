# Tell The Truth: Experiment Results Summary

*Project: Probe-based hallucination and deception detection in language models*
*Model: Gemma-2-2B-it (26 layers) on RTX 4070 Ti SUPER*
*Date: February 17, 2026*

---

## One-Line Summary

Truthfulness and deception are orthogonal signals in activation space, but neither probe generalizes well enough to replace language-level verification in production.

---

## Experiment 1: Truthfulness Probe

**Question:** Can a linear probe on hidden activations detect when Gemma knows its answer is wrong?

**Result:** Yes — 0.877 AUROC [0.836, 0.910] on TruthfulQA at layer 16 (last-token pooling).

| Key Finding | Value |
|-------------|-------|
| Best layer | 16 of 26 (middle-upper network) |
| Best pooling | Last-token (0.901 CV) >> mean (0.767) >> first (0.513) |
| Random labels ablation | 0.505 (confirms real signal) |
| Shuffled activations | 0.475 (signal is in geometry) |
| BoW baseline | 0.311 (not vocabulary-based) |

**Interpretation:** The model encodes truthfulness in its activation geometry, not surface vocabulary. The signal peaks in the upper-middle layers and concentrates at the last token position.

---

## Experiment 2: Deception Probe

**Question:** Can a linear probe detect deceptive intent in alignment-faking scratchpad reasoning?

**Result:** 0.981 AUROC in-distribution, but the signal is vocabulary-based.

| Key Finding | Value |
|-------------|-------|
| Best layer | 26 (final layer) |
| In-distribution test | 0.981 AUROC [0.959, 0.996] |
| gold_106 (OOD) | 0.663 AUROC — weak generalization |
| BoW baseline | **0.997** — matches probe |
| No scratchpad tags | 0.997 — tag removal doesn't matter |
| Truncated 128 tokens | 0.933 — signal appears early |
| False positive rate (clean text) | 96.7% — classifies everything as deceptive |

**Interpretation:** The deception "signal" is primarily distinctive vocabulary in AF scratchpad reasoning, not deeper activation geometry. The probe doesn't generalize to gold_106 (0.663) because it learned surface patterns, not intent. This contrasts with the truthfulness probe, where BoW was only 0.311.

---

## Experiment 3: Orthogonality Verification

**Question:** Are truthfulness and deception independent signals, or two names for the same thing?

**Result:** Near-perfectly orthogonal — cos(w_truth, w_decep) = -0.0012, angle = 89.9°.

| Key Finding | Value |
|-------------|-------|
| Cosine similarity | -0.0012 (threshold: < 0.15) |
| Angle | 89.9° (perfectly orthogonal = 90°) |
| Same-layer cos (layer 16) | -0.004 |
| Same-layer cos (layer 26) | -0.005 |
| PCA-1 subspace angle | 89.8° |
| PCA-50 subspace angle | 81.8° (still largely orthogonal) |
| Combined probe AUROC | 0.868 (vs truth-only 0.817, decep-only 0.796) |
| Cross-task: truth→decep | 0.741 (some format transfer) |
| Cross-task: decep→truth | 0.588 (near-chance) |

**Interpretation:** This replicates the lambda_results finding (99.6% orthogonal subspaces). A model can hallucinate without deceiving and deceive without hallucinating. The combined 2D probe outperforms either alone, confirming additive value from independent signals. The deception probe cannot detect factual errors, and the truthfulness probe cannot detect deceptive intent.

---

## Experiment 4: End-to-End Pipeline

**Question:** Does the probe-based pipeline actually reduce hallucinations in free-form generation?

**Result:** The probe fails on free-form text (0.584 AUROC). Claude self-check dominates (0.937). Intervention achieves 87.5% reduction, but Claude does the work, not the probe.

| Key Finding | Value |
|-------------|-------|
| Paragraphs generated | 20 (5 categories, 4 each) |
| Claims extracted | 248 total |
| Hallucination rate | 35.5% overall |
| Worst category | Technical specs (57.1%) |
| Best category | Paper citations (24.2%) |

### Method Comparison (189 verifiable claims)

| Method | AUROC | Accuracy | F1 |
|--------|-------|----------|-----|
| Claude self-check | **0.937** | **0.852** | **0.874** |
| Truthfulness probe | 0.584 | 0.524 | 0.250 |
| Random probe | 0.462 | 0.460 | 0.469 |

### Intervention

| Metric | Value |
|--------|-------|
| Claims flagged | 170 / 189 (90%) |
| True positives fixed | 77 / 88 incorrect |
| False positives | 16 correct claims wrongly retracted |
| Hallucination reduction | 87.5% |

**Interpretation:** The truthfulness probe trained on TruthfulQA Q&A pairs does not transfer to free-form factual generation. The domain shift is severe — the probe over-triggers on everything (170/189 flagged). The 87.5% reduction rate is misleading: it works because Claude's intervention step correctly distinguishes real errors from false alarms, not because the probe provides useful signal. The probe is essentially a no-op in this pipeline; Claude alone would perform better.

---

## Experiment 5: Adversarial Robustness

**Status:** Not yet run.

**Protocol:** Fine-tune Gemma-2-2B with adversarial loss opposing the truthfulness probe, then test if fresh probes recover. Based on obfuscation-probe-robustness findings that adversarial SFT can rotate the signal but never eliminate it.

---

## Cross-Cutting Findings

### What We Proved

1. **Truthfulness is a real geometric signal** — not vocabulary, not overfitting, not an artifact. AUROC 0.877 with all ablations passing (random labels ≈ 0.5, BoW = 0.31).

2. **Deception detection is vocabulary-dependent** — BoW matches the probe (0.997 vs 0.981). The AF scratchpad samples use distinctive language that any classifier can exploit. Weak OOD transfer (0.663 on gold_106).

3. **The two signals are orthogonal** — cos = -0.001, angle = 89.9°. A model can know something is false (truthfulness) without intending to deceive (deception), and vice versa. This matches lambda_results.

4. **Probes don't transfer across domains** — The truthfulness probe works well on TruthfulQA (0.877) but fails on free-form generation (0.584). Format matters: Q&A pairs ≠ generated paragraphs.

5. **Language-level verification beats activation probes** — Claude self-check (0.937) dramatically outperforms the probe (0.584) for detecting factual errors in generated text. The probe adds no value to the pipeline.

### What This Means for the RLFR Paper

The Goodfire "Features as Rewards" paper proposes using internal model features (probes) as reward signals for RL. Our replication suggests:

- **Probes work in-distribution** — on the exact format they were trained on, truthfulness probes provide strong signal (0.877-0.901 AUROC).
- **Probes fail out-of-distribution** — when applied to free-form generation (the actual use case), performance drops to near-random (0.584).
- **The reward signal may not generalize** — if the probe can't discriminate in the generation setting, it can't provide useful reward for RL training.
- **Domain-matched training data is critical** — the probe needs training examples that match the generation format, not just the conceptual task.

### Limitations

- **Single model:** All experiments on Gemma-2-2B-it. Larger models may behave differently.
- **Claude-as-judge:** Ground truth in Exp 4 comes from Claude, not human annotation. This introduces circular reasoning risk.
- **No RL training:** We test probes as classifiers, not as reward signals in an RL loop. The RLFR paper's results may depend on RL dynamics we don't capture.
- **Deception data quality:** AF organism samples from Mistral-7B may not represent natural deception patterns.

---

## Repository

**GitHub:** [bigsnarfdude/tellthetruth](https://github.com/bigsnarfdude/tellthetruth)

| File | Description |
|------|-------------|
| `exp1_truthfulness_probe.py` | Truthfulness probe: 26-layer sweep, ablations |
| `exp2_deception_probe.py` | Deception probe: AF organisms, gold_106 eval |
| `exp3_orthogonality.py` | Orthogonality: cosine, PCA, combined probe |
| `exp4_pipeline.py` | End-to-end: generation, verification, intervention |
| `EXPERIMENT_PROTOCOL.md` | Full experimental protocol |
| `exp[1-4]_results.{md,json}` | Raw results |
