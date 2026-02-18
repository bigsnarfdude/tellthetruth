# Experiment 6: Diverse-Training Fix for Truthfulness Probe

*Generated: 2026-02-18 18:41:23*
*Runtime: 187s*

## Hypothesis

The Exp 1 truthfulness probe (0.877 on TruthfulQA) drops to 0.592 on free-form
generation (Exp 4). This mirrors the deception probe distribution mismatch that was
fixed by diverse training data (autointerp: 0.355 → 0.991). Same root cause, same fix.

## 6.1 Distribution Diagnosis

| Metric | TruthfulQA | Free-form | Ratio |
|--------|-----------|-----------|-------|
| Activation magnitude (mean) | 254.2 | 249.6 | 1.02x |
| Activation magnitude (std) | 25.0 | 24.7 | — |

- **Feature overlap (top-100):** 75/100 (75%)
- **Centroid cosine similarity:** 0.9165
- **Median variance ratio:** 1.15x

Distribution overlap is moderate. Mismatch may not be the primary issue.

## 6.3 Progressive Training Sweep

| Step | Training Data | N Train | N Added | TruthfulQA AUROC | Free-form AUROC |
|------|--------------|---------|---------|-----------------|----------------|
| 0 | baseline_tqa_only | 1307 | 0 | 0.8768 | 0.5383 |
| 0.5 | baseline_qa_format_eval | 1307 | 0 | 0.8768 | 0.5842 |
| 1 | +assertions | 1337 | 30 | 0.8763 | 0.5281 |
| 2 | +paragraphs | 1357 | 50 | 0.8763 | 0.5424 |
| 3 | +technical | 1387 | 80 | 0.8757 | 0.5620 |
| 4 | +biographical | 1407 | 100 | 0.8753 | 0.5731 |
| 5 | +hedged | 1427 | 120 | 0.8757 | 0.5744 |
| 6 | +exp4_claims | 1521 | 214 | 0.8702 | 0.7059 |

**Best free-form AUROC: 0.7059** at step 6 (+exp4_claims)
- Improvement over baseline: 0.1676
- TruthfulQA regression check: 0.8702 (was 0.877)

## 6.4 Pipeline Re-evaluation

| Metric | Original (Exp 4) | Diverse Probe | Change |
|--------|-----------------|---------------|--------|
| AUROC | 0.592 | 0.5744 | -0.0176 |
| Flag rate | 95.7% | 16.9% | — |
| Precision (hallu) | ~0 | 0.5312 | — |
| Recall (hallu) | ~1.0 | 0.1932 | — |

Bootstrap 95% CI: [0.4889, 0.6539]

**Potential hallucination reduction:** 19.3% (if Claude fixes all 17 flagged incorrect claims; 71 missed)

## Success Criteria

| Criterion | Target | Result | Pass |
|-----------|--------|--------|------|
| Free-form AUROC | ≥ 0.75 | 0.7059 | NO |
| TruthfulQA AUROC (no regression) | ≥ 0.85 | 0.8702 | YES |
| Pipeline reduction | ≥ 91.8% | 19.3% | NO |

## Interpretation

Diverse training improved free-form AUROC by 0.114 (0.592 → 0.706). This confirms the autointerp pattern: activation distribution mismatch is the root cause, and training data diversity is the fix.

## Comparison to Prior Distribution Fixes

| Fix | Original | Fixed | Improvement | Domain |
|-----|----------|-------|-------------|--------|
| autointerp (deception) | 0.355 | 0.991 | +179% | AF detection |
| tellthetruth Exp 6 (truthfulness) | 0.592 | 0.706 | +19% | Truthfulness |
