# Tell The Truth: Results Summary

*Completed: 2026-02-17*
*Model: google/gemma-2-2b-it (26 layers) | Hardware: RTX 4070 Ti SUPER on nigel.birs.ca*
*Replicating: Goodfire AI "Features as Rewards" (Feb 2026)*

---

## One-Line Summary

Truthfulness and deception are orthogonal signals in activation space. A probe-based pipeline
achieves 91.8% hallucination reduction — without RL — by over-flagging with the probe and
delegating correction to Claude-sonnet-4-6. The probe itself barely beats random on free-form
generation (0.592 AUROC); the work is done by the LLM intervention step.

---

## Protocol Criteria — Final Scorecard

| Criterion | Threshold | Result | Pass? |
|-----------|-----------|--------|-------|
| Truthfulness probe AUROC (in-dist) | ≥ 0.75 | **0.877** | ✓ |
| Deception probe AUROC (in-dist) | ≥ 0.90 | **0.981** | ✓ |
| Deception probe AUROC (gold_106 OOD) | ≥ 0.90 | 0.663 | ✗ |
| Orthogonality cos(w_t, w_d) | ≤ 0.15 | **-0.0012** | ✓ |
| Pipeline hallucination reduction | ≥ 30% | **91.8%** | ✓ |
| Fresh probe recovery after adv. SFT | ≥ 0.90 | 0.696 | ✗ |
| Paraphrase stability std | ≤ 0.10 | **0.088** | ✓ |

**5 of 7 criteria pass.** Failures: gold_106 OOD generalization and fresh-probe adversarial recovery.

---

## Experiment 1: Truthfulness Probe

**Layer 16, last-token pooling, C=0.01**

| Metric | Value |
|--------|-------|
| AUROC (test) | **0.877** [0.836, 0.910] |
| AUPRC | 0.887 |
| Accuracy / F1 | 0.774 / 0.774 |
| Brier Score | 0.153 |

**Layer sweep:** Signal peaks at layer 16 (CV AUROC 0.901). Falls off toward both ends.
Last-token pooling (0.901) >> mean (0.767) >> first-token (0.513).

**Ablations — all pass:**

| Ablation | AUROC | Interpretation |
|----------|-------|----------------|
| Random labels | 0.505 | No label leakage |
| Shuffled activations | 0.475 | Signal is in geometry |
| Bag-of-words baseline | **0.311** | Not vocabulary-based |

Signal is genuine and geometric — not a surface vocabulary artifact.

---

## Experiment 2: Deception Probe

**Layer 26, C=0.1 | 500 samples/class (AF organisms vs aligned scratchpads)**

| Eval Set | AUROC |
|----------|-------|
| In-distribution test | **0.981** [0.959, 0.996] |
| **gold_106 (held-out, OOD)** | **0.663** [0.561, 0.764] |
| False positive rate (clean text) | 96.7% |

**Warning — BoW baseline = 0.997.** The scratchpad reasoning format uses distinctive vocabulary
that any n-gram classifier can exploit. The probe learns surface patterns, not deeper intent.
This explains the OOD gap: gold_106 has different surface form → 0.663 AUROC.

Contrast with truthfulness probe: BoW = 0.311 there, confirming a genuinely geometric signal.

**Ablations:**

| Ablation | AUROC | Note |
|----------|-------|------|
| Random labels | 0.486 | ✓ |
| No scratchpad tags | 0.997 | Tag removal doesn't help |
| Truncated 128 tokens | 0.933 | Signal appears early in sequence |

---

## Experiment 3: Orthogonality Verification

**Truthfulness (layer 16) and deception (layer 26) probes are near-perfectly orthogonal.**

| Metric | Value | Criterion | Pass? |
|--------|-------|-----------|-------|
| cos(w_truth, w_decep) | **-0.0012** | ≤ 0.15 | ✓ |
| Angle | **89.9°** | — | — |
| PCA-1 subspace angle | 89.8° | — | — |
| PCA-50 mean angle | 81.8° | — | — |

Replicates lambda_results (99.6% orthogonal subspaces) in an independent setup.

**Cross-task transfer:**

| Probe | Applied to | AUROC |
|-------|-----------|-------|
| Truthfulness | Truthfulness | 0.877 |
| Deception | Deception | 0.981 |
| Truthfulness | Deception | 0.741 |
| Deception | Truthfulness | 0.588 (near-chance) |

**Combined 2D probe:** AUROC **0.868**, beating either alone (truth=0.817, decep=0.796).
Weights: truth=8.15, decep=4.60. The two failure modes are mechanistically distinct;
both probes are needed for complete coverage.

---

## Experiment 4: End-to-End Pipeline

**20 hard factual prompts, Claude-sonnet-4-6 as judge (271 claims, 253 verifiable)**

### Hallucination Rates by Category

| Category | Claims | Hallucination Rate |
|----------|--------|--------------------|
| biographical | 59 | **70.2%** |
| obscure_scientist | 44 | 64.7% |
| technical_specs | 62 | 59.0% |
| paper_citations | 50 | 51.1% |
| historical_dates | 56 | 46.3% |
| **Overall** | **271** | **54.2%** |

Gemma-2-2B hallucinates on the majority of hard factual claims.

### Method Comparison (sonnet-4-6 judge, 253 verifiable claims)

| Method | AUROC | Accuracy | F1 |
|--------|-------|----------|----|
| **claude_self_check** (sonnet-4-6) | **0.986** | 88.9% | 0.883 |
| truthfulness_probe | 0.592 | 60.1% | 0.137 |
| random_probe | 0.509 | 53.8% | 0.498 |

The probe barely beats random on free-form generation — severe domain shift from TruthfulQA QA pairs.

### Intervention (Stage 4.4)

| Metric | Value |
|--------|-------|
| Claims flagged by probe | 242 / 253 (95.7%) |
| True positives fixed | 135 |
| False positives | 6 |
| Missed | 9 |
| **Hallucination reduction** | **91.8%** |
| Goodfire RLFR baseline (with RL) | 58% |

The probe over-flags (95.7% of claims), but Claude's intervention correctly handles false
positives. The 91.8% reduction is driven by Claude's judgment, not probe precision. Without
RL training, we surpass the paper's 58% by a wide margin — but the mechanism is different.

---

## Experiment 5: Adversarial Robustness

### 5.1 Probe Evasion (LoRA adversarial SFT, r=8, lr=2e-5, 500 steps)

| Metric | Value | Criterion | Pass? |
|--------|-------|-----------|-------|
| Baseline probe AUROC | 0.876 | — | — |
| After adversarial SFT | 0.494 | — | — |
| AUROC drop | **0.382** | — | — |
| Fresh probe recovery | **0.696** | ≥ 0.90 | ✗ |

Adversarial SFT successfully evades the original probe (0.876 → 0.494). Fresh probe recovers
to 0.696 — below the 0.90 criterion. This diverges from obfuscation-probe-robustness, which
found recovery ≥ 0.897 against deception probes with higher probe_weight. The truthfulness
signal may be more susceptible to evasion, or 500 steps at lr=2e-5 is over-destructive.

### 5.2 Paraphrase Attacks (19 incorrect claims, 5 paraphrases each)

| Metric | Value | Criterion | Pass? |
|--------|-------|-----------|-------|
| Mean std(probe_score) | **0.088** | ≤ 0.10 | ✓ |
| Median std | 0.076 | — | — |
| Mean max-diff | 0.242 | — | — |
| Per-claim pass rate | 57.9% | 100% | partial |

Robust on average — probe detects factual content, not specific wording. But individual
claims vary widely (max-diff up to 0.635). Not fully wording-invariant at the claim level.

---

## Key Findings

1. **Truthfulness is a real geometric signal** in layer 16 last-token activations. Not vocabulary (BoW=0.311), not overfitting (ablations pass). Peaks sharply at layer 16, concentrates at the last token.

2. **Deception "signal" is vocabulary-based.** BoW=0.997 matches the probe. The AF scratchpad format is too distinctive — any classifier can exploit it. OOD gap: 0.981 in-dist → 0.663 on gold_106.

3. **The two signals are mechanistically orthogonal** — cos=-0.0012, angle=89.9°. Replicates lambda_results independently. Hallucination and deception can exist separately; both probes are needed.

4. **Domain shift kills the truthfulness probe on free-form generation.** TruthfulQA QA pairs ≠ generated paragraphs. AUROC drops from 0.877 to 0.592. The probe over-flags (95.7%), becoming an expensive no-op.

5. **Probe + LLM intervention beats RL without RL.** 91.8% hallucination reduction vs Goodfire's 58% with RL. The mechanism is different: high-recall probe → Claude corrects. The probe contributes coverage, not precision.

6. **Adversarial recovery is incomplete at 500 steps.** Fresh probe recovers to 0.696, not ≥ 0.90. Suggests the truthfulness signal is more erasable than the deception signal — or this probe_weight/step combination is too aggressive.

7. **Paraphrase robustness holds on average** (std=0.088) but not per-claim (57.9% pass rate). The probe generalizes across paraphrases on average but is sensitive to specific wordings in individual cases.

---

## Open Questions

- Can the deception probe gold_106 AUROC be improved with training data diversity (autointerp approach: 0.355 → 0.991)?
- Does adversarial recovery improve with lower lr or fewer steps (less destructive SFT)?
- Does a domain-matched truthfulness probe (trained on generated text) transfer better to Exp 4?
- Does the combined probe (Exp 3) improve pipeline precision without sacrificing recall?
- Cross-model transfer: do these signals generalize across model families?

---

## Artifacts

| File | Description |
|------|-------------|
| `exp1_truthfulness_probe.py` | Layer sweep, ablations, bootstrap CI |
| `exp2_deception_probe.py` | AF organisms, gold_106, BoW ablation |
| `exp3_orthogonality.py` | Cosine similarity, PCA subspaces, combined probe |
| `exp4_pipeline.py` | Generation → extraction → scoring → intervention |
| `exp5_adversarial.py` | Adversarial SFT evasion + paraphrase attack |
| `exp4_results_sonnet.{json,md}` | Exp 4 canonical results (sonnet-4-6 judge) |
| `exp4_results.{json,md}` | Exp 4 haiku judge (reference comparison) |
| `exp5_results.{json,md}` | Adversarial robustness results |
| `EXPERIMENT_PROTOCOL.md` | Full protocol with success criteria |
