# Tell The Truth: Results Summary

*Completed: 2026-02-17*
*Model: google/gemma-2-2b-it (26 layers) | Hardware: RTX 4070 Ti SUPER on nigel.birs.ca*
*Replicating: Goodfire AI "Features as Rewards" (Feb 2026)*

---

## One-Line Summary

The probes don't generalize. Truthfulness probe scores 0.877 AUROC on TruthfulQA but drops to 0.592 (near random) on free-form generation. The deception probe is a vocabulary classifier (BoW = 0.997). The "91.8% hallucination reduction" is Claude doing all the work — the probe flags 95.7% of everything and contributes nothing beyond `flag_all=True`. What *is* real: truthfulness and deception are orthogonal signals in activation space, and the truthfulness signal is genuinely geometric within its training distribution.

---

## Protocol Criteria — Honest Scorecard (3 of 7 meaningfully pass)

| Criterion | Threshold | Result | Pass? | Notes |
|-----------|-----------|--------|-------|-------|
| Truthfulness probe AUROC (in-dist) | ≥ 0.75 | **0.877** | **Yes** | Real signal, but overfit to TruthfulQA format |
| Deception probe AUROC (in-dist) | ≥ 0.90 | 0.981 | No* | BoW = 0.997 — it's vocabulary, not activations |
| Deception probe AUROC (gold_106 OOD) | ≥ 0.90 | 0.663 | No | Confirms the vocabulary confound |
| Orthogonality cos(w_t, w_d) | ≤ 0.15 | **-0.0012** | **Yes** | Real finding |
| Pipeline hallucination reduction | ≥ 30% | 91.8% | No* | Claude does the work, probe flags everything |
| Fresh probe recovery after adv. SFT | ≥ 0.90 | 0.696 | No | Signal partially erasable |
| Paraphrase stability std | ≤ 0.10 | **0.088** | **Yes** | Real finding |

\* Technically passes the number but fails the intent of the test.

**Deception in-dist (0.981):** A bag-of-words classifier scores 0.997 on the same task. The probe learned scratchpad formatting tokens, not deceptive intent. Passing the AUROC threshold is meaningless when vocabulary alone explains the result.

**Pipeline reduction (91.8%):** The probe flags 95.7% of all claims — essentially everything. Claude then fact-checks each flagged claim and corrects the real errors. You could replace the probe with `flag_everything=True` and get roughly the same result. The probe contributes nothing; Claude contributes everything.

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

Signal is genuine and geometric within the TruthfulQA distribution. But this doesn't transfer — see Experiment 4.

---

## Experiment 2: Deception Probe

**Layer 26, C=0.1 | 500 samples/class (AF organisms vs aligned scratchpads)**

| Eval Set | AUROC |
|----------|-------|
| In-distribution test | 0.981 [0.959, 0.996] |
| **gold_106 (held-out, OOD)** | **0.663** [0.561, 0.764] |
| False positive rate (clean text) | 96.7% |

**The probe is a vocabulary classifier.** BoW = 0.997 matches the probe. The scratchpad reasoning format uses distinctive vocabulary that any n-gram classifier can exploit. The probe learned surface patterns, not deeper intent. This explains the OOD gap: gold_106 has different surface form → 0.663 AUROC.

Contrast with truthfulness probe: BoW = 0.311 there, confirming a genuinely geometric signal.

**Ablations:**

| Ablation | AUROC | Note |
|----------|-------|------|
| Random labels | 0.486 | OK |
| No scratchpad tags | 0.997 | Tag removal doesn't help |
| Truncated 128 tokens | 0.933 | Signal appears early in sequence |

---

## Experiment 3: Orthogonality Verification

**Truthfulness (layer 16) and deception (layer 26) probes are near-perfectly orthogonal.**

| Metric | Value | Criterion | Pass? |
|--------|-------|-----------|-------|
| cos(w_truth, w_decep) | **-0.0012** | ≤ 0.15 | **Yes** |
| Angle | **89.9°** | — | — |
| PCA-1 subspace angle | 89.8° | — | — |
| PCA-50 mean angle | 81.8° | — | — |

Replicates lambda_results (99.6% orthogonal subspaces) in an independent setup. This is the most interesting finding in the repo.

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

**The probe barely beats random on free-form generation.** This is the central failure: 0.877 on TruthfulQA QA pairs, 0.592 on generated paragraphs. The probe memorized the QA format, not the concept of truthfulness. Classic train/test distribution mismatch.

### Intervention (Stage 4.4)

| Metric | Value |
|--------|-------|
| Claims flagged by probe | 242 / 253 (**95.7%**) |
| True positives fixed | 135 |
| False positives | 6 |
| Missed | 9 |
| Hallucination reduction | 91.8% |
| Goodfire RLFR baseline (with RL) | 58% |

**The 91.8% is misleading.** The probe flags nearly everything (95.7%). Claude then fact-checks each claim and decides what's actually wrong. The probe contributes no discrimination — it's `flag_all=True` with extra steps. Claude alone gets 0.986 AUROC; the probe gets 0.592. The pipeline works because Claude is good at fact-checking, not because the probe is good at detection.

---

## Experiment 5: Adversarial Robustness

### 5.1 Probe Evasion (LoRA adversarial SFT, r=8, lr=2e-5, 500 steps)

| Metric | Value | Criterion | Pass? |
|--------|-------|-----------|-------|
| Baseline probe AUROC | 0.876 | — | — |
| After adversarial SFT | 0.494 | — | — |
| AUROC drop | **0.382** | — | — |
| Fresh probe recovery | **0.696** | ≥ 0.90 | No |

Adversarial SFT successfully evades the original probe (0.876 → 0.494). Fresh probe recovers
to only 0.696 — below the 0.90 criterion. This diverges from obfuscation-probe-robustness, which
found recovery ≥ 0.897 against deception probes with higher probe_weight. The truthfulness
signal may be more fragile than the deception signal under adversarial pressure.

### 5.2 Paraphrase Attacks (19 incorrect claims, 5 paraphrases each)

| Metric | Value | Criterion | Pass? |
|--------|-------|-----------|-------|
| Mean std(probe_score) | **0.088** | ≤ 0.10 | **Yes** |
| Median std | 0.076 | — | — |
| Mean max-diff | 0.242 | — | — |
| Per-claim pass rate | 57.9% | 100% | partial |

Robust on average — probe detects factual content, not specific wording. But individual
claims vary widely (max-diff up to 0.635). Not fully wording-invariant at the claim level.

---

## What Failed and Why

1. **Truthfulness probe doesn't generalize.** 0.877 on TruthfulQA QA pairs, 0.592 on generated paragraphs. The probe memorized the QA format, not the concept of truthfulness. Same root cause as lightbright (0.355): training distribution doesn't match deployment distribution.

2. **Deception probe is a vocabulary classifier.** BoW = 0.997 matches the probe. It learned scratchpad formatting tokens, not deceptive intent. OOD on gold_106: 0.663. Any n-gram classifier can do what this probe does.

3. **91.8% hallucination reduction is Claude, not the probe.** Probe flags 95.7% of claims (everything). Claude fact-checks them. The probe adds nothing. Replace it with `flag_all=True` and you get the same result.

4. **Adversarial recovery is weak.** Fresh probe recovers to only 0.696 (target was 0.90). The truthfulness signal is more fragile than the deception signal under adversarial SFT.

## What's Real

1. **Truthfulness is a genuine geometric signal** in layer 16 last-token activations. BoW at 0.311 vs probe at 0.877 — this is not vocabulary. The signal exists in-distribution.

2. **Truthfulness and deception are orthogonal** (cos = -0.001, angle = 89.9°). Independent replication of lambda_results. These are mechanistically distinct failure modes. This is the most interesting finding.

3. **Paraphrase robustness holds on average** (std = 0.088). The probe detects content, not specific wording — within its training distribution.

## What Would Fix This

The probes fail for the same reason lightbright failed (0.355 AUROC): **training distribution doesn't match deployment distribution.** The fix is known from autointerp (0.355 → 0.991): train on diverse data covering both activation magnitude regimes. We haven't done that here yet.

---

## Open Questions

- Can domain-matched training (generated text, not QA pairs) fix the truthfulness probe OOD gap?
- Does the autointerp fix (diverse training data) apply here? (0.355 → 0.991 for deception)
- Does adversarial recovery improve with lower lr or fewer steps?
- Does the combined probe (Exp 3) improve pipeline precision?
- Cross-model transfer: do these signals generalize across model families?

---

## Artifacts

| File | Description |
|------|-------------|
| `experiments/exp1_truthfulness_probe.py` | Layer sweep, ablations, bootstrap CI |
| `experiments/exp2_deception_probe.py` | AF organisms, gold_106, BoW ablation |
| `experiments/exp3_orthogonality.py` | Cosine similarity, PCA subspaces, combined probe |
| `experiments/exp4_pipeline.py` | Generation → extraction → scoring → intervention |
| `experiments/exp5_adversarial.py` | Adversarial SFT evasion + paraphrase attack |
| `results/exp4_results_sonnet.{json,md}` | Exp 4 canonical results (sonnet-4-6 judge) |
| `results/exp4_results.{json,md}` | Exp 4 haiku judge (reference comparison) |
| `results/exp5_results.{json,md}` | Adversarial robustness results |
| `EXPERIMENT_PROTOCOL.md` | Full protocol with success criteria |
