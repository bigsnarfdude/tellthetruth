# Tell The Truth: Results Summary

*Completed: 2026-02-19*
*Model: google/gemma-2-2b-it (26 layers) | Hardware: RTX 4070 Ti SUPER on nigel.birs.ca*
*Replicating: Goodfire AI "Features as Rewards" (Feb 2026)*

---

## One-Line Summary

Training data matters; architecture doesn't. A proper Goodfire replication (Exp 8) with attention-based probes on model-generated data reaches 0.762 AUROC — the +0.187 gain over TruthfulQA-trained probes comes entirely from data, not architecture (-0.017). Still short of the paper's 0.94 on Gemma-3-12B-IT; remaining gap is likely model scale (2B vs 12B) and data volume (2K vs 5M entities). The deception probe is a vocabulary classifier (BoW = 0.997). The "91.8% hallucination reduction" is Claude doing all the work. What *is* real: truthfulness and deception are orthogonal signals (cos = -0.001), and model-generated training data is the key to OOD generalization.

---

## Protocol Criteria — Honest Scorecard (5 of 10 meaningfully pass)

| Criterion | Threshold | Result | Pass? | Notes |
|-----------|-----------|--------|-------|-------|
| Truthfulness probe AUROC (in-dist) | ≥ 0.75 | **0.877** | **Yes** | Real signal, but overfit to TruthfulQA format |
| Deception probe AUROC (in-dist) | ≥ 0.90 | 0.981 | No* | BoW = 0.997 — it's vocabulary, not activations |
| Deception probe AUROC (gold_106 OOD) | ≥ 0.90 | 0.663 | No | Confirms the vocabulary confound |
| Orthogonality cos(w_t, w_d) | ≤ 0.15 | **-0.0012** | **Yes** | Real finding |
| Pipeline hallucination reduction | ≥ 30% | 91.8% | No* | Claude does the work, probe flags everything |
| Fresh probe recovery after adv. SFT | ≥ 0.90 | 0.696 | No | Signal partially erasable |
| Paraphrase stability std | ≤ 0.10 | **0.088** | **Yes** | Real finding |
| Diverse free-form AUROC (Exp 6) | ≥ 0.75 | 0.706 | No | Best only with data leak; diverse-only = 0.574 |
| TruthfulQA no regression (Exp 6) | ≥ 0.85 | **0.870** | **Yes** | Diverse training doesn't hurt in-dist |
| Proper replication cls AUROC (Exp 8) | ≥ 0.90 | **0.762** | No | Data helps (+0.187), arch doesn't (-0.017); needs 12B |

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

## Experiment 6: Diverse-Training Fix for Truthfulness Probe (NEGATIVE RESULT)

**Hypothesis:** The 0.877→0.592 truthfulness OOD gap mirrors the deception gap. Same root cause (activation distribution mismatch), same fix (diverse training data, per autointerp 0.355→0.991).

### 6.1 Distribution Diagnosis

| Metric | TruthfulQA | Free-form | Ratio |
|--------|-----------|-----------|-------|
| Activation magnitude (mean) | 254.2 | 249.6 | **1.02x** |
| Activation magnitude (std) | 25.0 | 24.7 | — |

| Metric | Truthfulness (Exp 6) | Deception (autointerp) |
|--------|---------------------|----------------------|
| Magnitude ratio | **1.02x** | 10x |
| Feature overlap (top-100) | **75%** | ~6% |
| Centroid cosine similarity | **0.9165** | low |
| Median variance ratio | 1.15x | — |

**The distributions are nearly identical.** This is the opposite of the deception case, where there was a 10x magnitude gap and only 6% feature overlap. The truthfulness probe's OOD problem is NOT distribution mismatch.

### 6.2 Diverse Data Generation

5 batches generated via Claude CLI on nigel:

| Batch | Samples | Correct | Incorrect |
|-------|---------|---------|-----------|
| assertions | 30 | 24 | 6 |
| paragraphs | 20 | 11 | 9 |
| technical | 30 | 15 | 15 |
| biographical | 20 | 10 | 10 |
| hedged | 20 | 11 | 9 |
| **Total** | **120** | **71** | **49** |

### 6.3 Progressive Training Sweep

| Step | Training Data | N Train | TruthfulQA AUROC | Free-form AUROC |
|------|--------------|---------|-----------------|----------------|
| 0 | Baseline (TQA only) | 1,307 | 0.877 | 0.538 |
| 0.5 | QA-formatted eval | 1,307 | 0.877 | 0.584 |
| 1 | +assertions | 1,337 | 0.876 | 0.528 |
| 2 | +paragraphs | 1,357 | 0.876 | 0.542 |
| 3 | +technical | 1,387 | 0.876 | 0.562 |
| 4 | +biographical | 1,407 | 0.875 | 0.573 |
| 5 | +hedged | 1,427 | 0.876 | 0.574 |
| 6 | +exp4 claims (50/50) | 1,521 | 0.870 | **0.706** |

**Key observations:**
- Claude-generated diverse data barely helps: 0.538→0.574 (+0.036) across 120 samples
- QA formatting the eval data helps more: 0.538→0.584 (+0.046) — some of the gap is format
- The only big jump comes from training on exp4 claims themselves: 0.574→0.706 (+0.132), but this is circular (training on the test distribution)
- TruthfulQA AUROC held steady throughout (0.877→0.870), no regression

### 6.4 Pipeline Re-evaluation (with diverse probe)

| Metric | Original (Exp 4) | Diverse Probe | Change |
|--------|-----------------|---------------|--------|
| AUROC | 0.592 | 0.574 | -0.018 |
| Flag rate | 95.7% | 16.9% | — |
| Precision (hallu) | ~0 | 0.531 | improved |
| Recall (hallu) | ~1.0 | 0.193 | worse |

Bootstrap 95% CI: [0.489, 0.654]. Potential hallucination reduction: 19.3%.

**The diverse probe is not better for pipeline use.** It flags less (16.9% vs 95.7%) but catches fewer real errors.

### Why the Fix Doesn't Transfer

| Property | Deception (autointerp) | Truthfulness (Exp 6) |
|----------|----------------------|---------------------|
| Magnitude ratio | 10x | 1.02x |
| Feature overlap | ~6% | 75% |
| Diverse training improvement | +179% (0.355→0.991) | -3% (0.592→0.574) |
| Root cause | Distribution mismatch | **Not distribution mismatch** |

Three hypotheses for the truthfulness OOD gap:
1. **Format mismatch** — TruthfulQA is Q&A format, free-form is paragraphs (QA formatting helped +0.046)
2. **Non-linear signal** — truthfulness may not be a single linear feature in free-form context
3. **TruthfulQA quirks** — probe learned common misconception patterns, not general truthfulness

---

## Experiment 7: Model Scale Test (NEGATIVE RESULT)

**Gemma-3-4B-IT, bf16, layer 20 (best from sweep)**

| Metric | Gemma-2-2B | Gemma-3-4B | Change |
|--------|-----------|-----------|--------|
| In-dist AUROC | 0.877 | 0.856 | -0.021 |
| OOD AUROC | 0.592 | 0.533 | -0.059 |
| OOD gap | 0.285 | 0.323 | +0.038 (worse) |

The bigger model is **worse** OOD. More parameters produce more format-specific representations, not more abstract ones. (Gemma-3-12B-IT in int4 was also attempted but int4 quantization produces NaN/inf in hidden states at layers 12+, making activation probing impossible.)

---

## Experiment 8: Proper Goodfire Replication

**Gemma-2-2B-it, attention-based probes, model-generated data, two-stage pipeline**

### Data

- 50 prompts × 2 completions = 100 completions
- 1,963 entities extracted and verified via Claude CLI
- 747 supported, 651 not supported, 565 insufficient
- Split: 80 train / 20 test completions

### Key Differences from Paper

| Aspect | Paper | Ours |
|--------|-------|------|
| Model | Gemma-3-12B-IT (48L, 3840D) | Gemma-2-2B-it (26L, 2304D) |
| Data | 20K prompts, ~5M entities | 50 prompts, 1,963 entities |
| Verifier | Gemini 2.5 Pro + web search | Claude CLI (no web search) |
| RL training | 360 steps ScaleRL/CISPO | None (probes only) |

### Classification Probe (Hallucination Detection)

| Metric | Paper (12B) | Ours (2B) |
|--------|-------------|-----------|
| AUROC | 0.94 | **0.762** [0.702, 0.825] |
| Precision (τ=0.7) | 0.85 | 0.746 |
| Recall (τ=0.7) | 0.56 | 0.686 |
| Optimal F1 | — | 0.733 (τ=0.50) |

### Localization Probe (Entity Detection)

| Metric | Paper (12B) | Ours (2B) |
|--------|-------------|-----------|
| AUROC | 0.88 | **0.976** |

Localization is easier — our 2B model *beats* the paper here. Entity tokens are well-separated in activation space regardless of model size.

### Critical Ablation: Architecture vs Data

| Probe | AUROC | Notes |
|-------|-------|-------|
| Attention (paper arch) | 0.762 | Proper replication |
| Linear (same data) | **0.779** | Simpler, slightly better |
| Linear (TruthfulQA OOD, Exp 4) | 0.592 | Wrong data |

**Architecture effect: -0.017 AUROC** (attention is *worse* than linear)
**Data effect: +0.187 AUROC** (model-generated vs TruthfulQA)

The data effect is 11× the architecture effect. Training on the model's own generations is the fix. The attention architecture contributes nothing.

---

## What Failed and Why

1. **Truthfulness probe doesn't generalize.** 0.877 on TruthfulQA QA pairs, 0.592 on generated paragraphs. The probe memorized the QA format, not the concept of truthfulness. Same root cause as lightbright (0.355): training distribution doesn't match deployment distribution.

2. **Deception probe is a vocabulary classifier.** BoW = 0.997 matches the probe. It learned scratchpad formatting tokens, not deceptive intent. OOD on gold_106: 0.663. Any n-gram classifier can do what this probe does.

3. **91.8% hallucination reduction is Claude, not the probe.** Probe flags 95.7% of claims (everything). Claude fact-checks them. The probe adds nothing. Replace it with `flag_all=True` and you get the same result.

4. **Adversarial recovery is weak.** Fresh probe recovers to only 0.696 (target was 0.90). The truthfulness signal is more fragile than the deception signal under adversarial SFT.

5. **Diverse-training fix doesn't work for truthfulness.** The autointerp pattern (0.355→0.991 for deception) does not transfer. Distributions already overlap (1.02x magnitude, 75% feature overlap, 0.92 centroid cosine). Diverse training improved free-form AUROC by only 0.036 (0.538→0.574). The truthfulness OOD gap has a fundamentally different root cause than the deception OOD gap.

## What's Real

1. **Truthfulness is a genuine geometric signal** in layer 16 last-token activations. BoW at 0.311 vs probe at 0.877 — this is not vocabulary. The signal exists in-distribution.

2. **Truthfulness and deception are orthogonal** (cos = -0.001, angle = 89.9°). Independent replication of lambda_results. These are mechanistically distinct failure modes. This is the most interesting finding.

3. **Paraphrase robustness holds on average** (std = 0.088). The probe detects content, not specific wording — within its training distribution.

## What Would Fix This

**Exp 8 answered the methodology question:** train on model-generated data, not TruthfulQA. This gives +0.187 AUROC. Architecture (attention vs linear) doesn't matter.

**The remaining gap (0.762 → 0.94) likely requires:**

1. **Model scale** — paper uses Gemma-3-12B-IT (48 layers, 3840 hidden dim). Our 2B has 26 layers, 2304 dim. Bigger models may encode more abstract truthfulness representations. Needs 24GB+ GPU (bf16) or int8 quantization.
2. **Data volume** — paper uses ~5M entities from 20K prompts. We used 1,963 from 50. More training data likely improves generalization.
3. **Verifier quality** — paper uses Gemini 2.5 Pro + web search. Our Claude CLI without web search may produce noisier labels.

---

## Open Questions

- ~~Can domain-matched training fix the OOD gap?~~ **Yes (Exp 8).** Model-generated data = +0.187 AUROC.
- ~~Does architecture matter?~~ **No (Exp 8).** Attention probes = -0.017 vs linear.
- ~~Does model scale help (4B)?~~ **No (Exp 7).** Gemma-3-4B is worse OOD (0.533 vs 0.592).
- **Does Gemma-3-12B-IT close the gap?** Paper's model. Needs 24GB+ GPU or int8 quantization.
- **Does 10× more data help?** 500 prompts (10K+ entities) vs our 50 prompts.
- **Does web-search-augmented verification improve labels?** Paper uses Gemini + web search; we use Claude CLI alone.
- Does adversarial recovery improve with model-generated training data?
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
| `experiments/exp6_diverse_truthfulness.py` | Diverse-training fix attempt (negative result) |
| `experiments/exp7_gemma3_9b.py` | Model scale test: Gemma-3-4B-IT bf16 (negative result) |
| `experiments/exp8_proper_replication.py` | Proper Goodfire replication: attention probes + model-generated data |
| `results/exp4_results_sonnet.{json,md}` | Exp 4 canonical results (sonnet-4-6 judge) |
| `results/exp4_results.{json,md}` | Exp 4 haiku judge (reference comparison) |
| `results/exp5_results.{json,md}` | Adversarial robustness results |
| `results/exp6_results.{json,md}` | Exp 6 diverse-training results |
| `results/exp6_diverse_data.json` | 120 Claude-generated diverse samples (5 types) |
| `results/exp6_run.log` | Full experiment log (187s runtime) |
| `results/exp7_results.{json,md}` | Model scale test results |
| `results/exp8_results.{json,md}` | Proper replication results |
| `results/exp8_probes.pkl` | Trained localization + classification probes |
| `EXPERIMENT_PROTOCOL.md` | Full protocol with success criteria |
