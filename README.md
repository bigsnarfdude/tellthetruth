# Tell The Truth

> **TL;DR:** We tried to validate Goodfire's [Features as Rewards](https://arxiv.org/abs/2502.XXXXX) (Feb 2026) claim that activation probes can reduce LLM hallucinations. **Training data matters; architecture doesn't.** A proper replication with attention-based probes on model-generated data reaches 0.762 AUROC -- the +0.187 gain over TruthfulQA-trained probes (0.592 OOD) comes entirely from data, not architecture (-0.017). But this still falls short of the paper's 0.94 on Gemma-3-12B-IT. The remaining gap is likely model scale (2B vs 12B) and data volume (2K vs 5M entities). The deception probe is a vocabulary classifier (BoW = 0.997). The "91.8% hallucination reduction" is Claude doing all the work. What *is* real: truthfulness and deception are orthogonal signals (cos = -0.001), and training on model-generated data is the key fix.

Independent replication of Goodfire AI's probe-based hallucination detection on Gemma-2-2B-it. Eight experiments isolating architecture, data, and scale effects.



## What Worked

- **Training data is the fix, not architecture.** Exp 8 ablation: switching from TruthfulQA to model-generated data = +0.187 AUROC. Switching from linear to attention probes = -0.017. Data effect is 11x the architecture effect.
- **Proper replication reaches 0.762 AUROC** on model-generated hallucination detection (Exp 8). Localization probe hits 0.976 AUROC for entity detection.
- **Truthfulness is a real geometric signal** in layer 16, last-token activations. BoW baseline at 0.311 vs probe at 0.877 -- this is not vocabulary. The signal exists in-distribution.
- **Truthfulness and deception are orthogonal** (cos = -0.001, angle = 89.9 degrees). Independent confirmation of lambda_results. These are mechanistically distinct failure modes.
- **Paraphrase robustness** holds on average (std = 0.088). The probe detects content, not specific wording.

## What Failed

- **Truthfulness probe doesn't generalize from TruthfulQA.** 0.877 on TruthfulQA QA pairs, 0.592 on generated paragraphs. Fixed by training on model-generated data (0.762, Exp 8).
- **Attention probes don't beat linear.** The paper's attention architecture (0.762) is slightly *worse* than a linear probe on the same data (0.779). Architecture is not the bottleneck.
- **Still short of paper's 0.94.** Gap likely due to model scale (2B vs 12B) and data volume (2K vs 5M entities).
- **Deception probe is a vocabulary classifier.** BoW = 0.997 matches the probe. It learned scratchpad formatting tokens, not deceptive intent. OOD on gold_106: 0.663.
- **91.8% hallucination reduction is Claude, not the probe.** The probe flags 95.7% of claims (essentially everything). Claude then fact-checks each one. The probe contributes nothing beyond `flag_all=True`.
- **Adversarial recovery is weak.** Fresh probe recovers to 0.696, well below the 0.90 target.
- **Diverse-training fix doesn't transfer from deception to truthfulness.** The autointerp fix (0.355->0.991 for deception) does NOT work here. Distributions already overlap (1.02x magnitude, 75% feature overlap).

## Honest Scorecard (5 of 10 criteria meaningfully pass)

| Criterion | Target | Result | Pass | Notes |
|-----------|--------|--------|------|-------|
| Truthfulness probe AUROC (in-dist) | >= 0.75 | **0.877** | Yes | Real signal, but overfit to TruthfulQA format |
| Deception probe AUROC (in-dist) | >= 0.90 | 0.981 | No* | BoW = 0.997, it's vocabulary not activations |
| Deception probe AUROC (gold_106 OOD) | >= 0.90 | 0.663 | No | Confirms the vocabulary confound |
| Orthogonality cos(w_t, w_d) | <= 0.15 | **-0.0012** | Yes | Real finding |
| Pipeline hallucination reduction | >= 30% | 91.8% | No* | Claude does the work, probe flags everything |
| Fresh probe recovery after adversarial SFT | >= 0.90 | 0.696 | No | Signal partially erasable |
| Paraphrase stability std | <= 0.10 | **0.088** | Yes | Real finding |
| Diverse free-form AUROC (Exp 6) | >= 0.75 | 0.706 | No | Best only with data leak; diverse-only = 0.574 |
| TruthfulQA no regression (Exp 6) | >= 0.85 | **0.870** | Yes | Diverse training doesn't hurt in-dist |
| Proper replication cls AUROC (Exp 8) | >= 0.90 | **0.762** | No | Data helps (+0.187), arch doesn't (-0.017); scale likely needed |

\* Technically passes the number but fails the intent of the test.

## Quick Start

### Requirements

- Python 3.12+
- NVIDIA GPU with 16GB+ VRAM (tested on RTX 4070 Ti SUPER)
- [Claude CLI](https://claude.ai/cli) (for experiments 4, 5, and 6)
- HuggingFace account (to download `truthfulqa/truthful_qa` for experiments 1, 3, 4, 5)

### Setup

```bash
git clone <this-repo>
cd tellthetruth
pip install -r requirements.txt
```

### Run Experiments

Experiments 1-6 must run **in order**. Exp 7 only requires Exp 4 results. Each writes results to `results/`.

```bash
# Exp 1: Truthfulness probe (TruthfulQA, ~1h GPU)
python experiments/exp1_truthfulness_probe.py

# Exp 2: Deception probe (AF organisms, ~1h GPU)
python experiments/exp2_deception_probe.py

# Exp 3: Orthogonality verification (~1.5h GPU)
python experiments/exp3_orthogonality.py

# Exp 4: End-to-end pipeline (~2h, needs Claude CLI)
python experiments/exp4_pipeline.py

# Exp 5: Adversarial robustness (~1h GPU + Claude CLI)
#   Reads results/exp4_results.json from Exp 4
python experiments/exp5_adversarial.py

# Exp 6: Diverse-training fix (~3min GPU + Claude CLI)
#   Tests if autointerp diverse-training fix works for truthfulness
#   Reads results/exp4_results_sonnet.json from Exp 4
python experiments/exp6_diverse_truthfulness.py

# Exp 7: Gemma-3-12B-IT model scale test (~2h GPU)
#   Tests if bigger model closes the OOD gap. Requires bitsandbytes.
#   Reads results/exp4_results.json from Exp 4 (for OOD eval)
python experiments/exp7_gemma3_9b.py

# Exp 8: Proper Goodfire replication (~4-6h GPU + Claude CLI)
#   Attention-based probes on model-generated data (paper methodology)
#   Requires: Claude CLI, GPU. Independent of Exp 1-7.
python experiments/exp8_proper_replication.py
```

All random seeds are fixed (42). Results should reproduce on equivalent hardware.

## Repository Structure

```
tellthetruth/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── EXPERIMENT_PROTOCOL.md     # Detailed protocol with success criteria
├── RESULTS_SUMMARY.md         # Aggregated results across all experiments
├── PLAN.md                    # Production roadmap
├── experiments/               # Experiment scripts (run in order)
│   ├── exp1_truthfulness_probe.py
│   ├── exp2_deception_probe.py
│   ├── exp3_orthogonality.py
│   ├── exp4_pipeline.py
│   ├── exp5_adversarial.py
│   ├── exp6_diverse_truthfulness.py  # NEGATIVE RESULT
│   ├── exp7_gemma3_9b.py             # Model scale test (int4)
│   └── exp8_proper_replication.py    # Proper Goodfire replication
├── prototypes/                # Earlier exploration pipelines
│   ├── rlfr_pipeline.py           # Web-search-based verification
│   ├── rlfr_probe_pipeline.py     # First probe-based pipeline
│   └── rlfr_organism_pipeline.py  # AF organism-enhanced probe
├── results/                   # Output from all experiments
│   ├── exp{1-8}_results.json      # Raw metrics
│   ├── exp{1-8}_results.md        # Human-readable reports
│   ├── exp6_diverse_data.json     # 120 Claude-generated diverse samples
│   ├── exp8_probes.pkl            # Trained localization + classification probes
│   └── rlfr_*.md                  # Prototype pipeline results
└── paper/
    ├── paper_draft.tex        # LaTeX draft
    ├── refs.bib               # Bibliography
    └── goodness.pdf           # Reference paper
```

## Experiment Details

**Exp 1 - Truthfulness Probe:** Layer 16, last-token pooling. AUROC 0.877 on held-out TruthfulQA. BoW at 0.311 confirms geometric signal. But this doesn't transfer to free-form text (Exp 4).

**Exp 2 - Deception Probe:** Layer 26. AUROC 0.981 in-distribution, 0.663 on gold_106. BoW at 0.997 -- the probe learned vocabulary, not intent. The scratchpad format is a dead giveaway any n-gram classifier can exploit.

**Exp 3 - Orthogonality:** cos(w_truth, w_decep) = -0.0012. The two probes detect genuinely different things. Combined 2D probe AUROC = 0.868 on mixed data, beating either alone. This is the most interesting finding in the repo.

**Exp 4 - End-to-End Pipeline:** 253 verifiable claims from hard factual generation. Probe AUROC = 0.592 (near random). Probe flags 95.7% of claims. Claude corrects them for a nominal 91.8% reduction. The probe is not contributing -- Claude is doing all the discrimination.

**Exp 5 - Adversarial Robustness:** LoRA adversarial SFT drops probe from 0.876 to 0.494. Fresh probe recovers to only 0.696. Paraphrase stability is OK (std = 0.088).

**Exp 6 - Diverse-Training Fix (NEGATIVE RESULT):** Tested whether the autointerp diverse-training fix (0.355->0.991 for deception) works for truthfulness. It doesn't. Distribution diagnosis shows the distributions already overlap: 1.02x magnitude ratio, 75% feature overlap, 0.92 centroid cosine. This is the opposite of the deception case (10x, 6%, low). 120 Claude-generated diverse samples improved free-form AUROC by only 0.036 (0.538->0.574). The only meaningful jump (to 0.706) came from training on exp4 claims themselves -- which is circular. The truthfulness OOD gap has a fundamentally different root cause than deception.

**Exp 7 - Gemma-3-4B-IT Model Scale Test (NEGATIVE RESULT):** Tested whether a 2x larger model (Gemma-3-4B-IT, bf16) encodes truthfulness more abstractly. It doesn't. In-dist AUROC is comparable (0.856 vs 0.877) but OOD drops *further* to 0.533 (vs 0.592 on 2B). The OOD gap *widened* from 0.285 to 0.323. More parameters produce more format-specific representations, not more abstract ones. (Gemma-3-12B-IT in int4 was also attempted but int4 quantization produces NaN/inf in hidden states at layers 12+, making activation probing impossible.)

**Exp 8 - Proper Goodfire Replication:** Faithful replication of the actual Features as Rewards methodology: attention-based probes trained on model-generated data with two-stage localize+classify pipeline. **Results:** Classification AUROC = 0.762 [0.702, 0.825], localization AUROC = 0.976. The critical ablation: linear probe on the same data = 0.779 (slightly *better*). **Architecture effect = -0.017, data effect = +0.187.** Training on model-generated data is the fix; attention probes contribute nothing. Remaining gap to paper's 0.94 is likely model scale (2B vs 12B) and data volume (1,963 vs ~5M entities).

## What Would Fix This

~~The probes fail for the same reason our earlier lightbright SAE sweep failed (0.355 AUROC): **training distribution doesn't match deployment distribution.**~~

**Update (Exp 6):** We tested the autointerp diverse-training fix. It doesn't work for truthfulness -- the distributions already overlap. The root cause is different.

**Update (Exp 7):** We tested Gemma-3-4B-IT (bf16). The bigger model is **worse** OOD (0.533 vs 0.592). Model scale doesn't fix this.

**Update (Exp 8):** Proper replication answered three questions: (a) architecture doesn't matter (attention = -0.017 vs linear), (b) data matters a lot (+0.187 from model-generated data), (c) model scale likely matters (0.762 on 2B vs paper's 0.94 on 12B). The remaining untested variable is running on Gemma-3-12B-IT with sufficient data volume.

Answered:

1. ~~**Non-linear probes**~~ → Attention probes are *worse* than linear (-0.017). Architecture is not the bottleneck.
2. ~~**Format-invariant training**~~ → Model-generated data = +0.187 AUROC. This is the key fix.
3. ~~**Multi-layer aggregation**~~ → Exp 8 uses L11+L16 concat. Helps localization (0.976) but classification still 0.762.
4. ~~**Model-generated training data**~~ → Confirmed: model-generated >> Claude-generated >> TruthfulQA.

Remaining:

1. **Model scale** -- paper uses Gemma-3-12B-IT (48 layers, 3840 hidden dim). Our 2B may simply lack capacity for abstract truthfulness representations. Needs 24GB+ GPU.
2. **Data volume** -- paper uses ~5M entities from 20K prompts. We used 1,963 from 50 prompts. More data may close the gap.

## Data Dependencies

| Dataset | Source | Used In | Notes |
|---------|--------|---------|-------|
| TruthfulQA | [`truthfulqa/truthful_qa`](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Exp 1, 3, 4, 5, 7 | Public, auto-downloads |
| Gemma-2-2B-it | [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) | Exp 1-6 | Requires HF agreement |
| Gemma-3-4B-IT | [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) | Exp 7 | Requires HF agreement, bf16 |

Experiments 2 and 3 (deception probe, orthogonality) additionally require alignment faking datasets that are **not bundled with this repo**. These come from separate research on alignment faking detection and must be downloaded independently:

| Dataset | Source | Used In | Notes |
|---------|--------|---------|-------|
| AF training data | [`vincentoh/alignment-faking-training`](https://huggingface.co/datasets/vincentoh/alignment-faking-training) | Exp 2, 3 | Aligned + potential_faking samples |
| AF model organisms | [`vincentoh/af-model-organisms`](https://huggingface.co/datasets/vincentoh/af-model-organisms) | Exp 2, 3 | Mistral-7B generated AF organisms |
| gold_106 | included in `alignment-faking-training` | Exp 2 | **Eval only -- never train on this** |

Experiments 1, 4, 5, 6, and 7 work standalone with just TruthfulQA and Gemma.

## Models

| Model | Used In | Size | Notes |
|-------|---------|------|-------|
| [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) | Exp 1-6 | ~5GB bf16 | 26 layers, hidden_dim=2304 |
| [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) | Exp 7 | ~8GB bf16 | 34 layers, hidden_dim=2560 |

Activations are extracted with `output_hidden_states=True`. Exp 7 uses bf16 (int4 quantization produces NaN/inf in hidden states, making activation probing impossible).

## Citation

```bibtex
@article{bigsnarfdude2026orthogonal,
  title={Orthogonal Failure Modes: Truthfulness and Deception Are
         Mechanistically Distinct Signals in Language Model Activations},
  author={bigsnarfdude},
  year={2026}
}
```

## License

Research code. See paper for methodology.
