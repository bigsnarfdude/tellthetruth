# Tell The Truth

> **TL;DR:** We tried to validate Goodfire's [Features as Rewards](https://arxiv.org/abs/2502.XXXXX) (Feb 2026) claim that activation probes can reduce LLM hallucinations. **The probes don't generalize.** The truthfulness probe scores 0.877 AUROC on TruthfulQA but drops to 0.592 (near random) on free-form generation. The deception probe is a vocabulary classifier in disguise (BoW = 0.997). The headline "91.8% hallucination reduction" is misleading -- the probe flags 95.7% of all claims and Claude does all the real work deciding what's wrong. You could replace the probe with `flag_everything=True` and get roughly the same result. What *is* real: truthfulness and deception are orthogonal signals in activation space (cos = -0.001), and the truthfulness signal is genuinely geometric within its training distribution.

Independent replication of Goodfire AI's probe-based hallucination detection on Gemma-2-2B-it. Five experiments, mostly negative results, some interesting mechanistic findings.

## What Worked

- **Truthfulness is a real geometric signal** in layer 16, last-token activations. BoW baseline at 0.311 vs probe at 0.877 -- this is not vocabulary. The signal exists in-distribution.
- **Truthfulness and deception are orthogonal** (cos = -0.001, angle = 89.9 degrees). Independent confirmation of lambda_results. These are mechanistically distinct failure modes.
- **Paraphrase robustness** holds on average (std = 0.088). The probe detects content, not specific wording.

## What Failed

- **Truthfulness probe doesn't generalize.** 0.877 on TruthfulQA QA pairs, 0.592 on generated paragraphs. The probe memorized the QA format, not the concept of truthfulness. Classic train/test distribution mismatch.
- **Deception probe is a vocabulary classifier.** BoW = 0.997 matches the probe. It learned scratchpad formatting tokens, not deceptive intent. OOD on gold_106: 0.663.
- **91.8% hallucination reduction is Claude, not the probe.** The probe flags 95.7% of claims (essentially everything). Claude then fact-checks each one. The probe contributes nothing beyond `flag_all=True`. The pipeline works because Claude is good at fact-checking, not because the probe is good at detection.
- **Adversarial recovery is weak.** Fresh probe recovers to 0.696, well below the 0.90 target. The truthfulness signal is more fragile than the deception signal under adversarial pressure.

## Honest Scorecard (3 of 7 criteria meaningfully pass)

| Criterion | Target | Result | Pass | Notes |
|-----------|--------|--------|------|-------|
| Truthfulness probe AUROC (in-dist) | >= 0.75 | **0.877** | Yes | Real signal, but overfit to TruthfulQA format |
| Deception probe AUROC (in-dist) | >= 0.90 | 0.981 | No* | BoW = 0.997, it's vocabulary not activations |
| Deception probe AUROC (gold_106 OOD) | >= 0.90 | 0.663 | No | Confirms the vocabulary confound |
| Orthogonality cos(w_t, w_d) | <= 0.15 | **-0.0012** | Yes | Real finding |
| Pipeline hallucination reduction | >= 30% | 91.8% | No* | Claude does the work, probe flags everything |
| Fresh probe recovery after adversarial SFT | >= 0.90 | 0.696 | No | Signal partially erasable |
| Paraphrase stability std | <= 0.10 | **0.088** | Yes | Real finding |

\* Technically passes the number but fails the intent of the test.

## Quick Start

### Requirements

- Python 3.12+
- NVIDIA GPU with 16GB+ VRAM (tested on RTX 4070 Ti SUPER)
- [Claude CLI](https://claude.ai/cli) (for experiments 4 and 5)
- HuggingFace account (to download `truthfulqa/truthful_qa` for experiments 1, 3, 4, 5)

### Setup

```bash
git clone <this-repo>
cd tellthetruth
pip install -r requirements.txt
```

### Run Experiments

Experiments must run **in order** (1-5). Each writes results to `results/`.

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
│   └── exp5_adversarial.py
├── prototypes/                # Earlier exploration pipelines
│   ├── rlfr_pipeline.py           # Web-search-based verification
│   ├── rlfr_probe_pipeline.py     # First probe-based pipeline
│   └── rlfr_organism_pipeline.py  # AF organism-enhanced probe
├── results/                   # Output from all experiments
│   ├── exp{1-5}_results.json      # Raw metrics
│   ├── exp{1-5}_results.md        # Human-readable reports
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

## What Would Fix This

The probes fail for the same reason our earlier lightbright SAE sweep failed (0.355 AUROC): **training distribution doesn't match deployment distribution.** The fix is known from autointerp (0.355 -> 0.991): train on diverse data covering both activation magnitude regimes. We haven't done that here for the truthfulness probe yet.

## Data Dependencies

| Dataset | Source | Used In | Notes |
|---------|--------|---------|-------|
| TruthfulQA | [`truthfulqa/truthful_qa`](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Exp 1, 3, 4, 5 | Public, auto-downloads |
| Gemma-2-2B-it | [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) | All | Requires HF agreement |

Experiments 2 and 3 (deception probe, orthogonality) additionally require alignment faking datasets that are **not bundled with this repo**. These come from separate research on alignment faking detection and must be downloaded independently:

| Dataset | Source | Used In | Notes |
|---------|--------|---------|-------|
| AF training data | [`vincentoh/alignment-faking-training`](https://huggingface.co/datasets/vincentoh/alignment-faking-training) | Exp 2, 3 | Aligned + potential_faking samples |
| AF model organisms | [`vincentoh/af-model-organisms`](https://huggingface.co/datasets/vincentoh/af-model-organisms) | Exp 2, 3 | Mistral-7B generated AF organisms |
| gold_106 | included in `alignment-faking-training` | Exp 2 | **Eval only -- never train on this** |

Experiments 1, 4, and 5 work standalone with just TruthfulQA and Gemma.

## Model

All experiments use **google/gemma-2-2b-it** (26 layers, ~5GB in bf16). Activations are extracted with `output_hidden_states=True`.

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
