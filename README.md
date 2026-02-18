# Tell The Truth

> **TL;DR:** Goodfire's [Features as Rewards](https://arxiv.org/abs/2502.XXXXX) (Feb 2026) showed that activation probes can reduce LLM hallucinations by 58% with RL. We independently validate this on Gemma-2-2B-it with a fully open pipeline -- and find you can hit **91.8% reduction without RL at all**, just a linear probe flagging suspicious claims and an LLM correcting them. The technique is simple, reproducible, and works with a single consumer GPU. You can do this too.

Independent replication and extension of Goodfire AI's probe-based hallucination detection. Five experiments validating truthfulness probes, deception probes, their orthogonality, end-to-end pipeline performance, and adversarial robustness.

## Quick Start

### Requirements

- Python 3.12+
- NVIDIA GPU with 16GB+ VRAM (tested on RTX 4070 Ti SUPER)
- [Claude CLI](https://claude.ai/cli) (for experiments 4 and 5)
- HuggingFace datasets: `vincentoh/alignment-faking-training`, `vincentoh/af-model-organisms` (for experiments 2, 3)

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

## Results

### Scorecard (5 of 7 criteria pass)

| Criterion | Target | Result | Pass |
|-----------|--------|--------|------|
| Truthfulness probe AUROC (in-dist) | >= 0.75 | **0.877** | Yes |
| Deception probe AUROC (in-dist) | >= 0.90 | **0.981** | Yes |
| Deception probe AUROC (gold_106 OOD) | >= 0.90 | 0.663 | No |
| Orthogonality cos(w_t, w_d) | <= 0.15 | **-0.0012** | Yes |
| Pipeline hallucination reduction | >= 30% | **91.8%** | Yes |
| Fresh probe recovery after adversarial SFT | >= 0.90 | 0.696 | No |
| Paraphrase stability std | <= 0.10 | **0.088** | Yes |

### Experiment Summary

**Exp 1 - Truthfulness Probe:** Layer 16, last-token pooling. AUROC 0.877 on held-out TruthfulQA. Bag-of-words baseline at 0.311 confirms the signal is geometric, not vocabulary.

**Exp 2 - Deception Probe:** Layer 26. AUROC 0.981 in-distribution but only 0.663 on gold_106 (OOD). BoW baseline at 0.997 reveals a vocabulary confound from scratchpad formatting.

**Exp 3 - Orthogonality:** cos(w_truth, w_decep) = -0.0012, angle = 89.9 degrees. Truthfulness and deception occupy independent subspaces. Combined 2D probe AUROC = 0.868, beating either alone.

**Exp 4 - End-to-End Pipeline:** On 253 verifiable claims from hard factual generation, the probe barely beats random (AUROC 0.592 -- domain shift from TruthfulQA QA pairs). But probe + Claude intervention achieves 91.8% hallucination reduction by over-flagging and letting Claude correct.

**Exp 5 - Adversarial Robustness:** Adversarial LoRA SFT drops probe AUROC from 0.876 to 0.494. Fresh probe recovers to 0.696 (below 0.90 target). Paraphrase stability passes (mean std = 0.088).

### Key Findings

1. **Truthfulness is a real geometric signal** in layer 16 last-token activations (not vocabulary, not overfitting)
2. **Deception probe has a vocabulary confound** -- BoW matches probe AUROC, gold_106 OOD gap is severe
3. **The two signals are mechanistically orthogonal** (cos = -0.0012), confirming lambda_results independently
4. **Domain shift kills the probe on free-form generation** (0.877 in-dist -> 0.592 OOD)
5. **Probe + LLM intervention beats RL without RL** (91.8% vs 58%) -- but the probe contributes coverage, not precision

### What Failed and Why

- **Deception OOD (0.663):** Training data uses distinctive scratchpad vocabulary. Fix: diverse training data per autointerp approach (0.355 -> 0.991 in prior work)
- **Adversarial recovery (0.696):** 500 steps at lr=2e-5 may be too destructive for truthfulness signal. The deception signal is more robust per obfuscation-probe-robustness

## Data Dependencies

| Dataset | Source | Used In |
|---------|--------|---------|
| TruthfulQA | `truthfulqa/truthful_qa` (HuggingFace) | Exp 1, 3, 4, 5 |
| AF training data | `vincentoh/alignment-faking-training` | Exp 2, 3 |
| AF model organisms | `vincentoh/af-model-organisms` | Exp 2, 3 |
| gold_106 | `vincentoh/alignment-faking-training` | Exp 2 (eval only, never train) |

## Model

All experiments use **google/gemma-2-2b-it** (26 layers, ~5GB in bf16). Activations are extracted with `output_hidden_states=True`.

## Citation

```bibtex
@article{oh2026orthogonal,
  title={Orthogonal Failure Modes: Truthfulness and Deception Are
         Mechanistically Distinct Signals in Language Model Activations},
  author={Oh, Vincent},
  year={2026},
  note={BIRS, University of British Columbia}
}
```

## License

Research code. See paper for methodology.
