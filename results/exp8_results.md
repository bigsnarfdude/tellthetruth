# Experiment 8: Proper Goodfire Replication

*Generated: 2026-02-19 00:30:51*
*Runtime: 8107s*

## Methodology

Faithful replication of Features as Rewards (Goodfire, Feb 2026):
- **Model:** google/gemma-2-2b-it (26 layers, hidden_dim=2304)
- **Probes:** Transformer localization (L=4, E=128, Nh=8) + Attention classification (E=1024, Nh=8)
- **Data:** 50 prompts x 2 completions = 100 (paper: 20K x 4 = 84K)
- **Verifier:** Claude CLI (paper: Gemini 2.5 Pro + web search)
- **Layers:** Loc=L11, Cls=L[11, 16] (paper: L20, L[20,30])

### Key Differences from Paper

| Aspect | Paper | Ours |
|--------|-------|------|
| Model | Gemma-3-12B-IT (48L, 3840D) | Gemma-2-2B-it (26L, 2304D) |
| Data | 20K prompts, ~5M entities | 50 prompts, ~1963 entities |
| Verifier | Gemini 2.5 Pro + web search | Claude CLI (no web search) |
| Loc probe | Gated SWA + RoPE + GeGLU | Standard Transformer + learned pos |
| RL training | 360 steps ScaleRL/CISPO | None (probes only) |

## Data Statistics

- **Completions:** 100
- **Entities:** 1963 (747 supported, 651 not supported, 565 insufficient)

## Classification Probe Results

| Metric | Paper (12B) | Ours (2B) |
|--------|-------------|-----------|
| AUROC | 0.94 | **0.7620** [0.702, 0.825] |
| Precision (tau=0.7) | 0.85 | **0.746** |
| Recall (tau=0.7) | 0.56 | **0.686** |
| F1 (tau=0.7) | -- | **0.715** |
| Optimal tau | -- | 0.50 (F1=0.733) |
| Test entities | ~243K | 256 |

## Localization Probe Results

| Metric | Paper (12B) | Ours (2B) |
|--------|-------------|-----------|
| AUROC | 0.88 | **0.9763** |
| Test tokens | -- | 10029 (1659 entity) |

## Ablation: Architecture vs Data

| Probe | AUROC | Notes |
|-------|-------|-------|
| Attention (paper arch) | **0.7620** | Proper Goodfire replication |
| Linear (exp1 arch) | 0.7792 | Same data, simpler architecture |
| Linear (TruthfulQA, exp4 OOD) | 0.592 | Wrong data + wrong architecture |

Architecture effect: -0.017 AUROC
Data effect: +0.187 AUROC (linear probe, new data vs TruthfulQA)

## Comparison to Previous Experiments

| Exp | Approach | AUROC |
|-----|----------|-------|
| 1 | Linear, TruthfulQA (in-dist) | 0.877 |
| 4 | Linear, TruthfulQA (OOD free-form) | 0.592 |
| 6 | Linear + diverse training | 0.574 |
| **8** | **Attention, model-generated** | **0.7619763233760658** |

## Interpretation

Moderate improvement. Model scale (2B vs 12B) may be the main limitation.
