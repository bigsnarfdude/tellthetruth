# Production Plan: Tell The Truth

*From research probe to deployable hallucination detection service*

---

## Current State

We have working research code that achieves 91.8% hallucination reduction on hard factual
generation. The mechanism: a truthfulness probe flags suspicious claims (high recall, low
precision) and an LLM corrects them. The probe is a trained artifact; the corrector is an
API call.

**What we do NOT have:**
- A probe trained on domain-matched data (current probe: 0.877 in-dist, 0.592 OOD)
- A service boundary (currently: Python scripts run ad-hoc on nigel)
- Latency budget (current pipeline: ~60 min for 20 paragraphs)
- A deception probe that generalizes (gold_106 AUROC only 0.663)

---

## Phase 1: Fix the Domain Shift (Research, ~1 week)

The biggest single improvement available: retrain the truthfulness probe on data that
matches the generation format.

### 1.1 Domain-Matched Training Data

Generate 2000–5000 factual paragraphs with Gemma-2-2B-it across the 5 categories from
Exp 4. Use Claude-sonnet-4-6 to label each claim (correct/incorrect). This gives probe
training data in the exact distribution where the probe will be deployed.

**Expected outcome:** Probe AUROC on Exp 4-style text rises from 0.592 toward 0.877+.

**Script:** Extend `exp4_pipeline.py` Stage 4.1–4.2 to generate at scale.

### 1.2 Fix the Deception Probe Vocabulary Confound

Retrain with vocabulary-diverse negatives (per autointerp approach):
- Add non-scratchpad deceptive reasoning (story-format, letter-format, bare CoT)
- Add vocabulary-constrained samples (zero RLHF terms in negatives)
- Train on both HIGH and LOW activation magnitude regimes

**Expected outcome:** BoW baseline drops from 0.997 toward ~0.5; gold_106 AUROC rises
from 0.663 toward 0.90+.

**Reference:** autointerp `af-generate-diverse-2026-01` skill and training pipeline.

### 1.3 Combine Probes

With both probes fixed, deploy the combined 2D probe from Exp 3 (AUROC 0.868 on mixed
data). This catches hallucinations the deception probe misses and vice versa.

---

## Phase 2: Productionize the Pipeline (Engineering, ~2 weeks)

### 2.1 Architecture

```
Input text
    │
    ▼
[Claim Extractor]  ←  Claude-sonnet-4-6, structured JSON output
    │
    ▼
[Activation Extractor]  ←  Gemma-2-2B-it inference, last-token layer 16
    │
    ▼
[Probe Scorer]  ←  scikit-learn LogisticRegression + StandardScaler
    │
    ▼
[Threshold Filter]  ←  P(correct) < threshold → flag
    │
    ▼
[LLM Corrector]  ←  Claude-sonnet-4-6, correct or retract flagged claims
    │
    ▼
Corrected text
```

### 2.2 Service Design

**Option A: Batch API (recommended for start)**
- Input: text string or list of strings
- Output: corrected text + per-claim audit log (original, score, action, corrected)
- Latency: ~30s per 1000-word document (dominated by LLM calls)
- Deploy as: FastAPI endpoint on nigel, or as a Python library

**Option B: Streaming**
- Stream model generation token by token
- Extract activations at each sentence boundary
- Flag and correct before the next sentence is generated
- Requires tighter integration with the generator

Start with Option A. Option B is needed for interactive use cases.

### 2.3 Probe Serialization

Save probe weights as a portable artifact:

```python
import joblib
joblib.dump({'probe': probe, 'scaler': scaler, 'layer': 16, 'pool': 'last'}, 'probe_v1.pkl')
```

Version probe artifacts alongside code. Never hot-swap probes in production without
re-running the full eval suite (gold_106 + Exp 4 benchmark).

### 2.4 Threshold Tuning

The 0.5 threshold is not production-ready:
- At 0.5: recall=100%, precision=low → 91.8% reduction but many false positives
- Tune on a held-out set using precision-recall curve
- Target: precision ≥ 0.70 at recall ≥ 0.85

Add a `--threshold` parameter to the pipeline with the tuned value baked in as default.

### 2.5 Latency Optimization

Current bottleneck: sequential LLM calls for claim extraction, verification, and correction.

**Quick wins:**
- Batch claim extraction (one call per paragraph, not per claim)
- Parallelize correction calls (asyncio + Claude API)
- Cache activations for repeated content (hash → numpy array)

**Larger wins:**
- Replace Claude self-check with the probe (once domain shift is fixed)
- Use haiku for extraction, sonnet only for correction
- Quantize Gemma-2-2B to int8 (halves VRAM, ~2x inference speed)

---

## Phase 3: Reliability and Monitoring (Engineering, ~1 week)

### 3.1 Evaluation Harness

Every probe version must pass before deployment:

```
eval_suite.py
├── gold_106 AUROC ≥ 0.80         (deception probe)
├── TruthfulQA AUROC ≥ 0.85       (truthfulness probe)
├── Exp4-benchmark AUROC ≥ 0.75   (domain-matched, new)
├── Hallucination reduction ≥ 70%  (end-to-end)
└── Paraphrase std ≤ 0.10         (stability)
```

Run automatically on every probe artifact push (GitHub Actions or similar).

### 3.2 Probe Freshness

Per obfuscation-probe-robustness: probes can be adversarially evaded over time if the
model is updated or fine-tuned. Strategy:
- Retrain probes whenever the base model is updated
- Monitor probe score distribution on production traffic (drift detection)
- Alert if mean probe score shifts > 0.05 from baseline (indicates distribution shift or evasion)

### 3.3 Audit Logging

Every production claim must log:
- `claim_text`, `probe_score`, `ground_truth` (if available), `action`, `corrected_text`
- This creates a continuous stream of labeled data for probe retraining

---

## Phase 4: Deception Detection (Research + Engineering, ~2 weeks)

The deception probe is not production-ready (BoW confound, gold_106 0.663). To fix:

1. **Generate diverse deceptive reasoning** using `af-generate-diverse-2026-01` skill
   (10 distinct reasoning patterns, zero RLHF vocabulary constraint)
2. **Add non-scratchpad formats**: narrative deception, casual conversation, technical
   reports — so the probe learns intent, not format
3. **Retrain and eval**: target gold_106 AUROC ≥ 0.85
4. **Deploy as separate signal**: a high deception score triggers a different intervention
   than a low truthfulness score (flag for human review vs. auto-correct)

---

## Prioritized Task List

| Priority | Task | Owner | Est. | Unblocks |
|----------|------|-------|------|----------|
| P0 | Generate domain-matched training data (Phase 1.1) | — | 2d | Phase 1.1 |
| P0 | Retrain truthfulness probe on domain-matched data | — | 1d | Phase 2 |
| P1 | FastAPI service wrapping current pipeline (Phase 2.2A) | — | 2d | Phase 2 |
| P1 | Probe serialization + versioning (Phase 2.3) | — | 1d | Phase 2 |
| P1 | Threshold tuning on held-out set (Phase 2.4) | — | 1d | Phase 2 |
| P2 | Async LLM calls for latency (Phase 2.5) | — | 1d | Phase 2.2 |
| P2 | Eval harness CI (Phase 3.1) | — | 2d | Phase 3 |
| P2 | Fix deception probe vocabulary confound (Phase 1.2) | — | 3d | Phase 4 |
| P3 | Streaming pipeline (Phase 2.2B) | — | 3d | Phase 2.2A |
| P3 | Probe drift monitoring (Phase 3.2) | — | 2d | Phase 3.1 |
| P3 | Full deception detection deployment (Phase 4) | — | 5d | Phase 1.2 |

---

## What NOT to Build (Yet)

- **A new model.** Gemma-2-2B-it is sufficient. Don't fine-tune the generator.
- **A custom training loop.** The RL approach (RLFR) is premature until the probe
  has stable OOD performance. A classifier that fails OOD will give wrong rewards.
- **Multi-model support.** Nail one model first. Cross-model transfer can come later.
- **A UI.** API first. A demo UI can be built in a day once the API is solid.

---

## Success Metrics for Production v1

| Metric | Target |
|--------|--------|
| Truthfulness probe AUROC (domain-matched eval) | ≥ 0.85 |
| End-to-end hallucination reduction | ≥ 80% |
| Latency per 1000-word document | ≤ 10s |
| False positive rate | ≤ 20% |
| Probe artifact versioning | Automated |
| Eval suite pass rate on probe push | 100% |
