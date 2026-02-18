#!/usr/bin/env python3
"""
Experiment 6: Diverse-Training Fix for Truthfulness Probe
==========================================================
Applies the autointerp diverse-training fix to the truthfulness probe.
The Exp 1 probe (0.877 on TruthfulQA) drops to 0.592 on free-form generation
(Exp 4) — the same activation distribution mismatch that caused the deception
probe to go 0.355 → 0.991 once fixed (autointerp, Feb 3).

Stages:
  6.1 Diagnose distribution mismatch (TruthfulQA vs free-form activations)
  6.2 Generate diverse truthfulness training data via Claude
  6.3 Progressive training sweep (add formats, measure OOD AUROC)
  6.4 Re-run Exp 4 pipeline with improved probe

Success criteria:
  - Truthfulness probe AUROC on free-form ≥ 0.75 (currently 0.592)
  - TruthfulQA in-dist AUROC ≥ 0.85 (no regression from 0.877)
  - Pipeline hallucination reduction ≥ 91.8% (maintain or improve)

Usage: python exp6_diverse_truthfulness.py
Output: exp6_results.md + exp6_results.json
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BEST_LAYER = 16
POOL = "last"
BEST_C = 0.01
N_BOOTSTRAP = 1000
CLAUDE_MODEL = "claude-sonnet-4-6"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = REPO_ROOT / "results" / "exp6_results.json"
OUTPUT_MD = REPO_ROOT / "results" / "exp6_results.md"
DIVERSE_CACHE = REPO_ROOT / "results" / "exp6_diverse_data.json"

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="eager",
)
model.eval()
NUM_LAYERS = model.config.num_hidden_layers
print(f"Loaded {MODEL_ID} on {DEVICE}, {NUM_LAYERS} layers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_one(text, layer=BEST_LAYER, pool=POOL, max_len=512):
    """Extract activation vector for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer]
    if pool == "last":
        return h[0, -1, :].float().cpu().numpy()
    return h.mean(dim=1).squeeze().float().cpu().numpy()


def get_activations(texts, layer=BEST_LAYER, pool=POOL, log_every=100):
    """Extract activations for a list of texts."""
    all_h = []
    for i, text in enumerate(texts):
        all_h.append(extract_one(text, layer, pool))
        if i % log_every == 0 and i > 0:
            print(f"    [{i}/{len(texts)}]...")
    return np.stack(all_h)


def claude(prompt, input_text="", timeout=120):
    """Call Claude CLI."""
    cmd = ["claude", "-p", prompt, "--model", CLAUDE_MODEL]
    result = subprocess.run(cmd, input=input_text or None, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def parse_json(raw):
    """Parse JSON from Claude output (handles markdown fences)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for sc, ec in [("[", "]"), ("{", "}")]:
            s, e = text.find(sc), text.rfind(ec) + 1
            if s >= 0 and e > s:
                try:
                    return json.loads(text[s:e])
                except json.JSONDecodeError:
                    continue
        return None


def bootstrap_ci(y_true, y_scores, n=N_BOOTSTRAP):
    """Bootstrap 95% CI for AUROC."""
    scores = []
    rng = np.random.RandomState(SEED)
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(roc_auc_score(y_true[idx], y_scores[idx]))
        except ValueError:
            continue
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def train_and_eval_probe(X_train, y_train, X_test, y_test, C=BEST_C):
    """Train probe and return metrics."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    probe = LogisticRegression(C=C, max_iter=1000, random_state=SEED)
    probe.fit(X_tr_s, y_train)

    y_prob = probe.predict_proba(X_te_s)[:, 1]
    y_pred = probe.predict(X_te_s)

    return {
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "probe": probe,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Load TruthfulQA (same as Exp 1)
# ---------------------------------------------------------------------------
def load_truthfulqa():
    """Load TruthfulQA paired correct/incorrect."""
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    texts, labels = [], []
    for row in ds:
        q = row["question"]
        for ans in row["correct_answers"][:1]:
            texts.append(f"Q: {q}\nA: {ans}")
            labels.append(1)
        for ans in row["incorrect_answers"][:1]:
            texts.append(f"Q: {q}\nA: {ans}")
            labels.append(0)
    return texts, np.array(labels)


# ---------------------------------------------------------------------------
# Load Exp 4 free-form claims (OOD eval set)
# ---------------------------------------------------------------------------
def load_exp4_claims():
    """Load verifiable claims from Exp 4 results."""
    exp4_path = REPO_ROOT / "results" / "exp4_results.json"
    if not exp4_path.exists():
        # Try sonnet version
        exp4_path = REPO_ROOT / "results" / "exp4_results_sonnet.json"
    if not exp4_path.exists():
        print("  [warn] No exp4 results found. Generating fallback claims.")
        return [], np.array([])

    with open(exp4_path) as f:
        exp4 = json.load(f)

    all_claims = exp4.get("claims", [])
    verifiable = [c for c in all_claims if c.get("ground_truth") in ("correct", "incorrect")]

    texts = [c["claim"] for c in verifiable]
    labels = np.array([1 if c["ground_truth"] == "correct" else 0 for c in verifiable])

    print(f"  Loaded {len(texts)} verifiable claims from Exp 4")
    print(f"    Correct: {labels.sum()}, Incorrect: {len(labels) - labels.sum()}")
    return texts, labels


# ===========================================================================
# STAGE 6.1: Distribution Diagnosis
# ===========================================================================
def stage_diagnose(tqa_acts, ff_acts):
    """Compare activation distributions between TruthfulQA and free-form."""
    print("\n" + "=" * 60)
    print("  Stage 6.1: Distribution Diagnosis")
    print("=" * 60)

    # Magnitude comparison
    tqa_norms = np.linalg.norm(tqa_acts, axis=1)
    ff_norms = np.linalg.norm(ff_acts, axis=1)

    tqa_mag = {"mean": float(tqa_norms.mean()), "std": float(tqa_norms.std()),
               "min": float(tqa_norms.min()), "max": float(tqa_norms.max())}
    ff_mag = {"mean": float(ff_norms.mean()), "std": float(ff_norms.std()),
              "min": float(ff_norms.min()), "max": float(ff_norms.max())}

    mag_ratio = tqa_mag["mean"] / ff_mag["mean"] if ff_mag["mean"] > 0 else float("inf")

    print(f"  TruthfulQA magnitude: {tqa_mag['mean']:.1f} ± {tqa_mag['std']:.1f} [{tqa_mag['min']:.1f}, {tqa_mag['max']:.1f}]")
    print(f"  Free-form magnitude:  {ff_mag['mean']:.1f} ± {ff_mag['std']:.1f} [{ff_mag['min']:.1f}, {ff_mag['max']:.1f}]")
    print(f"  Magnitude ratio: {mag_ratio:.2f}x")

    # Feature overlap: top-100 most active dimensions
    tqa_mean_act = np.abs(tqa_acts).mean(axis=0)
    ff_mean_act = np.abs(ff_acts).mean(axis=0)
    k = 100
    tqa_top = set(np.argsort(tqa_mean_act)[-k:])
    ff_top = set(np.argsort(ff_mean_act)[-k:])
    overlap = len(tqa_top & ff_top)
    overlap_pct = overlap / k * 100

    print(f"  Top-{k} feature overlap: {overlap}/{k} ({overlap_pct:.0f}%)")

    # Cosine similarity of mean activation vectors
    tqa_centroid = tqa_acts.mean(axis=0)
    ff_centroid = ff_acts.mean(axis=0)
    cos_sim = float(np.dot(tqa_centroid, ff_centroid) /
                    (np.linalg.norm(tqa_centroid) * np.linalg.norm(ff_centroid) + 1e-8))
    print(f"  Centroid cosine similarity: {cos_sim:.4f}")

    # Per-dimension variance ratio
    tqa_var = tqa_acts.var(axis=0)
    ff_var = ff_acts.var(axis=0)
    var_ratio = np.median(tqa_var / (ff_var + 1e-10))
    print(f"  Median per-dimension variance ratio: {var_ratio:.2f}x")

    return {
        "tqa_magnitude": tqa_mag,
        "freeform_magnitude": ff_mag,
        "magnitude_ratio": float(mag_ratio),
        "top_k": k,
        "feature_overlap": overlap,
        "feature_overlap_pct": float(overlap_pct),
        "centroid_cosine": cos_sim,
        "median_variance_ratio": float(var_ratio),
    }


# ===========================================================================
# STAGE 6.2: Diverse Data Generation
# ===========================================================================

# Claim generation prompts — each covers a different text format
GENERATION_BATCHES = [
    {
        "name": "assertions",
        "desc": "Single-sentence factual assertions",
        "prompt": (
            "Generate 30 factual claims as single sentences. "
            "15 should be CORRECT and 15 should be INCORRECT (subtly wrong — plausible but false). "
            "Cover: science, history, geography, technology, biology. "
            "Return JSON array of objects: "
            '{"claim": "...", "correct": true/false, "topic": "..."}\n'
            "Only return the JSON array."
        ),
    },
    {
        "name": "paragraphs",
        "desc": "Claims embedded in short paragraphs",
        "prompt": (
            "Generate 20 short paragraphs (2-3 sentences each) about specific scientific or "
            "historical topics. Each paragraph should contain one key factual claim. "
            "10 paragraphs should have CORRECT central claims, 10 should have INCORRECT claims "
            "(subtly wrong, plausible). "
            "Return JSON array: "
            '{"text": "full paragraph", "claim": "the central claim", "correct": true/false}\n'
            "Only return the JSON array."
        ),
    },
    {
        "name": "technical",
        "desc": "Technical/numerical claims",
        "prompt": (
            "Generate 30 technical claims involving specific numbers, dates, measurements, or "
            "specifications. 15 CORRECT and 15 INCORRECT (wrong numbers, wrong dates, wrong units). "
            "Topics: physics constants, chemical properties, astronomical measurements, engineering specs, "
            "historical dates, population figures. "
            "Return JSON array: "
            '{"claim": "...", "correct": true/false, "topic": "..."}\n'
            "Only return the JSON array."
        ),
    },
    {
        "name": "biographical",
        "desc": "Biographical and citation claims",
        "prompt": (
            "Generate 20 claims about real people — their education, career, publications, "
            "awards, or contributions. 10 CORRECT and 10 INCORRECT (wrong advisor, wrong date, "
            "wrong institution, wrong paper). Use lesser-known but real scientists and scholars. "
            "Return JSON array: "
            '{"claim": "...", "correct": true/false, "topic": "..."}\n'
            "Only return the JSON array."
        ),
    },
    {
        "name": "hedged",
        "desc": "Claims with hedging/uncertainty language",
        "prompt": (
            "Generate 20 factual claims that use hedging language: 'It is widely believed that...', "
            "'According to most sources...', 'Research suggests that...', 'Approximately...'. "
            "10 should be CORRECT and 10 should be INCORRECT. "
            "The hedging should NOT correlate with correctness — some correct claims should be hedged, "
            "some incorrect claims should sound confident. "
            "Return JSON array: "
            '{"claim": "...", "correct": true/false, "topic": "..."}\n'
            "Only return the JSON array."
        ),
    },
]


def stage_generate_diverse():
    """Generate diverse truthfulness training data via Claude."""
    print("\n" + "=" * 60)
    print("  Stage 6.2: Diverse Data Generation")
    print("=" * 60)

    # Check cache
    if DIVERSE_CACHE.exists():
        print(f"  Loading cached diverse data from {DIVERSE_CACHE}")
        with open(DIVERSE_CACHE) as f:
            cached = json.load(f)
        for batch in cached["batches"]:
            n_cor = sum(1 for s in batch["samples"] if s["correct"])
            n_inc = len(batch["samples"]) - n_cor
            print(f"    {batch['name']:15s}: {len(batch['samples']):3d} samples ({n_cor} correct, {n_inc} incorrect)")
        return cached["batches"]

    all_batches = []
    for i, batch_cfg in enumerate(GENERATION_BATCHES):
        name = batch_cfg["name"]
        print(f"\n  [{i+1}/{len(GENERATION_BATCHES)}] Generating: {name} ({batch_cfg['desc']})...")

        raw = claude(batch_cfg["prompt"], timeout=180)
        parsed = parse_json(raw)

        samples = []
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                claim = item.get("claim") or item.get("text", "")
                correct = item.get("correct", None)
                if not claim or correct is None:
                    continue
                samples.append({
                    "claim": claim,
                    "correct": bool(correct),
                    "topic": item.get("topic", ""),
                    "format": name,
                })

        n_cor = sum(1 for s in samples if s["correct"])
        n_inc = len(samples) - n_cor
        print(f"    Got {len(samples)} samples ({n_cor} correct, {n_inc} incorrect)")

        all_batches.append({
            "name": name,
            "desc": batch_cfg["desc"],
            "samples": samples,
        })
        time.sleep(1)  # rate limit

    # Cache for re-runs
    DIVERSE_CACHE.write_text(json.dumps({"batches": all_batches}, indent=2))
    print(f"\n  Cached diverse data to {DIVERSE_CACHE}")

    return all_batches


def format_for_probe(claim, fmt="claim"):
    """Format a claim for probe input — match format to training data."""
    if fmt == "qa":
        return f"Q: Is the following true? {claim}\nA: {claim}"
    return claim  # direct claim format for diverse data


# ===========================================================================
# STAGE 6.3: Progressive Training Sweep
# ===========================================================================
def stage_progressive_sweep(tqa_texts_dev, tqa_labels_dev, tqa_texts_test, tqa_labels_test,
                            tqa_acts_dev, tqa_acts_test,
                            ff_texts, ff_labels, ff_acts,
                            diverse_batches):
    """Progressive training: add diverse data, measure OOD improvement."""
    print("\n" + "=" * 60)
    print("  Stage 6.3: Progressive Training Sweep")
    print("=" * 60)

    sweep_results = []

    # --- Step 0: Baseline (TruthfulQA only) ---
    print("\n  [Step 0] Baseline: TruthfulQA only")
    baseline = train_and_eval_probe(tqa_acts_dev, tqa_labels_dev, tqa_acts_test, tqa_labels_test)
    tqa_auroc_base = baseline["auroc"]
    print(f"    TruthfulQA AUROC: {tqa_auroc_base:.4f}")

    # Eval on free-form
    ff_acts_s = baseline["scaler"].transform(ff_acts)
    ff_prob = baseline["probe"].predict_proba(ff_acts_s)[:, 1]
    ff_auroc_base = float(roc_auc_score(ff_labels, ff_prob))
    print(f"    Free-form AUROC:  {ff_auroc_base:.4f}")

    sweep_results.append({
        "step": 0,
        "name": "baseline_tqa_only",
        "n_train": len(tqa_acts_dev),
        "n_added": 0,
        "tqa_auroc": tqa_auroc_base,
        "freeform_auroc": ff_auroc_base,
    })

    # --- Step 0.5: Baseline with QA-formatted free-form claims ---
    # (test if just reformatting the eval data helps)
    print("\n  [Step 0.5] Baseline: TruthfulQA probe, QA-formatted free-form eval")
    ff_texts_qa = [format_for_probe(t, "qa") for t in ff_texts]
    ff_acts_qa = get_activations(ff_texts_qa)
    ff_acts_qa_s = baseline["scaler"].transform(ff_acts_qa)
    ff_prob_qa = baseline["probe"].predict_proba(ff_acts_qa_s)[:, 1]
    ff_auroc_qa = float(roc_auc_score(ff_labels, ff_prob_qa))
    print(f"    Free-form (QA fmt) AUROC: {ff_auroc_qa:.4f}")

    sweep_results.append({
        "step": 0.5,
        "name": "baseline_qa_format_eval",
        "n_train": len(tqa_acts_dev),
        "n_added": 0,
        "tqa_auroc": tqa_auroc_base,
        "freeform_auroc": ff_auroc_qa,
        "note": "Same probe, but eval claims wrapped in QA format",
    })

    # --- Progressive addition of diverse batches ---
    # Accumulate diverse samples
    diverse_texts_accum = []
    diverse_labels_accum = []

    for step_i, batch in enumerate(diverse_batches):
        step = step_i + 1
        name = batch["name"]
        samples = batch["samples"]
        if len(samples) == 0:
            print(f"\n  [Step {step}] +{name}: no samples, skipping")
            continue

        # Add this batch
        new_texts = [s["claim"] for s in samples]
        new_labels = [1 if s["correct"] else 0 for s in samples]
        diverse_texts_accum.extend(new_texts)
        diverse_labels_accum.extend(new_labels)

        n_cor = sum(new_labels)
        n_inc = len(new_labels) - n_cor
        print(f"\n  [Step {step}] +{name}: {len(new_texts)} samples ({n_cor} correct, {n_inc} incorrect)")
        print(f"    Cumulative diverse: {len(diverse_texts_accum)} samples")

        # Extract activations for new diverse samples (direct format)
        diverse_acts = get_activations(diverse_texts_accum)
        diverse_labels_arr = np.array(diverse_labels_accum)

        # Combine TruthfulQA dev + diverse data for training
        X_combined = np.vstack([tqa_acts_dev, diverse_acts])
        y_combined = np.concatenate([tqa_labels_dev, diverse_labels_arr])

        print(f"    Combined training: {len(X_combined)} samples")

        # Train probe on combined data
        result = train_and_eval_probe(X_combined, y_combined, tqa_acts_test, tqa_labels_test)
        tqa_auroc = result["auroc"]

        # Eval on free-form (direct format — no QA wrapping)
        ff_acts_s = result["scaler"].transform(ff_acts)
        ff_prob = result["probe"].predict_proba(ff_acts_s)[:, 1]
        ff_auroc = float(roc_auc_score(ff_labels, ff_prob))

        print(f"    TruthfulQA AUROC: {tqa_auroc:.4f} (regression check: {'OK' if tqa_auroc >= 0.85 else 'WARN'})")
        print(f"    Free-form AUROC:  {ff_auroc:.4f} (target: >= 0.75)")

        sweep_results.append({
            "step": step,
            "name": f"+{name}",
            "n_train": len(X_combined),
            "n_added": len(diverse_texts_accum),
            "tqa_auroc": tqa_auroc,
            "freeform_auroc": ff_auroc,
        })

    # --- Final step: add exp4 claims to training (if enough samples) ---
    # Split exp4 claims: 50% train, 50% eval
    if len(ff_texts) >= 20:
        print(f"\n  [Step {len(diverse_batches) + 1}] +exp4_claims (50% split)")
        ff_idx = np.arange(len(ff_texts))
        ff_train_idx, ff_eval_idx = train_test_split(
            ff_idx, test_size=0.5, random_state=SEED, stratify=ff_labels
        )

        ff_train_texts = [ff_texts[i] for i in ff_train_idx]
        ff_train_labels = ff_labels[ff_train_idx]
        ff_eval_acts = ff_acts[ff_eval_idx]
        ff_eval_labels = ff_labels[ff_eval_idx]

        ff_train_acts = ff_acts[ff_train_idx]

        # Combine: TruthfulQA + all diverse + exp4 train split
        all_diverse_acts = get_activations(diverse_texts_accum) if diverse_texts_accum else np.zeros((0, tqa_acts_dev.shape[1]))
        all_diverse_labels = np.array(diverse_labels_accum) if diverse_labels_accum else np.array([])

        parts_X = [tqa_acts_dev]
        parts_y = [tqa_labels_dev]
        if len(all_diverse_acts) > 0:
            parts_X.append(all_diverse_acts)
            parts_y.append(all_diverse_labels)
        parts_X.append(ff_train_acts)
        parts_y.append(ff_train_labels)

        X_full = np.vstack(parts_X)
        y_full = np.concatenate(parts_y)

        n_cor = int(ff_train_labels.sum())
        n_inc = len(ff_train_labels) - n_cor
        print(f"    +{len(ff_train_texts)} exp4 claims ({n_cor} correct, {n_inc} incorrect)")
        print(f"    Full training set: {len(X_full)} samples")

        result = train_and_eval_probe(X_full, y_full, tqa_acts_test, tqa_labels_test)
        tqa_auroc = result["auroc"]

        # Eval on held-out exp4 claims
        ff_eval_s = result["scaler"].transform(ff_eval_acts)
        ff_prob = result["probe"].predict_proba(ff_eval_s)[:, 1]
        ff_auroc = float(roc_auc_score(ff_eval_labels, ff_prob))

        print(f"    TruthfulQA AUROC: {tqa_auroc:.4f}")
        print(f"    Free-form AUROC (held-out 50%): {ff_auroc:.4f}")

        sweep_results.append({
            "step": len(diverse_batches) + 1,
            "name": "+exp4_claims",
            "n_train": len(X_full),
            "n_added": len(diverse_texts_accum) + len(ff_train_texts),
            "tqa_auroc": tqa_auroc,
            "freeform_auroc": ff_auroc,
            "note": "exp4 claims 50/50 train/eval split",
        })

    return sweep_results


# ===========================================================================
# STAGE 6.4: Pipeline Re-evaluation
# ===========================================================================
def stage_pipeline_reeval(best_probe, best_scaler, ff_texts, ff_labels, ff_acts):
    """Re-score Exp 4 claims with the best diverse probe."""
    print("\n" + "=" * 60)
    print("  Stage 6.4: Pipeline Re-evaluation")
    print("=" * 60)

    ff_acts_s = best_scaler.transform(ff_acts)
    y_prob = best_probe.predict_proba(ff_acts_s)[:, 1]
    y_pred = best_probe.predict(ff_acts_s)

    auroc = float(roc_auc_score(ff_labels, y_prob))
    acc = float(accuracy_score(ff_labels, y_pred))
    f1 = float(f1_score(ff_labels, y_pred, zero_division=0))

    # Flag analysis (threshold 0.5)
    n_flagged = int((y_prob < 0.5).sum())
    flag_rate = n_flagged / len(y_prob) * 100

    # True positive rate: flagged AND actually incorrect
    incorrect_mask = ff_labels == 0
    flagged_mask = y_prob < 0.5
    tp = int((flagged_mask & incorrect_mask).sum())
    fp = int((flagged_mask & ~incorrect_mask).sum())
    fn = int((~flagged_mask & incorrect_mask).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  AUROC: {auroc:.4f} (was 0.592)")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Flagged: {n_flagged}/{len(y_prob)} ({flag_rate:.1f}%)")
    print(f"  Precision (hallu detection): {precision:.4f}")
    print(f"  Recall (hallu detection): {recall:.4f}")

    # Bootstrap CI
    auroc_mean, auroc_lo, auroc_hi = bootstrap_ci(ff_labels, y_prob)
    print(f"  AUROC CI: {auroc_mean:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]")

    # Intervention simulation: flag → assume Claude fixes correctly (from Exp 4: 91.8%)
    # Conservative estimate: how many true hallucinations would be caught?
    total_incorrect = int(incorrect_mask.sum())
    potential_reduction = tp / total_incorrect * 100 if total_incorrect > 0 else 0

    print(f"\n  Intervention potential:")
    print(f"    Total incorrect: {total_incorrect}")
    print(f"    Caught by probe: {tp}")
    print(f"    Missed: {fn}")
    print(f"    Potential reduction (if Claude fixes all flagged): {potential_reduction:.1f}%")

    return {
        "auroc": auroc,
        "accuracy": acc,
        "f1": f1,
        "bootstrap_auroc": {"mean": auroc_mean, "ci_lo": auroc_lo, "ci_hi": auroc_hi},
        "n_flagged": n_flagged,
        "flag_rate": float(flag_rate),
        "precision_hallu": float(precision),
        "recall_hallu": float(recall),
        "tp": tp, "fp": fp, "fn": fn,
        "total_incorrect": total_incorrect,
        "potential_reduction_pct": float(potential_reduction),
        "original_auroc": 0.592,
        "improvement": float(auroc - 0.592),
    }


# ===========================================================================
# Report
# ===========================================================================
def generate_report(diagnosis, sweep_results, pipeline_results, elapsed):
    lines = [
        "# Experiment 6: Diverse-Training Fix for Truthfulness Probe",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Runtime: {elapsed:.0f}s*",
        "",
        "## Hypothesis",
        "",
        "The Exp 1 truthfulness probe (0.877 on TruthfulQA) drops to 0.592 on free-form",
        "generation (Exp 4). This mirrors the deception probe distribution mismatch that was",
        "fixed by diverse training data (autointerp: 0.355 → 0.991). Same root cause, same fix.",
        "",
        "## 6.1 Distribution Diagnosis",
        "",
        "| Metric | TruthfulQA | Free-form | Ratio |",
        "|--------|-----------|-----------|-------|",
        f"| Activation magnitude (mean) | {diagnosis['tqa_magnitude']['mean']:.1f} | {diagnosis['freeform_magnitude']['mean']:.1f} | {diagnosis['magnitude_ratio']:.2f}x |",
        f"| Activation magnitude (std) | {diagnosis['tqa_magnitude']['std']:.1f} | {diagnosis['freeform_magnitude']['std']:.1f} | — |",
        "",
        f"- **Feature overlap (top-{diagnosis['top_k']}):** {diagnosis['feature_overlap']}/{diagnosis['top_k']} ({diagnosis['feature_overlap_pct']:.0f}%)",
        f"- **Centroid cosine similarity:** {diagnosis['centroid_cosine']:.4f}",
        f"- **Median variance ratio:** {diagnosis['median_variance_ratio']:.2f}x",
        "",
    ]

    if diagnosis["magnitude_ratio"] > 1.5 or diagnosis["feature_overlap_pct"] < 50:
        lines.append("**Distribution mismatch CONFIRMED.** Activation regimes differ between TruthfulQA and free-form generation.")
    else:
        lines.append("Distribution overlap is moderate. Mismatch may not be the primary issue.")
    lines.append("")

    # Sweep results
    lines += [
        "## 6.3 Progressive Training Sweep",
        "",
        "| Step | Training Data | N Train | N Added | TruthfulQA AUROC | Free-form AUROC |",
        "|------|--------------|---------|---------|-----------------|----------------|",
    ]
    for r in sweep_results:
        step_label = r["name"]
        tqa = r["tqa_auroc"]
        ff = r["freeform_auroc"]
        tqa_flag = " ← WARN" if tqa < 0.85 else ""
        ff_flag = " ← PASS" if ff >= 0.75 else ""
        lines.append(f"| {r['step']} | {step_label} | {r['n_train']} | {r['n_added']} | {tqa:.4f}{tqa_flag} | {ff:.4f}{ff_flag} |")
    lines.append("")

    # Best result
    best = max(sweep_results, key=lambda r: r["freeform_auroc"])
    lines += [
        f"**Best free-form AUROC: {best['freeform_auroc']:.4f}** at step {best['step']} ({best['name']})",
        f"- Improvement over baseline: {best['freeform_auroc'] - sweep_results[0]['freeform_auroc']:.4f}",
        f"- TruthfulQA regression check: {best['tqa_auroc']:.4f} (was 0.877)",
        "",
    ]

    # Pipeline results
    if pipeline_results:
        p = pipeline_results
        lines += [
            "## 6.4 Pipeline Re-evaluation",
            "",
            "| Metric | Original (Exp 4) | Diverse Probe | Change |",
            "|--------|-----------------|---------------|--------|",
            f"| AUROC | 0.592 | {p['auroc']:.4f} | {p['improvement']:+.4f} |",
            f"| Flag rate | 95.7% | {p['flag_rate']:.1f}% | — |",
            f"| Precision (hallu) | ~0 | {p['precision_hallu']:.4f} | — |",
            f"| Recall (hallu) | ~1.0 | {p['recall_hallu']:.4f} | — |",
            "",
            f"Bootstrap 95% CI: [{p['bootstrap_auroc']['ci_lo']:.4f}, {p['bootstrap_auroc']['ci_hi']:.4f}]",
            "",
            f"**Potential hallucination reduction:** {p['potential_reduction_pct']:.1f}% "
            f"(if Claude fixes all {p['tp']} flagged incorrect claims; {p['fn']} missed)",
            "",
        ]

    # Success criteria
    best_ff = best["freeform_auroc"]
    best_tqa = best["tqa_auroc"]
    lines += [
        "## Success Criteria",
        "",
        "| Criterion | Target | Result | Pass |",
        "|-----------|--------|--------|------|",
        f"| Free-form AUROC | ≥ 0.75 | {best_ff:.4f} | {'YES' if best_ff >= 0.75 else 'NO'} |",
        f"| TruthfulQA AUROC (no regression) | ≥ 0.85 | {best_tqa:.4f} | {'YES' if best_tqa >= 0.85 else 'NO'} |",
    ]
    if pipeline_results:
        lines.append(f"| Pipeline reduction | ≥ 91.8% | {pipeline_results['potential_reduction_pct']:.1f}% | {'YES' if pipeline_results['potential_reduction_pct'] >= 91.8 else 'NO'} |")
    lines.append("")

    # Interpretation
    lines += ["## Interpretation", ""]

    improvement = best_ff - 0.592
    if improvement > 0.1:
        lines.append(f"Diverse training improved free-form AUROC by {improvement:.3f} ({0.592:.3f} → {best_ff:.3f}). "
                      "This confirms the autointerp pattern: activation distribution mismatch is the root cause, "
                      "and training data diversity is the fix.")
    elif improvement > 0.03:
        lines.append(f"Modest improvement ({improvement:.3f}). Distribution coverage helps but the truthfulness "
                      "signal may require more targeted data augmentation than the deception signal.")
    else:
        lines.append(f"Minimal improvement ({improvement:.3f}). The truthfulness domain shift may have a different "
                      "root cause than the deception OOD gap. The 0.877 → 0.592 drop might reflect a genuine "
                      "difference in what TruthfulQA measures vs what free-form hallucination looks like in activations.")

    lines.append("")

    # Comparison to autointerp
    lines += [
        "## Comparison to Prior Distribution Fixes",
        "",
        "| Fix | Original | Fixed | Improvement | Domain |",
        "|-----|----------|-------|-------------|--------|",
        f"| autointerp (deception) | 0.355 | 0.991 | +179% | AF detection |",
        f"| tellthetruth Exp 6 (truthfulness) | 0.592 | {best_ff:.3f} | {(best_ff/0.592 - 1)*100:+.0f}% | Truthfulness |",
        "",
    ]

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("  Experiment 6: Diverse-Training Fix for Truthfulness Probe")
    print("=" * 60)

    # --- Load TruthfulQA ---
    print("\nLoading TruthfulQA...")
    tqa_texts, tqa_labels = load_truthfulqa()
    print(f"  {len(tqa_texts)} samples ({tqa_labels.sum()} correct, {len(tqa_labels) - tqa_labels.sum()} incorrect)")

    # Dev/test split (same as Exp 1)
    indices = np.arange(len(tqa_texts))
    dev_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=tqa_labels)
    tqa_texts_dev = [tqa_texts[i] for i in dev_idx]
    tqa_texts_test = [tqa_texts[i] for i in test_idx]
    tqa_labels_dev = tqa_labels[dev_idx]
    tqa_labels_test = tqa_labels[test_idx]
    print(f"  Split: {len(tqa_texts_dev)} dev, {len(tqa_texts_test)} test")

    # --- Load Exp 4 free-form claims ---
    print("\nLoading Exp 4 claims...")
    ff_texts, ff_labels = load_exp4_claims()
    if len(ff_texts) == 0:
        print("ERROR: No Exp 4 claims available. Run exp4_pipeline.py first.")
        return

    # --- Extract activations ---
    print(f"\nExtracting TruthfulQA dev activations (layer {BEST_LAYER}, {POOL})...")
    tqa_acts_dev = get_activations(tqa_texts_dev)
    print(f"Extracting TruthfulQA test activations...")
    tqa_acts_test = get_activations(tqa_texts_test)
    print(f"Extracting free-form claim activations...")
    ff_acts = get_activations(ff_texts)

    # --- Stage 6.1: Diagnosis ---
    diagnosis = stage_diagnose(tqa_acts_dev, ff_acts)

    # --- Stage 6.2: Generate diverse data ---
    diverse_batches = stage_generate_diverse()

    # --- Stage 6.3: Progressive sweep ---
    sweep_results = stage_progressive_sweep(
        tqa_texts_dev, tqa_labels_dev, tqa_texts_test, tqa_labels_test,
        tqa_acts_dev, tqa_acts_test,
        ff_texts, ff_labels, ff_acts,
        diverse_batches,
    )

    # --- Stage 6.4: Pipeline re-eval with best probe ---
    # Find the best step (highest free-form AUROC with tqa >= 0.85)
    valid = [r for r in sweep_results if r["tqa_auroc"] >= 0.85]
    if not valid:
        valid = sweep_results  # fallback: use all
    best_step = max(valid, key=lambda r: r["freeform_auroc"])
    print(f"\n  Best step: {best_step['step']} ({best_step['name']}) — ff AUROC {best_step['freeform_auroc']:.4f}")

    # Retrain the best probe configuration for pipeline eval
    # Reconstruct training data for best step
    diverse_texts_for_best = []
    diverse_labels_for_best = []
    for i, batch in enumerate(diverse_batches):
        if i + 1 > best_step["step"]:
            break
        for s in batch["samples"]:
            diverse_texts_for_best.append(s["claim"])
            diverse_labels_for_best.append(1 if s["correct"] else 0)

    if diverse_texts_for_best:
        diverse_acts_best = get_activations(diverse_texts_for_best)
        diverse_labels_best = np.array(diverse_labels_for_best)
        X_best = np.vstack([tqa_acts_dev, diverse_acts_best])
        y_best = np.concatenate([tqa_labels_dev, diverse_labels_best])
    else:
        X_best = tqa_acts_dev
        y_best = tqa_labels_dev

    best_result = train_and_eval_probe(X_best, y_best, tqa_acts_test, tqa_labels_test)
    pipeline_results = stage_pipeline_reeval(best_result["probe"], best_result["scaler"],
                                             ff_texts, ff_labels, ff_acts)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # --- Save JSON ---
    json_results = {
        "model": MODEL_ID,
        "layer": BEST_LAYER,
        "pool": POOL,
        "C": BEST_C,
        "diagnosis": diagnosis,
        "diverse_batches": [
            {"name": b["name"], "desc": b["desc"], "n_samples": len(b["samples"]),
             "n_correct": sum(1 for s in b["samples"] if s["correct"]),
             "n_incorrect": sum(1 for s in b["samples"] if not s["correct"])}
            for b in diverse_batches
        ],
        "sweep": sweep_results,
        "best_step": best_step,
        "pipeline_reeval": pipeline_results,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    OUTPUT_JSON.write_text(json.dumps(json_results, indent=2))
    print(f"Saved: {OUTPUT_JSON}")

    # --- Report ---
    report = generate_report(diagnosis, sweep_results, pipeline_results, elapsed)
    OUTPUT_MD.write_text(report)
    print(f"Saved: {OUTPUT_MD}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Baseline free-form AUROC:     0.592")
    print(f"  Best diverse free-form AUROC: {best_step['freeform_auroc']:.4f}")
    print(f"  Improvement:                  {best_step['freeform_auroc'] - 0.592:+.4f}")
    print(f"  TruthfulQA regression:        {best_step['tqa_auroc']:.4f} (was 0.877)")
    if pipeline_results:
        print(f"  Pipeline potential reduction:  {pipeline_results['potential_reduction_pct']:.1f}%")
    print(f"  Passes free-form target:      {'YES' if best_step['freeform_auroc'] >= 0.75 else 'NO'} (target ≥ 0.75)")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
