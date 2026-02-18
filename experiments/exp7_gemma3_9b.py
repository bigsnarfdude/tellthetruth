#!/usr/bin/env python3
"""
Experiment 7: Gemma-3-4B-IT Truthfulness Probe (Bigger Model Test)
===================================================================
Tests whether Gemma-3-4B-IT's richer representations produce a truthfulness
probe that generalizes better across formats (QA → free-form).

Gemma-2-2B-it (Exp 1) scores 0.877 on TruthfulQA but drops to 0.592 on
free-form generation (Exp 4). The hypothesis: a 2x larger model encodes
truthfulness more abstractly, reducing the OOD gap.

Model loaded in bf16 (~8GB). Fits on 16GB VRAM without quantization.
Gemma 3 is multimodal (Gemma3ForConditionalGeneration) but we use text-only;
hidden states come from the text model (34 layers, hidden_dim=2560).

NOTE: We initially tried Gemma-3-4B-IT in int4, but quantization produced
NaN/inf in hidden states at layers 12+, making probe training impossible.
The 4B model in bf16 avoids this artifact entirely.

Stages:
  7.1 Full layer sweep (all layers, 5-fold CV on TruthfulQA dev set)
  7.2 Final probe on best layer (train dev, eval held-out test)
  7.3 Ablations (random labels, shuffled activations, BoW, pooling)
  7.4 OOD evaluation on Exp 4 free-form claims
  7.5 Gap comparison to Gemma-2-2B (Exp 1)

Usage: python exp7_gemma3_9b.py
Output: exp7_results.md + exp7_results.json
"""

import json
import time
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_BOOTSTRAP = 1000
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = REPO_ROOT / "results" / "exp7_results.json"
OUTPUT_MD = REPO_ROOT / "results" / "exp7_results.md"

# Gemma-2-2B baselines from Exp 1 and Exp 4 (for comparison)
GEMMA2_IN_DIST_AUROC = 0.877
GEMMA2_OOD_AUROC = 0.592
GEMMA2_BEST_LAYER = 16

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load model (bf16 — fits on 16GB without quantization)
# ---------------------------------------------------------------------------
print("Loading model (bf16)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
# Gemma 3 is multimodal: config is nested under text_config
_cfg = getattr(model.config, "text_config", model.config)
NUM_LAYERS = _cfg.num_hidden_layers
HIDDEN_DIM = _cfg.hidden_size
print(f"Loaded {MODEL_ID} (int4), {NUM_LAYERS} layers, hidden_dim={HIDDEN_DIM}")
print(f"Model class: {model.__class__.__name__}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_truthfulqa():
    """Build paired correct/incorrect from TruthfulQA."""
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


def load_exp4_claims():
    """Load verifiable claims from Exp 4 results (OOD eval)."""
    exp4_path = REPO_ROOT / "results" / "exp4_results.json"
    if not exp4_path.exists():
        exp4_path = REPO_ROOT / "results" / "exp4_results_sonnet.json"
    if not exp4_path.exists():
        print("  [warn] No exp4 results found. OOD evaluation skipped.")
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


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------
def _clean(arr):
    """Replace NaN/inf with 0 — int4 quantization can produce numerical artifacts."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_all_layers(texts, max_len=256):
    """Extract activations at ALL layers for all texts. Returns dict[pool][layer] -> np.array."""
    acts = {
        "mean": defaultdict(list),
        "first": defaultdict(list),
        "last": defaultdict(list),
    }
    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"    [{i}/{len(texts)}]...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for l in range(NUM_LAYERS + 1):  # 0 = embedding, 1..N = layers
            h = out.hidden_states[l]  # (1, seq, dim)
            acts["mean"][l].append(_clean(h.mean(dim=1).squeeze().float().cpu().numpy()))
            acts["first"][l].append(_clean(h[0, 0, :].float().cpu().numpy()))
            acts["last"][l].append(_clean(h[0, -1, :].float().cpu().numpy()))

    # Stack into arrays
    n_inf = 0
    for pool in acts:
        for l in acts[pool]:
            stacked = np.stack(acts[pool][l])
            n_inf += np.isinf(stacked).sum() + np.isnan(stacked).sum()
            acts[pool][l] = stacked
    if n_inf > 0:
        print(f"  [warn] {n_inf} NaN/inf values cleaned from activations")
    return acts


def extract_single_layer(texts, layer, pool="last", max_len=256):
    """Extract activations at a single layer. More memory-efficient for OOD eval."""
    all_acts = []
    for i, text in enumerate(texts):
        if i % 50 == 0 and i > 0:
            print(f"    [{i}/{len(texts)}]...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[layer]
        if pool == "last":
            all_acts.append(_clean(h[0, -1, :].float().cpu().numpy()))
        elif pool == "first":
            all_acts.append(_clean(h[0, 0, :].float().cpu().numpy()))
        else:
            all_acts.append(_clean(h.mean(dim=1).squeeze().float().cpu().numpy()))
    return np.stack(all_acts)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def eval_probe(probe, scaler, X_test, y_test):
    """Compute full metrics for a trained probe."""
    X_s = scaler.transform(X_test)
    y_pred = probe.predict(X_s)
    y_prob = probe.predict_proba(X_s)[:, 1]

    return {
        "auroc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "brier": brier_score_loss(y_test, y_prob),
    }


def bootstrap_ci(y_true, y_scores, metric_fn, n=N_BOOTSTRAP, ci=0.95):
    """Bootstrap 95% CI for a metric."""
    scores = []
    dropped = 0
    rng = np.random.RandomState(SEED)
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_scores[idx]))
        except ValueError:
            dropped += 1
            continue
    if dropped > 0:
        print(f"    [bootstrap] {dropped}/{n} samples dropped (single-class)")
    alpha = (1 - ci) / 2
    lo = np.percentile(scores, alpha * 100)
    hi = np.percentile(scores, (1 - alpha) * 100)
    return float(np.mean(scores)), float(lo), float(hi)


def calibration_bins(y_true, y_prob, n_bins=10):
    """Compute calibration curve data."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_data.append({
            "bin_center": float((bins[i] + bins[i + 1]) / 2),
            "mean_predicted": float(y_prob[mask].mean()),
            "mean_actual": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return bin_data


# ---------------------------------------------------------------------------
# Experiment 7.1: Full layer sweep with 5-fold CV
# ---------------------------------------------------------------------------
def exp7_1_layer_sweep(acts, labels):
    """Full layer sweep, 5-fold CV, C selection."""
    print("\n" + "=" * 60)
    print("  Exp 7.1: Full Layer Sweep (5-fold CV)")
    print("=" * 60)

    C_values = [0.01, 0.1, 1.0, 10.0]
    results = {}

    for pool in ["mean", "first", "last"]:
        print(f"\n  Pooling: {pool}")
        results[pool] = {}

        for l in range(NUM_LAYERS + 1):
            X = acts[pool][l]
            best_auroc = 0
            best_C = 1.0

            for C in C_values:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                fold_aurocs = []
                for train_idx, val_idx in skf.split(X, labels):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X[train_idx])
                    X_val = scaler.transform(X[val_idx])

                    probe = LogisticRegression(C=C, max_iter=1000, random_state=SEED)
                    probe.fit(X_tr, labels[train_idx])
                    y_prob = probe.predict_proba(X_val)[:, 1]
                    fold_aurocs.append(roc_auc_score(labels[val_idx], y_prob))

                mean_auroc = np.mean(fold_aurocs)
                if mean_auroc > best_auroc:
                    best_auroc = mean_auroc
                    best_C = C

            results[pool][l] = {
                "cv_auroc": float(best_auroc),
                "best_C": best_C,
            }

            if l % 6 == 0 or l == NUM_LAYERS:
                print(f"    Layer {l:2d}: AUROC={best_auroc:.4f} (C={best_C})")

    return results


# ---------------------------------------------------------------------------
# Exp 7.2: Final probe on best layer with full metrics
# ---------------------------------------------------------------------------
def exp7_2_final_probe(acts_dev, acts_test, labels_dev, labels_test, layer_results):
    """Train final probe on best layer, compute all metrics + bootstrap CI."""
    print("\n" + "=" * 60)
    print("  Exp 7.2: Final Probe + Bootstrap CI")
    print("=" * 60)

    # Find best pool/layer combo
    best_auroc = 0
    best_pool, best_layer, best_C = "mean", 15, 1.0
    for pool in layer_results:
        for l in layer_results[pool]:
            if layer_results[pool][l]["cv_auroc"] > best_auroc:
                best_auroc = layer_results[pool][l]["cv_auroc"]
                best_pool = pool
                best_layer = l
                best_C = layer_results[pool][l]["best_C"]

    print(f"  Best: pool={best_pool}, layer={best_layer}, C={best_C}, CV AUROC={best_auroc:.4f}")

    # Train on full dev set, evaluate on held-out test set
    X_train = acts_dev[best_pool][best_layer]
    y_train = labels_dev
    X_test = acts_test[best_pool][best_layer]
    y_test = labels_test

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    probe = LogisticRegression(C=best_C, max_iter=1000, random_state=SEED)
    probe.fit(X_train_s, y_train)

    metrics = eval_probe(probe, scaler, X_test, y_test)
    print(f"  Test metrics: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")

    # Bootstrap CI
    y_prob = probe.predict_proba(X_test_s)[:, 1]
    auroc_mean, auroc_lo, auroc_hi = bootstrap_ci(y_test, y_prob, roc_auc_score)
    print(f"  AUROC: {auroc_mean:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}] (95% CI)")

    # Calibration
    cal = calibration_bins(y_test, y_prob)

    # Classification report
    report = classification_report(y_test, probe.predict(X_test_s),
                                   target_names=["incorrect", "correct"], output_dict=True)

    return {
        "best_pool": best_pool,
        "best_layer": int(best_layer),
        "best_C": best_C,
        "metrics": metrics,
        "bootstrap_auroc": {"mean": auroc_mean, "ci_lo": auroc_lo, "ci_hi": auroc_hi},
        "calibration": cal,
        "classification_report": report,
        "probe": probe,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Exp 7.3: Ablations
# ---------------------------------------------------------------------------
def exp7_3_ablations(acts, labels, best_pool, best_layer, best_C, texts):
    """Run all ablation experiments."""
    print("\n" + "=" * 60)
    print("  Exp 7.3: Ablations")
    print("=" * 60)

    X = acts[best_pool][best_layer]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    # --- Ablation 1: Random labels ---
    print("  [1/4] Random labels...")
    random_labels = np.random.RandomState(SEED).permutation(labels)
    fold_aurocs = []
    for train_idx, val_idx in skf.split(X, random_labels):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        probe = LogisticRegression(C=best_C, max_iter=1000, random_state=SEED)
        probe.fit(X_tr, random_labels[train_idx])
        y_prob = probe.predict_proba(X_val)[:, 1]
        fold_aurocs.append(roc_auc_score(random_labels[val_idx], y_prob))
    results["random_labels"] = {
        "auroc_mean": float(np.mean(fold_aurocs)),
        "auroc_std": float(np.std(fold_aurocs)),
        "expected": "~0.5",
    }
    print(f"    AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

    # --- Ablation 2: Shuffled activations ---
    print("  [2/4] Shuffled activations...")
    X_shuffled = X.copy()
    rng = np.random.RandomState(SEED)
    for col in range(X_shuffled.shape[1]):
        rng.shuffle(X_shuffled[:, col])
    fold_aurocs = []
    for train_idx, val_idx in skf.split(X_shuffled, labels):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_shuffled[train_idx])
        X_val = scaler.transform(X_shuffled[val_idx])
        probe = LogisticRegression(C=best_C, max_iter=1000, random_state=SEED)
        probe.fit(X_tr, labels[train_idx])
        y_prob = probe.predict_proba(X_val)[:, 1]
        fold_aurocs.append(roc_auc_score(labels[val_idx], y_prob))
    results["shuffled_activations"] = {
        "auroc_mean": float(np.mean(fold_aurocs)),
        "auroc_std": float(np.std(fold_aurocs)),
        "expected": "~0.5",
    }
    print(f"    AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

    # --- Ablation 3: BoW baseline ---
    print("  [3/4] Bag-of-words baseline...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_bow = tfidf.fit_transform(texts).toarray()
    fold_aurocs = []
    for train_idx, val_idx in skf.split(X_bow, labels):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_bow[train_idx])
        X_val = scaler.transform(X_bow[val_idx])
        probe = LogisticRegression(C=best_C, max_iter=1000, random_state=SEED)
        probe.fit(X_tr, labels[train_idx])
        y_prob = probe.predict_proba(X_val)[:, 1]
        fold_aurocs.append(roc_auc_score(labels[val_idx], y_prob))
    results["bow_baseline"] = {
        "auroc_mean": float(np.mean(fold_aurocs)),
        "auroc_std": float(np.std(fold_aurocs)),
        "note": "If close to probe AUROC, signal may be vocabulary-based",
    }
    print(f"    AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

    # --- Ablation 4: Pooling comparison ---
    print("  [4/4] Pooling comparison at best layer...")
    for pool in ["mean", "first", "last"]:
        X_p = acts[pool][best_layer]
        fold_aurocs = []
        for train_idx, val_idx in skf.split(X_p, labels):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_p[train_idx])
            X_val = scaler.transform(X_p[val_idx])
            probe = LogisticRegression(C=best_C, max_iter=1000, random_state=SEED)
            probe.fit(X_tr, labels[train_idx])
            y_prob = probe.predict_proba(X_val)[:, 1]
            fold_aurocs.append(roc_auc_score(labels[val_idx], y_prob))
        results[f"pooling_{pool}"] = {
            "auroc_mean": float(np.mean(fold_aurocs)),
            "auroc_std": float(np.std(fold_aurocs)),
        }
    print(f"    mean: {results['pooling_mean']['auroc_mean']:.4f}")
    print(f"    first: {results['pooling_first']['auroc_mean']:.4f}")
    print(f"    last: {results['pooling_last']['auroc_mean']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Exp 7.4: OOD evaluation on Exp 4 free-form claims
# ---------------------------------------------------------------------------
def exp7_4_ood_eval(probe, scaler, best_layer, best_pool):
    """Evaluate probe on Exp 4 free-form claims (the critical OOD test)."""
    print("\n" + "=" * 60)
    print("  Exp 7.4: OOD Evaluation (Exp 4 Free-Form Claims)")
    print("=" * 60)

    ff_texts, ff_labels = load_exp4_claims()
    if len(ff_texts) == 0:
        print("  Skipping OOD eval — no exp4 claims available.")
        return None

    # Extract activations from Gemma-3-4B for the same claims
    print(f"  Extracting Gemma-3-4B activations for {len(ff_texts)} claims (layer {best_layer}, {best_pool})...")
    ff_acts = extract_single_layer(ff_texts, best_layer, best_pool)

    # Score with probe
    ff_acts_s = scaler.transform(ff_acts)
    ff_prob = probe.predict_proba(ff_acts_s)[:, 1]
    ff_pred = probe.predict(ff_acts_s)

    # Metrics
    ood_auroc = float(roc_auc_score(ff_labels, ff_prob))
    ood_acc = float(accuracy_score(ff_labels, ff_pred))
    ood_f1 = float(f1_score(ff_labels, ff_pred, zero_division=0))
    ood_prec = float(precision_score(ff_labels, ff_pred, zero_division=0))
    ood_rec = float(recall_score(ff_labels, ff_pred, zero_division=0))

    # Bootstrap CI
    auroc_mean, auroc_lo, auroc_hi = bootstrap_ci(ff_labels, ff_prob, roc_auc_score)

    # Flag analysis
    n_flagged = int((ff_prob < 0.5).sum())
    flag_rate = n_flagged / len(ff_prob) * 100

    print(f"  OOD AUROC: {ood_auroc:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]")
    print(f"  OOD Accuracy: {ood_acc:.4f}")
    print(f"  OOD F1: {ood_f1:.4f}")
    print(f"  Flag rate: {flag_rate:.1f}%")

    # Distribution analysis: compare TruthfulQA vs free-form activation norms
    ff_norms = np.linalg.norm(ff_acts, axis=1)
    print(f"  Free-form activation norms: {ff_norms.mean():.1f} ± {ff_norms.std():.1f}")

    return {
        "auroc": ood_auroc,
        "accuracy": ood_acc,
        "f1": ood_f1,
        "precision": ood_prec,
        "recall": ood_rec,
        "bootstrap_auroc": {"mean": auroc_mean, "ci_lo": auroc_lo, "ci_hi": auroc_hi},
        "n_claims": len(ff_texts),
        "n_correct": int(ff_labels.sum()),
        "n_incorrect": int(len(ff_labels) - ff_labels.sum()),
        "n_flagged": n_flagged,
        "flag_rate": float(flag_rate),
        "activation_norms": {
            "mean": float(ff_norms.mean()),
            "std": float(ff_norms.std()),
            "min": float(ff_norms.min()),
            "max": float(ff_norms.max()),
        },
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(layer_results, final, ablations, ood_results, elapsed):
    """Write markdown report."""
    lines = [
        "# Experiment 7: Gemma-3-4B-IT Truthfulness Probe",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Runtime: {elapsed:.0f}s*",
        "",
        "## Hypothesis",
        "",
        "Gemma-2-2B-it's truthfulness probe (Exp 1) scores 0.877 in-distribution but drops",
        "to 0.592 on free-form generation (Exp 4). A larger model (Gemma-3-4B-IT) may encode",
        "truthfulness more abstractly, producing a probe that generalizes across formats.",
        "",
        "## Setup",
        "",
        f"- **Model:** {MODEL_ID} ({NUM_LAYERS} layers, hidden_dim={HIDDEN_DIM})",
        f"- **Precision:** bf16 (no quantization)",
        f"- **Data:** TruthfulQA generation split (817 questions → 1634 paired samples)",
        f"- **Protocol:** 5-fold stratified CV, C ∈ {{0.01, 0.1, 1.0, 10.0}}",
        f"- **Bootstrap:** {N_BOOTSTRAP} resamples for 95% CI",
        "",
    ]

    # Layer sweep table for best pooling
    bp = final["best_pool"]
    lines += [
        "## 7.1 Full Layer Sweep",
        "",
        f"Best pooling: **{bp}**",
        "",
        "| Layer | AUROC (5-fold CV) | Best C |",
        "|-------|-------------------|--------|",
    ]
    for l in sorted(layer_results[bp].keys()):
        r = layer_results[bp][l]
        marker = " ← best" if l == final["best_layer"] else ""
        lines.append(f"| {l} | {r['cv_auroc']:.4f} | {r['best_C']} |{marker}")

    # Cross-pooling comparison at best layer
    bl = final["best_layer"]
    lines += [
        "",
        f"### Pooling comparison at layer {bl}",
        "",
        "| Pooling | AUROC |",
        "|---------|-------|",
    ]
    for pool in ["mean", "first", "last"]:
        lines.append(f"| {pool} | {layer_results[pool][bl]['cv_auroc']:.4f} |")

    # Final probe
    m = final["metrics"]
    b = final["bootstrap_auroc"]
    lines += [
        "",
        "## 7.2 Final Probe (Test Set — In-Distribution)",
        "",
        f"- **Layer:** {final['best_layer']}",
        f"- **Pooling:** {final['best_pool']}",
        f"- **C:** {final['best_C']}",
        "",
        "| Metric | Gemma-3-4B | Gemma-2-2B (Exp 1) |",
        "|--------|-----------|-------------------|",
        f"| AUROC | {m['auroc']:.4f} [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}] | {GEMMA2_IN_DIST_AUROC:.4f} [0.836, 0.910] |",
        f"| AUPRC | {m['auprc']:.4f} | 0.887 |",
        f"| Accuracy | {m['accuracy']:.4f} | 0.774 |",
        f"| F1 | {m['f1']:.4f} | 0.774 |",
        f"| Brier Score | {m['brier']:.4f} | 0.153 |",
        "",
    ]

    in_dist_diff = m["auroc"] - GEMMA2_IN_DIST_AUROC
    lines.append(f"**In-distribution difference:** {in_dist_diff:+.4f}")
    lines.append("")

    # Classification report
    lines += [
        "### Classification Report",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for cls in ["incorrect", "correct"]:
        cr = final["classification_report"][cls]
        lines.append(f"| {cls} | {cr['precision']:.3f} | {cr['recall']:.3f} | {cr['f1-score']:.3f} | {cr['support']} |")

    # Calibration
    lines += [
        "",
        "### Calibration",
        "",
        "| Bin Center | Mean Predicted | Mean Actual | Count |",
        "|------------|---------------|-------------|-------|",
    ]
    for b in final["calibration"]:
        lines.append(f"| {b['bin_center']:.2f} | {b['mean_predicted']:.3f} | {b['mean_actual']:.3f} | {b['count']} |")

    # Ablations
    lines += [
        "",
        "## 7.3 Ablations",
        "",
        "| Ablation | AUROC | Expected | Pass? |",
        "|----------|-------|----------|-------|",
    ]
    rl = ablations["random_labels"]
    sa = ablations["shuffled_activations"]
    bow = ablations["bow_baseline"]

    best_cv_auroc = max(layer_results[bp][l]["cv_auroc"] for l in layer_results[bp])
    lines.append(f"| Random labels | {rl['auroc_mean']:.4f} ± {rl['auroc_std']:.4f} | ~0.5 | {'YES' if abs(rl['auroc_mean'] - 0.5) < 0.1 else 'NO'} |")
    lines.append(f"| Shuffled activations | {sa['auroc_mean']:.4f} ± {sa['auroc_std']:.4f} | ~0.5 | {'YES' if abs(sa['auroc_mean'] - 0.5) < 0.1 else 'NO'} |")
    lines.append(f"| BoW baseline | {bow['auroc_mean']:.4f} ± {bow['auroc_std']:.4f} | < probe | {'YES' if bow['auroc_mean'] < best_cv_auroc else 'NO — vocabulary confound?'} |")

    lines += [
        "",
        "### Pooling Strategy Comparison",
        "",
        "| Pooling | AUROC (5-fold CV) |",
        "|---------|-------------------|",
        f"| mean | {ablations['pooling_mean']['auroc_mean']:.4f} ± {ablations['pooling_mean']['auroc_std']:.4f} |",
        f"| first | {ablations['pooling_first']['auroc_mean']:.4f} ± {ablations['pooling_first']['auroc_std']:.4f} |",
        f"| last | {ablations['pooling_last']['auroc_mean']:.4f} ± {ablations['pooling_last']['auroc_std']:.4f} |",
    ]

    # OOD evaluation (the critical test)
    lines += ["", "## 7.4 OOD Evaluation (Free-Form Claims from Exp 4)", ""]

    if ood_results:
        ood = ood_results
        ood_b = ood["bootstrap_auroc"]
        ood_gap_9b = m["auroc"] - ood["auroc"]
        ood_gap_2b = GEMMA2_IN_DIST_AUROC - GEMMA2_OOD_AUROC

        lines += [
            "| Metric | Gemma-3-4B | Gemma-2-2B (Exp 1/4) |",
            "|--------|-----------|---------------------|",
            f"| In-distribution AUROC | {m['auroc']:.4f} | {GEMMA2_IN_DIST_AUROC:.4f} |",
            f"| OOD AUROC (free-form) | **{ood['auroc']:.4f}** [{ood_b['ci_lo']:.4f}, {ood_b['ci_hi']:.4f}] | **{GEMMA2_OOD_AUROC:.4f}** |",
            f"| OOD gap (in-dist - OOD) | **{ood_gap_9b:.4f}** | **{ood_gap_2b:.4f}** |",
            f"| OOD Accuracy | {ood['accuracy']:.4f} | — |",
            f"| OOD F1 | {ood['f1']:.4f} | — |",
            f"| Flag rate | {ood['flag_rate']:.1f}% | 95.7% |",
            "",
            f"- **Claims evaluated:** {ood['n_claims']} ({ood['n_correct']} correct, {ood['n_incorrect']} incorrect)",
            f"- **Free-form activation norms:** {ood['activation_norms']['mean']:.1f} ± {ood['activation_norms']['std']:.1f}",
            "",
        ]

        # Interpretation of OOD gap
        lines += ["## 7.5 Gap Comparison", ""]

        gap_improvement = ood_gap_2b - ood_gap_9b
        ood_improvement = ood["auroc"] - GEMMA2_OOD_AUROC

        lines += [
            "| Metric | Gemma-2-2B | Gemma-3-4B | Change |",
            "|--------|-----------|-----------|--------|",
            f"| In-dist AUROC | {GEMMA2_IN_DIST_AUROC:.4f} | {m['auroc']:.4f} | {in_dist_diff:+.4f} |",
            f"| OOD AUROC | {GEMMA2_OOD_AUROC:.4f} | {ood['auroc']:.4f} | {ood_improvement:+.4f} |",
            f"| OOD gap | {ood_gap_2b:.4f} | {ood_gap_9b:.4f} | {-gap_improvement:+.4f} |",
            f"| Best layer | L{GEMMA2_BEST_LAYER} | L{final['best_layer']} | — |",
            f"| Hidden dim | 2304 (Gemma 2) | {HIDDEN_DIM} (Gemma 3) | +{HIDDEN_DIM - 2304} |",
            "",
        ]

        if ood_improvement > 0.10:
            lines.append(f"**The bigger model substantially improves OOD generalization** ({ood_improvement:+.3f} AUROC). "
                          f"The OOD gap shrank from {ood_gap_2b:.3f} to {ood_gap_9b:.3f}. "
                          "Gemma-3-4B-IT encodes truthfulness more abstractly, producing a probe "
                          "that transfers better across text formats.")
        elif ood_improvement > 0.03:
            lines.append(f"**Modest improvement** ({ood_improvement:+.3f} AUROC). The bigger model helps somewhat "
                          f"but the OOD gap ({ood_gap_9b:.3f}) is still substantial. Model size alone "
                          "does not fully solve the format generalization problem.")
        elif ood_improvement > -0.03:
            lines.append(f"**No meaningful change** ({ood_improvement:+.3f} AUROC). The OOD gap persists "
                          f"({ood_gap_9b:.3f} vs {ood_gap_2b:.3f}) despite 2x more parameters. "
                          "The format overfitting is not a capacity issue — it's a representation structure issue "
                          "that more parameters don't fix.")
        else:
            lines.append(f"**The bigger model is worse OOD** ({ood_improvement:+.3f} AUROC). More parameters "
                          "may lead to more format-specific representations, not more abstract ones. "
                          "The truthfulness probe problem is fundamentally about what the probe training data "
                          "teaches, not model capacity.")
        lines.append("")

    else:
        lines += [
            "**Skipped** — no Exp 4 claims available. Run exp4_pipeline.py first.",
            "",
        ]

    # Final interpretation
    lines += ["## Interpretation", ""]

    probe_auroc = m["auroc"]
    bow_auroc = bow["auroc_mean"]
    if probe_auroc > 0.75 and abs(rl["auroc_mean"] - 0.5) < 0.1:
        lines.append("The truthfulness probe shows genuine signal in Gemma-3-4B-IT's activation space.")
        if bow_auroc > best_cv_auroc - 0.05:
            lines.append("**Warning:** BoW baseline is close to probe AUROC — the signal may be partially vocabulary-based.")
        else:
            lines.append(f"The probe (CV {best_cv_auroc:.3f}, test {probe_auroc:.3f}) substantially outperforms "
                          f"BoW ({bow_auroc:.3f}), confirming the signal is geometric, not vocabulary.")

    if ood_results:
        ood_auroc_val = ood_results["auroc"]
        if ood_auroc_val > 0.75:
            lines.append(f"\nThe probe generalizes to free-form text (OOD AUROC = {ood_auroc_val:.3f}), "
                          "confirming that larger models encode truthfulness more format-invariantly.")
        elif ood_auroc_val > GEMMA2_OOD_AUROC + 0.05:
            lines.append(f"\nOOD performance improved over Gemma-2-2B ({GEMMA2_OOD_AUROC:.3f} → {ood_auroc_val:.3f}) "
                          "but remains below the useful threshold of 0.75.")
        else:
            lines.append(f"\nOOD performance ({ood_auroc_val:.3f}) is comparable to Gemma-2-2B ({GEMMA2_OOD_AUROC:.3f}). "
                          "Model scale does not fix the format generalization problem for truthfulness probes.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("  Experiment 7: Gemma-3-4B-IT Truthfulness Probe")
    print("=" * 60)

    # Load data
    print("\nLoading TruthfulQA...")
    texts, labels = load_truthfulqa()
    print(f"  {len(texts)} samples ({labels.sum()} correct, {len(labels) - labels.sum()} incorrect)")

    # Split dev/test UPFRONT
    indices = np.arange(len(texts))
    dev_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=labels
    )
    texts_dev = [texts[i] for i in dev_idx]
    texts_test = [texts[i] for i in test_idx]
    labels_dev = labels[dev_idx]
    labels_test = labels[test_idx]
    print(f"  Split: {len(texts_dev)} dev, {len(texts_test)} test (held-out)")

    # Extract activations (all layers)
    print("\nExtracting dev activations (all layers, all pooling)...")
    acts_dev = extract_all_layers(texts_dev)
    print("\nExtracting test activations (all layers, all pooling)...")
    acts_test = extract_all_layers(texts_test)

    # Exp 7.1: Layer sweep on dev set
    layer_results = exp7_1_layer_sweep(acts_dev, labels_dev)

    # Exp 7.2: Final probe (train on dev, eval on test)
    final = exp7_2_final_probe(acts_dev, acts_test, labels_dev, labels_test, layer_results)

    # Exp 7.3: Ablations (use dev set)
    ablations = exp7_3_ablations(
        acts_dev, labels_dev,
        final["best_pool"], final["best_layer"], final["best_C"],
        texts_dev,
    )

    # Exp 7.4: OOD evaluation on Exp 4 claims
    ood_results = exp7_4_ood_eval(
        final["probe"], final["scaler"],
        final["best_layer"], final["best_pool"],
    )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save results
    json_results = {
        "model": MODEL_ID,
        "quantization": "none_bf16",
        "num_layers": NUM_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "n_samples": len(texts),
        "n_dev": len(texts_dev),
        "n_test": len(texts_test),
        "layer_sweep": {
            pool: {str(l): v for l, v in layers.items()}
            for pool, layers in layer_results.items()
        },
        "final_probe": {
            "best_pool": final["best_pool"],
            "best_layer": final["best_layer"],
            "best_C": final["best_C"],
            "metrics": final["metrics"],
            "bootstrap_auroc": final["bootstrap_auroc"],
            "calibration": final["calibration"],
            "classification_report": final["classification_report"],
        },
        "ablations": ablations,
        "ood_evaluation": ood_results,
        "comparison_to_gemma2": {
            "gemma2_in_dist_auroc": GEMMA2_IN_DIST_AUROC,
            "gemma2_ood_auroc": GEMMA2_OOD_AUROC,
            "gemma2_best_layer": GEMMA2_BEST_LAYER,
            "gemma3_in_dist_auroc": final["metrics"]["auroc"],
            "gemma3_ood_auroc": ood_results["auroc"] if ood_results else None,
            "gemma3_best_layer": final["best_layer"],
            "in_dist_diff": final["metrics"]["auroc"] - GEMMA2_IN_DIST_AUROC,
            "ood_diff": (ood_results["auroc"] - GEMMA2_OOD_AUROC) if ood_results else None,
            "ood_gap_gemma2": GEMMA2_IN_DIST_AUROC - GEMMA2_OOD_AUROC,
            "ood_gap_gemma3": (final["metrics"]["auroc"] - ood_results["auroc"]) if ood_results else None,
        },
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    OUTPUT_JSON.write_text(json.dumps(json_results, indent=2))
    print(f"Saved: {OUTPUT_JSON}")

    # Save probe weights
    probe_path = REPO_ROOT / "results" / "exp7_probe.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump({
            "probe": final["probe"],
            "scaler": final["scaler"],
            "layer": final["best_layer"],
            "pool": final["best_pool"],
            "model": MODEL_ID,
        }, f)
    print(f"Saved: {probe_path}")

    # Generate report
    report = generate_report(layer_results, final, ablations, ood_results, elapsed)
    OUTPUT_MD.write_text(report)
    print(f"Saved: {OUTPUT_MD}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Model: {MODEL_ID} (int4)")
    print(f"  Best layer: {final['best_layer']} (Gemma-2-2B was L{GEMMA2_BEST_LAYER})")
    print(f"  In-dist AUROC: {final['metrics']['auroc']:.4f} (Gemma-2-2B: {GEMMA2_IN_DIST_AUROC})")
    if ood_results:
        ood_gap = final["metrics"]["auroc"] - ood_results["auroc"]
        print(f"  OOD AUROC:     {ood_results['auroc']:.4f} (Gemma-2-2B: {GEMMA2_OOD_AUROC})")
        print(f"  OOD gap:       {ood_gap:.4f} (Gemma-2-2B gap: {GEMMA2_IN_DIST_AUROC - GEMMA2_OOD_AUROC:.4f})")
        if ood_results["auroc"] > GEMMA2_OOD_AUROC + 0.05:
            print(f"  --> BIGGER MODEL HELPS OOD (+{ood_results['auroc'] - GEMMA2_OOD_AUROC:.3f})")
        elif ood_results["auroc"] < GEMMA2_OOD_AUROC - 0.05:
            print(f"  --> BIGGER MODEL WORSE OOD ({ood_results['auroc'] - GEMMA2_OOD_AUROC:+.3f})")
        else:
            print(f"  --> NO MEANINGFUL DIFFERENCE ({ood_results['auroc'] - GEMMA2_OOD_AUROC:+.3f})")
    else:
        print(f"  OOD AUROC:     N/A (run exp4_pipeline.py first)")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
