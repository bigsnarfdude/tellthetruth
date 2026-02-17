#!/usr/bin/env python3
"""
Experiment 1: Truthfulness Probe Validation
=============================================
Full 26-layer sweep, 5-fold CV, ablations, calibration, bootstrap CIs.
Per EXPERIMENT_PROTOCOL.md Section 1.

Usage: python exp1_truthfulness_probe.py
Output: exp1_results.md + exp1_results.json
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
MODEL_ID = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_BOOTSTRAP = 1000
OUTPUT_JSON = Path("exp1_results.json")
OUTPUT_MD = Path("exp1_results.md")

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


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------
def extract_all_layers(texts, max_len=256):
    """Extract activations at ALL layers for all texts. Returns dict[layer] -> np.array."""
    # We store 3 pooling strategies: mean, first, last
    acts = {
        "mean": defaultdict(list),
        "first": defaultdict(list),
        "last": defaultdict(list),
    }
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"    [{i}/{len(texts)}]...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)
        seq_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for l in range(NUM_LAYERS + 1):  # 0 = embedding, 1..N = layers
            h = out.hidden_states[l]  # (1, seq, dim)
            acts["mean"][l].append(h.mean(dim=1).squeeze().float().cpu().numpy())
            acts["first"][l].append(h[0, 0, :].float().cpu().numpy())
            acts["last"][l].append(h[0, -1, :].float().cpu().numpy())

    # Stack into arrays
    for pool in acts:
        for l in acts[pool]:
            acts[pool][l] = np.stack(acts[pool][l])
    return acts


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
    rng = np.random.RandomState(SEED)
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_scores[idx]))
        except ValueError:
            continue
    alpha = (1 - ci) / 2
    lo = np.percentile(scores, alpha * 100)
    hi = np.percentile(scores, (1 - alpha) * 100)
    return float(np.mean(scores)), float(lo), float(hi)


def calibration_bins(y_true, y_prob, n_bins=10):
    """Compute calibration curve data."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    for i in range(n_bins):
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
# Experiment 1.1: Full layer sweep with 5-fold CV
# ---------------------------------------------------------------------------
def exp1_1_layer_sweep(acts, labels):
    """Full 26-layer sweep, 5-fold CV, C selection."""
    print("\n" + "=" * 60)
    print("  Exp 1.1: Full Layer Sweep (5-fold CV)")
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

            if l % 5 == 0 or l == NUM_LAYERS:
                print(f"    Layer {l:2d}: AUROC={best_auroc:.4f} (C={best_C})")

    return results


# ---------------------------------------------------------------------------
# Exp 1.2: Final probe on best layer with full metrics
# ---------------------------------------------------------------------------
def exp1_2_final_probe(acts, labels, layer_results):
    """Train final probe on best layer, compute all metrics + bootstrap CI."""
    print("\n" + "=" * 60)
    print("  Exp 1.2: Final Probe + Bootstrap CI")
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

    X = acts[best_pool][best_layer]
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

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
# Exp 1.3: Ablations
# ---------------------------------------------------------------------------
def exp1_3_ablations(acts, labels, best_pool, best_layer, best_C, texts):
    """Run all ablation experiments."""
    print("\n" + "=" * 60)
    print("  Exp 1.3: Ablations")
    print("=" * 60)

    X = acts[best_pool][best_layer]
    results = {}

    # --- Ablation 1: Random labels ---
    print("  [1/4] Random labels...")
    random_labels = np.random.RandomState(SEED).permutation(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
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
    print("  [4/4] Pooling comparison (already computed in layer sweep)")
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
# Report generation
# ---------------------------------------------------------------------------
def generate_report(layer_results, final, ablations, elapsed):
    """Write markdown report."""
    lines = [
        "# Experiment 1: Truthfulness Probe Validation",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Runtime: {elapsed:.0f}s*",
        "",
        "## Setup",
        "",
        f"- **Model:** {MODEL_ID} ({NUM_LAYERS} layers)",
        f"- **Data:** TruthfulQA generation split (817 questions → 1634 paired samples)",
        f"- **Protocol:** 5-fold stratified CV, C ∈ {{0.01, 0.1, 1.0, 10.0}}",
        f"- **Bootstrap:** {N_BOOTSTRAP} resamples for 95% CI",
        "",
        "## 1.1 Full Layer Sweep",
        "",
    ]

    # Layer sweep table for best pooling
    bp = final["best_pool"]
    lines += [
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
        "## 1.2 Final Probe (Test Set)",
        "",
        f"- **Layer:** {final['best_layer']}",
        f"- **Pooling:** {final['best_pool']}",
        f"- **C:** {final['best_C']}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| AUROC | {m['auroc']:.4f} [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}] |",
        f"| AUPRC | {m['auprc']:.4f} |",
        f"| Accuracy | {m['accuracy']:.4f} |",
        f"| Precision | {m['precision']:.4f} |",
        f"| Recall | {m['recall']:.4f} |",
        f"| F1 | {m['f1']:.4f} |",
        f"| Brier Score | {m['brier']:.4f} |",
        "",
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
        "## 1.3 Ablations",
        "",
        "| Ablation | AUROC | Expected | Pass? |",
        "|----------|-------|----------|-------|",
    ]
    rl = ablations["random_labels"]
    sa = ablations["shuffled_activations"]
    bow = ablations["bow_baseline"]
    pm = ablations["pooling_mean"]

    lines.append(f"| Random labels | {rl['auroc_mean']:.4f} ± {rl['auroc_std']:.4f} | ~0.5 | {'YES' if abs(rl['auroc_mean'] - 0.5) < 0.1 else 'NO'} |")
    lines.append(f"| Shuffled activations | {sa['auroc_mean']:.4f} ± {sa['auroc_std']:.4f} | ~0.5 | {'YES' if abs(sa['auroc_mean'] - 0.5) < 0.1 else 'NO'} |")
    lines.append(f"| BoW baseline | {bow['auroc_mean']:.4f} ± {bow['auroc_std']:.4f} | < probe | {'YES' if bow['auroc_mean'] < m['auroc'] else 'NO — vocabulary confound?'} |")

    lines += [
        "",
        "### Pooling Strategy Comparison",
        "",
        "| Pooling | AUROC (5-fold CV) |",
        "|---------|-------------------|",
        f"| mean | {ablations['pooling_mean']['auroc_mean']:.4f} ± {ablations['pooling_mean']['auroc_std']:.4f} |",
        f"| first | {ablations['pooling_first']['auroc_mean']:.4f} ± {ablations['pooling_first']['auroc_std']:.4f} |",
        f"| last | {ablations['pooling_last']['auroc_mean']:.4f} ± {ablations['pooling_last']['auroc_std']:.4f} |",
        "",
        "## Interpretation",
        "",
    ]

    # Auto-interpret results
    probe_auroc = m["auroc"]
    bow_auroc = bow["auroc_mean"]
    if probe_auroc > 0.75 and abs(rl["auroc_mean"] - 0.5) < 0.1:
        lines.append("The truthfulness probe shows genuine signal in the activation space.")
        if bow_auroc > probe_auroc - 0.05:
            lines.append("**Warning:** BoW baseline is close to probe AUROC — the signal may be partially vocabulary-based.")
        else:
            lines.append(f"The probe ({probe_auroc:.3f}) substantially outperforms BoW ({bow_auroc:.3f}), confirming the signal is in the representation geometry, not surface vocabulary.")
    elif probe_auroc < 0.6:
        lines.append("The truthfulness probe shows weak signal. TruthfulQA may not provide a strong enough training signal for this model.")
    else:
        lines.append(f"Moderate signal detected (AUROC={probe_auroc:.3f}). Further investigation needed.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("  Experiment 1: Truthfulness Probe Validation")
    print("=" * 60)

    # Load data
    print("\nLoading TruthfulQA...")
    texts, labels = load_truthfulqa()
    print(f"  {len(texts)} samples ({labels.sum()} correct, {len(labels) - labels.sum()} incorrect)")

    # Extract activations
    print("\nExtracting activations (all layers, all pooling)...")
    acts = extract_all_layers(texts)

    # Exp 1.1: Layer sweep
    layer_results = exp1_1_layer_sweep(acts, labels)

    # Exp 1.2: Final probe
    final = exp1_2_final_probe(acts, labels, layer_results)

    # Exp 1.3: Ablations
    ablations = exp1_3_ablations(
        acts, labels,
        final["best_pool"], final["best_layer"], final["best_C"],
        texts,
    )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save results
    # Remove non-serializable objects for JSON
    json_results = {
        "model": MODEL_ID,
        "num_layers": NUM_LAYERS,
        "n_samples": len(texts),
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
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    OUTPUT_JSON.write_text(json.dumps(json_results, indent=2))
    print(f"Saved: {OUTPUT_JSON}")

    # Save probe weights
    probe_path = Path("exp1_probe.pkl")
    with open(probe_path, "wb") as f:
        pickle.dump({
            "probe": final["probe"],
            "scaler": final["scaler"],
            "layer": final["best_layer"],
            "pool": final["best_pool"],
        }, f)
    print(f"Saved: {probe_path}")

    # Generate report
    report = generate_report(layer_results, final, ablations, elapsed)
    OUTPUT_MD.write_text(report)
    print(f"Saved: {OUTPUT_MD}")
    print("\nDone!")


if __name__ == "__main__":
    main()
