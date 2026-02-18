#!/usr/bin/env python3
"""
Experiment 3: Orthogonality Verification
==========================================
Verifies that truthfulness and deception probes detect orthogonal signals.
Per EXPERIMENT_PROTOCOL.md Section 3.

Protocol:
  3.1 Cosine similarity between probe weight vectors
  3.2 Subspace analysis via PCA + principal angles
  3.3 Combined 2D probe from both signals

Usage: python exp3_orthogonality.py
Output: exp3_results.md + exp3_results.json
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_BOOTSTRAP = 1000
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = REPO_ROOT / "results" / "exp3_results.json"
OUTPUT_MD = REPO_ROOT / "results" / "exp3_results.md"

np.random.seed(SEED)
torch.manual_seed(SEED)

# HF cache for AF data
HF_CACHE = Path.home() / ".cache/huggingface/hub"


def _find_cached(dataset_name, filename):
    pattern_dir = HF_CACHE / f"datasets--vincentoh--{dataset_name}"
    matches = sorted(pattern_dir.rglob(filename), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"{filename} not found in {pattern_dir}")
    return matches[-1]


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
# Activation extraction
# ---------------------------------------------------------------------------
def extract_activations(texts, layer, max_len=512, pooling="last"):
    """Extract activations at a single layer."""
    acts = []
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"    [{i}/{len(texts)}]...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[layer]
        if pooling == "last":
            vec = h[0, -1, :].float().cpu().numpy()
        elif pooling == "mean":
            vec = h.mean(dim=1).squeeze().float().cpu().numpy()
        else:
            vec = h[0, 0, :].float().cpu().numpy()
        acts.append(vec)
    return np.stack(acts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_truthfulqa():
    """TruthfulQA paired correct/incorrect."""
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


def load_deception_data(max_per_class=500):
    """AF organisms + potential_faking vs aligned."""
    training_path = _find_cached("alignment-faking-training", "training_data_final.json")
    organism_path = _find_cached("af-model-organisms", "best_samples.json")

    with open(training_path) as f:
        training = json.load(f)
    with open(organism_path) as f:
        organisms = json.load(f)

    aligned = [s["text"] for s in training["samples"] if s["label"] == "aligned"]
    deceptive = [s["text"] for s in organisms["samples"]]
    deceptive += [s["text"] for s in training["samples"] if s["label"] == "potential_faking"]

    rng = np.random.RandomState(SEED)
    n = min(max_per_class, len(aligned), len(deceptive))
    al_idx = rng.choice(len(aligned), n, replace=False)
    de_idx = rng.choice(len(deceptive), n, replace=False)

    texts = [aligned[i] for i in al_idx] + [deceptive[i] for i in de_idx]
    labels = np.array([1] * n + [0] * n)  # 1=honest, 0=deceptive
    return texts, labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def principal_angles(U, V):
    """Compute principal angles between subspaces spanned by columns of U and V.
    Uses SVD of U^T V (Björck & Golub method)."""
    # Orthonormalize
    Uq, _ = np.linalg.qr(U)
    Vq, _ = np.linalg.qr(V)
    # SVD of cross-correlation
    _, sigmas, _ = np.linalg.svd(Uq.T @ Vq)
    # Clip for numerical stability
    sigmas = np.clip(sigmas, -1, 1)
    angles = np.arccos(sigmas)
    return angles  # in radians


def bootstrap_ci(y_true, y_scores, metric_fn, n=N_BOOTSTRAP):
    scores = []
    rng = np.random.RandomState(SEED)
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_scores[idx]))
        except ValueError:
            continue
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ---------------------------------------------------------------------------
# Exp 3.1: Train both probes, measure cosine similarity
# ---------------------------------------------------------------------------
def exp3_1_cosine(truth_layer=16, decep_layer=26):
    """Train both probes and measure weight vector cosine similarity."""
    print("\n" + "=" * 60)
    print("  Exp 3.1: Probe Weight Cosine Similarity")
    print("=" * 60)

    # --- Truthfulness probe ---
    print("\n  Loading TruthfulQA...")
    t_texts, t_labels = load_truthfulqa()
    # Dev/test split
    t_idx = np.arange(len(t_texts))
    t_dev_idx, t_test_idx = train_test_split(t_idx, test_size=0.2, random_state=SEED, stratify=t_labels)
    t_texts_dev = [t_texts[i] for i in t_dev_idx]
    t_labels_dev = t_labels[t_dev_idx]
    t_texts_test = [t_texts[i] for i in t_test_idx]
    t_labels_test = t_labels[t_test_idx]

    print(f"  Extracting truthfulness activations at layer {truth_layer}...")
    X_t_dev = extract_activations(t_texts_dev, truth_layer)
    X_t_test = extract_activations(t_texts_test, truth_layer)

    scaler_t = StandardScaler()
    X_t_dev_s = scaler_t.fit_transform(X_t_dev)
    X_t_test_s = scaler_t.transform(X_t_test)

    probe_t = LogisticRegression(C=0.01, max_iter=1000, random_state=SEED)
    probe_t.fit(X_t_dev_s, t_labels_dev)
    t_auroc = roc_auc_score(t_labels_test, probe_t.predict_proba(X_t_test_s)[:, 1])
    print(f"  Truthfulness probe: AUROC={t_auroc:.4f} (layer {truth_layer})")

    # --- Deception probe ---
    print("\n  Loading deception data...")
    d_texts, d_labels = load_deception_data()
    d_idx = np.arange(len(d_texts))
    d_dev_idx, d_test_idx = train_test_split(d_idx, test_size=0.2, random_state=SEED, stratify=d_labels)
    d_texts_dev = [d_texts[i] for i in d_dev_idx]
    d_labels_dev = d_labels[d_dev_idx]
    d_texts_test = [d_texts[i] for i in d_test_idx]
    d_labels_test = d_labels[d_test_idx]

    print(f"  Extracting deception activations at layer {decep_layer}...")
    X_d_dev = extract_activations(d_texts_dev, decep_layer)
    X_d_test = extract_activations(d_texts_test, decep_layer)

    scaler_d = StandardScaler()
    X_d_dev_s = scaler_d.fit_transform(X_d_dev)
    X_d_test_s = scaler_d.transform(X_d_test)

    probe_d = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
    probe_d.fit(X_d_dev_s, d_labels_dev)
    d_auroc = roc_auc_score(d_labels_test, probe_d.predict_proba(X_d_test_s)[:, 1])
    print(f"  Deception probe: AUROC={d_auroc:.4f} (layer {decep_layer})")

    # --- Cosine similarity ---
    w_t = probe_t.coef_[0]  # (hidden_dim,)
    w_d = probe_d.coef_[0]  # (hidden_dim,)
    cos_sim = cosine_similarity(w_t, w_d)
    print(f"\n  cos(w_truth, w_decep) = {cos_sim:.6f}")
    print(f"  Angle: {np.degrees(np.arccos(np.clip(abs(cos_sim), 0, 1))):.1f}°")

    # Same-layer comparison (both at each layer)
    print("\n  Same-layer comparison:")
    same_layer_results = {}
    for layer in [truth_layer, decep_layer]:
        # Re-extract at the same layer for fair comparison
        if layer == truth_layer:
            X_t_l = X_t_dev_s
        else:
            print(f"  Extracting truthfulness activations at layer {layer}...")
            X_t_l_raw = extract_activations(t_texts_dev, layer)
            sc = StandardScaler()
            X_t_l = sc.fit_transform(X_t_l_raw)

        if layer == decep_layer:
            X_d_l = X_d_dev_s
        else:
            print(f"  Extracting deception activations at layer {layer}...")
            X_d_l_raw = extract_activations(d_texts_dev, layer)
            sc = StandardScaler()
            X_d_l = sc.fit_transform(X_d_l_raw)

        p_t = LogisticRegression(C=0.01, max_iter=1000, random_state=SEED)
        p_t.fit(X_t_l, t_labels_dev)
        p_d = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        p_d.fit(X_d_l, d_labels_dev)

        cos = cosine_similarity(p_t.coef_[0], p_d.coef_[0])
        same_layer_results[layer] = float(cos)
        print(f"    Layer {layer}: cos = {cos:.6f}")

    return {
        "truth_layer": truth_layer,
        "decep_layer": decep_layer,
        "truth_auroc": float(t_auroc),
        "decep_auroc": float(d_auroc),
        "cosine_similarity": float(cos_sim),
        "angle_degrees": float(np.degrees(np.arccos(np.clip(abs(cos_sim), 0, 1)))),
        "same_layer_cosine": same_layer_results,
        "w_t": w_t,
        "w_d": w_d,
        "probe_t": probe_t,
        "scaler_t": scaler_t,
        "probe_d": probe_d,
        "scaler_d": scaler_d,
        "t_texts_dev": t_texts_dev,
        "t_labels_dev": t_labels_dev,
        "t_texts_test": t_texts_test,
        "t_labels_test": t_labels_test,
        "d_texts_dev": d_texts_dev,
        "d_labels_dev": d_labels_dev,
        "d_texts_test": d_texts_test,
        "d_labels_test": d_labels_test,
        "X_t_dev": X_t_dev,
        "X_t_test": X_t_test,
        "X_d_dev": X_d_dev,
        "X_d_test": X_d_test,
    }


# ---------------------------------------------------------------------------
# Exp 3.2: Subspace analysis
# ---------------------------------------------------------------------------
def exp3_2_subspace(data):
    """PCA subspace analysis + principal angles."""
    print("\n" + "=" * 60)
    print("  Exp 3.2: Subspace Analysis")
    print("=" * 60)

    X_t = data["X_t_dev"]
    X_d = data["X_d_dev"]

    results = {}
    for k in [1, 2, 5, 10, 20, 50]:
        pca_t = PCA(n_components=k, random_state=SEED)
        pca_d = PCA(n_components=k, random_state=SEED)
        pca_t.fit(X_t)
        pca_d.fit(X_d)

        # Principal angles between top-k subspaces
        U = pca_t.components_.T  # (hidden_dim, k)
        V = pca_d.components_.T  # (hidden_dim, k)
        angles = principal_angles(U, V)

        mean_angle = float(np.mean(angles))
        min_angle = float(np.min(angles))
        max_angle = float(np.max(angles))

        # Variance explained
        var_t = float(sum(pca_t.explained_variance_ratio_))
        var_d = float(sum(pca_d.explained_variance_ratio_))

        results[k] = {
            "mean_angle_rad": mean_angle,
            "mean_angle_deg": float(np.degrees(mean_angle)),
            "min_angle_deg": float(np.degrees(min_angle)),
            "max_angle_deg": float(np.degrees(max_angle)),
            "variance_explained_truth": var_t,
            "variance_explained_decep": var_d,
        }
        print(f"  k={k:2d}: mean angle={np.degrees(mean_angle):.1f}°, "
              f"min={np.degrees(min_angle):.1f}°, max={np.degrees(max_angle):.1f}°, "
              f"var_t={var_t:.3f}, var_d={var_d:.3f}")

    # Projection overlap: project both datasets into each other's top-10 PCA space
    pca_t10 = PCA(n_components=10, random_state=SEED).fit(X_t)
    pca_d10 = PCA(n_components=10, random_state=SEED).fit(X_d)

    # How much variance of deception data is captured by truthfulness PCA?
    X_d_proj_t = pca_t10.transform(X_d)
    X_d_recon = pca_t10.inverse_transform(X_d_proj_t)
    recon_err_d_in_t = float(np.mean(np.sum((X_d - X_d_recon) ** 2, axis=1)))
    total_var_d = float(np.mean(np.sum(X_d ** 2, axis=1)))

    X_t_proj_d = pca_d10.transform(X_t)
    X_t_recon = pca_d10.inverse_transform(X_t_proj_d)
    recon_err_t_in_d = float(np.mean(np.sum((X_t - X_t_recon) ** 2, axis=1)))
    total_var_t = float(np.mean(np.sum(X_t ** 2, axis=1)))

    cross_capture = {
        "decep_var_captured_by_truth_pca10": 1 - recon_err_d_in_t / total_var_d if total_var_d > 0 else 0,
        "truth_var_captured_by_decep_pca10": 1 - recon_err_t_in_d / total_var_t if total_var_t > 0 else 0,
    }
    print(f"\n  Cross-capture (PCA-10):")
    print(f"    Deception variance in truthfulness subspace: {cross_capture['decep_var_captured_by_truth_pca10']:.4f}")
    print(f"    Truthfulness variance in deception subspace: {cross_capture['truth_var_captured_by_decep_pca10']:.4f}")

    return {"principal_angles": results, "cross_capture": cross_capture}


# ---------------------------------------------------------------------------
# Exp 3.3: Combined probe
# ---------------------------------------------------------------------------
def exp3_3_combined(data):
    """Combined 2D probe using both truthfulness and deception probe scores."""
    print("\n" + "=" * 60)
    print("  Exp 3.3: Combined Probe")
    print("=" * 60)

    probe_t = data["probe_t"]
    scaler_t = data["scaler_t"]
    probe_d = data["probe_d"]
    scaler_d = data["scaler_d"]

    # We need a common dataset with both truth and deception labels.
    # Use truthfulness test set and score with both probes.
    # For the combined probe test, we create a simple binary task:
    # "Is this text problematic?" (either hallucinated OR deceptive)

    # Score truthfulness data with both probes
    print("  Scoring truthfulness data with both probes...")
    X_t_dev_ts = scaler_t.transform(data["X_t_dev"])
    X_t_test_ts = scaler_t.transform(data["X_t_test"])
    t_dev_truth_score = probe_t.predict_proba(X_t_dev_ts)[:, 1]  # P(truthful)
    t_test_truth_score = probe_t.predict_proba(X_t_test_ts)[:, 1]

    # For deception scores on truthfulness data, we need deception-layer activations
    truth_layer = data["truth_layer"]
    decep_layer = data["decep_layer"]

    if truth_layer != decep_layer:
        print(f"  Extracting truthfulness data at deception layer {decep_layer}...")
        X_t_dev_dl = extract_activations(data["t_texts_dev"], decep_layer)
        X_t_test_dl = extract_activations(data["t_texts_test"], decep_layer)
    else:
        X_t_dev_dl = data["X_t_dev"]
        X_t_test_dl = data["X_t_test"]

    X_t_dev_ds = scaler_d.transform(X_t_dev_dl)
    X_t_test_ds = scaler_d.transform(X_t_test_dl)
    t_dev_decep_score = probe_d.predict_proba(X_t_dev_ds)[:, 1]  # P(honest)
    t_test_decep_score = probe_d.predict_proba(X_t_test_ds)[:, 1]

    # Score deception data with both probes
    print("  Scoring deception data with both probes...")
    if truth_layer != decep_layer:
        print(f"  Extracting deception data at truthfulness layer {truth_layer}...")
        X_d_dev_tl = extract_activations(data["d_texts_dev"], truth_layer)
        X_d_test_tl = extract_activations(data["d_texts_test"], truth_layer)
    else:
        X_d_dev_tl = data["X_d_dev"]
        X_d_test_tl = data["X_d_test"]

    X_d_dev_ts2 = scaler_t.transform(X_d_dev_tl)
    X_d_test_ts2 = scaler_t.transform(X_d_test_tl)
    d_dev_truth_score = probe_t.predict_proba(X_d_dev_ts2)[:, 1]
    d_test_truth_score = probe_t.predict_proba(X_d_test_ts2)[:, 1]

    X_d_dev_ds2 = scaler_d.transform(data["X_d_dev"])
    X_d_test_ds2 = scaler_d.transform(data["X_d_test"])
    d_dev_decep_score = probe_d.predict_proba(X_d_dev_ds2)[:, 1]
    d_test_decep_score = probe_d.predict_proba(X_d_test_ds2)[:, 1]

    # Individual probe AUROCs on each task
    results = {}

    # Truthfulness probe on truthfulness task
    t_on_t = roc_auc_score(data["t_labels_test"], t_test_truth_score)
    # Deception probe on deception task
    d_on_d = roc_auc_score(data["d_labels_test"], d_test_decep_score)
    # Cross: truthfulness probe on deception task
    t_on_d = roc_auc_score(data["d_labels_test"], d_test_truth_score)
    # Cross: deception probe on truthfulness task
    d_on_t = roc_auc_score(data["t_labels_test"], t_test_decep_score)

    results["individual"] = {
        "truth_probe_on_truth_task": float(t_on_t),
        "decep_probe_on_decep_task": float(d_on_d),
        "truth_probe_on_decep_task": float(t_on_d),
        "decep_probe_on_truth_task": float(d_on_t),
    }
    print(f"\n  Cross-task AUROC:")
    print(f"    Truth probe → truth task: {t_on_t:.4f}")
    print(f"    Decep probe → decep task: {d_on_d:.4f}")
    print(f"    Truth probe → decep task: {t_on_d:.4f} (cross)")
    print(f"    Decep probe → truth task: {d_on_t:.4f} (cross)")

    # Combined 2D probe: [truth_score, decep_score] → binary
    # Task: detect "problematic" text (hallucinated OR deceptive)
    # Build combined dataset: truthfulness incorrect + deception deceptive = positive
    #                         truthfulness correct + deception honest = negative

    # For truthfulness data
    t_dev_2d = np.column_stack([t_dev_truth_score, t_dev_decep_score])
    t_test_2d = np.column_stack([t_test_truth_score, t_test_decep_score])

    # For deception data
    d_dev_2d = np.column_stack([d_dev_truth_score, d_dev_decep_score])
    d_test_2d = np.column_stack([d_test_truth_score, d_test_decep_score])

    # Combined: truthfulness labels (1=correct=good, 0=incorrect=bad)
    # Deception labels (1=honest=good, 0=deceptive=bad)
    # Unified: 1=good, 0=bad
    X_dev_combined = np.vstack([t_dev_2d, d_dev_2d])
    y_dev_combined = np.concatenate([data["t_labels_dev"], data["d_labels_dev"]])
    X_test_combined = np.vstack([t_test_2d, d_test_2d])
    y_test_combined = np.concatenate([data["t_labels_test"], data["d_labels_test"]])

    # Train combined probe
    combined_probe = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    combined_probe.fit(X_dev_combined, y_dev_combined)

    y_prob_combined = combined_probe.predict_proba(X_test_combined)[:, 1]
    combined_auroc = roc_auc_score(y_test_combined, y_prob_combined)
    combined_acc = accuracy_score(y_test_combined, combined_probe.predict(X_test_combined))

    # Bootstrap CI
    cm, clo, chi = bootstrap_ci(y_test_combined, y_prob_combined, roc_auc_score)

    # Compare: truth-only and decep-only on the combined task
    # Truth-only score on combined test
    y_truth_only = np.concatenate([t_test_truth_score, d_test_truth_score])
    truth_only_auroc = roc_auc_score(y_test_combined, y_truth_only)

    y_decep_only = np.concatenate([t_test_decep_score, d_test_decep_score])
    decep_only_auroc = roc_auc_score(y_test_combined, y_decep_only)

    results["combined"] = {
        "auroc": float(combined_auroc),
        "accuracy": float(combined_acc),
        "bootstrap": {"mean": cm, "ci_lo": clo, "ci_hi": chi},
        "truth_only_auroc": float(truth_only_auroc),
        "decep_only_auroc": float(decep_only_auroc),
        "combined_weights": combined_probe.coef_[0].tolist(),
        "combined_intercept": float(combined_probe.intercept_[0]),
    }

    print(f"\n  Combined task (detect any problem):")
    print(f"    Truth-only AUROC:  {truth_only_auroc:.4f}")
    print(f"    Decep-only AUROC:  {decep_only_auroc:.4f}")
    print(f"    Combined AUROC:    {combined_auroc:.4f} [{clo:.4f}, {chi:.4f}]")
    print(f"    Combined weights:  truth={combined_probe.coef_[0][0]:.4f}, decep={combined_probe.coef_[0][1]:.4f}")

    # Score distribution summary
    print(f"\n  Score distributions:")
    print(f"    Truth data - truth score: mean={t_test_truth_score.mean():.3f}, std={t_test_truth_score.std():.3f}")
    print(f"    Truth data - decep score: mean={t_test_decep_score.mean():.3f}, std={t_test_decep_score.std():.3f}")
    print(f"    Decep data - truth score: mean={d_test_truth_score.mean():.3f}, std={d_test_truth_score.std():.3f}")
    print(f"    Decep data - decep score: mean={d_test_decep_score.mean():.3f}, std={d_test_decep_score.std():.3f}")

    results["score_distributions"] = {
        "truth_data_truth_score": {"mean": float(t_test_truth_score.mean()), "std": float(t_test_truth_score.std())},
        "truth_data_decep_score": {"mean": float(t_test_decep_score.mean()), "std": float(t_test_decep_score.std())},
        "decep_data_truth_score": {"mean": float(d_test_truth_score.mean()), "std": float(d_test_truth_score.std())},
        "decep_data_decep_score": {"mean": float(d_test_decep_score.mean()), "std": float(d_test_decep_score.std())},
    }

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(cosine_data, subspace_data, combined_data, elapsed):
    cos = cosine_data["cosine_similarity"]
    angle = cosine_data["angle_degrees"]

    lines = [
        "# Experiment 3: Orthogonality Verification",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Runtime: {elapsed:.0f}s*",
        "",
        "## Setup",
        "",
        f"- **Model:** {MODEL_ID} ({NUM_LAYERS} layers)",
        f"- **Truthfulness probe:** layer {cosine_data['truth_layer']}, C=0.01 (from Exp 1)",
        f"- **Deception probe:** layer {cosine_data['decep_layer']}, C=0.1 (from Exp 2)",
        f"- **Protocol:** Dev/test split upfront; probes trained on dev, evaluated on test",
        "",
        "## 3.1 Cosine Similarity",
        "",
        f"- **cos(w_truth, w_decep) = {cos:.6f}**",
        f"- **Angle: {angle:.1f}°**",
        f"- Truthfulness probe test AUROC: {cosine_data['truth_auroc']:.4f}",
        f"- Deception probe test AUROC: {cosine_data['decep_auroc']:.4f}",
        "",
    ]

    if cosine_data.get("same_layer_cosine"):
        lines += [
            "### Same-layer comparison",
            "",
            "| Layer | cos(w_t, w_d) |",
            "|-------|---------------|",
        ]
        for l, c in sorted(cosine_data["same_layer_cosine"].items()):
            lines.append(f"| {l} | {c:.6f} |")
        lines.append("")

    if abs(cos) < 0.15:
        lines.append(f"**PASS:** Probes are near-orthogonal (cos={cos:.4f} < 0.15 threshold).")
    else:
        lines.append(f"**FAIL:** Probes share significant direction (cos={cos:.4f} >= 0.15).")
    lines.append("")

    # Subspace analysis
    lines += [
        "## 3.2 Subspace Analysis (Principal Angles)",
        "",
        "| k (PCA dims) | Mean Angle | Min Angle | Max Angle | Var Truth | Var Decep |",
        "|-------------|------------|-----------|-----------|-----------|-----------|",
    ]
    for k in sorted(subspace_data["principal_angles"].keys()):
        r = subspace_data["principal_angles"][k]
        lines.append(
            f"| {k} | {r['mean_angle_deg']:.1f}° | {r['min_angle_deg']:.1f}° | "
            f"{r['max_angle_deg']:.1f}° | {r['variance_explained_truth']:.3f} | "
            f"{r['variance_explained_decep']:.3f} |"
        )

    cc = subspace_data["cross_capture"]
    lines += [
        "",
        "### Cross-Subspace Capture (PCA-10)",
        "",
        f"- Deception variance captured by truthfulness PCA-10: {cc['decep_var_captured_by_truth_pca10']:.4f}",
        f"- Truthfulness variance captured by deception PCA-10: {cc['truth_var_captured_by_decep_pca10']:.4f}",
        "",
    ]

    # Combined probe
    ind = combined_data["individual"]
    comb = combined_data["combined"]
    lines += [
        "## 3.3 Combined Probe",
        "",
        "### Cross-Task AUROC",
        "",
        "| Probe | Task | AUROC |",
        "|-------|------|-------|",
        f"| Truthfulness | Truthfulness | {ind['truth_probe_on_truth_task']:.4f} |",
        f"| Deception | Deception | {ind['decep_probe_on_decep_task']:.4f} |",
        f"| Truthfulness | Deception | {ind['truth_probe_on_decep_task']:.4f} (cross) |",
        f"| Deception | Truthfulness | {ind['decep_probe_on_truth_task']:.4f} (cross) |",
        "",
        "### Combined 2D Probe (detect any problem)",
        "",
        "| Method | AUROC |",
        "|--------|-------|",
        f"| Truth score only | {comb['truth_only_auroc']:.4f} |",
        f"| Decep score only | {comb['decep_only_auroc']:.4f} |",
        f"| **Combined [truth, decep]** | **{comb['auroc']:.4f}** [{comb['bootstrap']['ci_lo']:.4f}, {comb['bootstrap']['ci_hi']:.4f}] |",
        "",
        f"Combined probe weights: truth={comb['combined_weights'][0]:.4f}, decep={comb['combined_weights'][1]:.4f}",
        "",
    ]

    # Score distributions
    sd = combined_data["score_distributions"]
    lines += [
        "### Score Distributions",
        "",
        "| Dataset | Truth Score (mean±std) | Decep Score (mean±std) |",
        "|---------|----------------------|----------------------|",
        f"| Truthfulness data | {sd['truth_data_truth_score']['mean']:.3f}±{sd['truth_data_truth_score']['std']:.3f} | {sd['truth_data_decep_score']['mean']:.3f}±{sd['truth_data_decep_score']['std']:.3f} |",
        f"| Deception data | {sd['decep_data_truth_score']['mean']:.3f}±{sd['decep_data_truth_score']['std']:.3f} | {sd['decep_data_decep_score']['mean']:.3f}±{sd['decep_data_decep_score']['std']:.3f} |",
        "",
    ]

    # Interpretation
    lines += ["## Interpretation", ""]

    if abs(cos) < 0.15:
        lines.append(
            f"The truthfulness and deception probes are **near-orthogonal** "
            f"(cos={cos:.4f}, angle={angle:.1f}°), consistent with the lambda_results "
            f"finding of 99.6% orthogonal subspaces."
        )
    else:
        lines.append(
            f"The probes share more direction than expected (cos={cos:.4f}). "
            f"This may reflect shared format features rather than shared signal."
        )

    cross_t_on_d = ind["truth_probe_on_decep_task"]
    cross_d_on_t = ind["decep_probe_on_truth_task"]
    if cross_t_on_d < 0.6 and cross_d_on_t < 0.6:
        lines.append(
            "Cross-task performance is near-chance, confirming the probes detect "
            "genuinely different phenomena."
        )
    elif cross_t_on_d > 0.75 or cross_d_on_t > 0.75:
        lines.append(
            "**Warning:** Significant cross-task transfer detected — the probes may "
            "share a common signal component (possibly format/style rather than content)."
        )

    if comb["auroc"] > max(comb["truth_only_auroc"], comb["decep_only_auroc"]) + 0.02:
        lines.append(
            f"\nThe combined probe ({comb['auroc']:.3f}) outperforms either individual probe "
            f"(truth={comb['truth_only_auroc']:.3f}, decep={comb['decep_only_auroc']:.3f}), "
            f"confirming additive value from orthogonal signals."
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("  Experiment 3: Orthogonality Verification")
    print("=" * 60)

    # Use best layers from Exp 1 (layer 16) and Exp 2 (layer 26)
    cosine_data = exp3_1_cosine(truth_layer=16, decep_layer=26)
    subspace_data = exp3_2_subspace(cosine_data)
    combined_data = exp3_3_combined(cosine_data)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save JSON (remove non-serializable objects)
    json_results = {
        "model": MODEL_ID,
        "num_layers": NUM_LAYERS,
        "cosine_similarity": cosine_data["cosine_similarity"],
        "angle_degrees": cosine_data["angle_degrees"],
        "truth_layer": cosine_data["truth_layer"],
        "decep_layer": cosine_data["decep_layer"],
        "truth_auroc": cosine_data["truth_auroc"],
        "decep_auroc": cosine_data["decep_auroc"],
        "same_layer_cosine": {str(k): v for k, v in cosine_data["same_layer_cosine"].items()},
        "subspace_analysis": {
            str(k): v for k, v in subspace_data["principal_angles"].items()
        },
        "cross_capture": subspace_data["cross_capture"],
        "combined_probe": combined_data["combined"],
        "cross_task": combined_data["individual"],
        "score_distributions": combined_data["score_distributions"],
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    OUTPUT_JSON.write_text(json.dumps(json_results, indent=2))
    print(f"Saved: {OUTPUT_JSON}")

    # Save probe weights for downstream use
    with open(REPO_ROOT / "results" / "exp3_probes.pkl", "wb") as f:
        pickle.dump({
            "probe_t": cosine_data["probe_t"],
            "scaler_t": cosine_data["scaler_t"],
            "truth_layer": cosine_data["truth_layer"],
            "probe_d": cosine_data["probe_d"],
            "scaler_d": cosine_data["scaler_d"],
            "decep_layer": cosine_data["decep_layer"],
        }, f)

    report = generate_report(cosine_data, subspace_data, combined_data, elapsed)
    OUTPUT_MD.write_text(report)
    print(f"Saved: {OUTPUT_MD}")
    print("Done!")


if __name__ == "__main__":
    main()
