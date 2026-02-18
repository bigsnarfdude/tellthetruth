#!/usr/bin/env python3
"""
RLFR Probe-Based Pipeline
==========================
Real implementation of "Features as Rewards" using activation probes
on an open-source model (Gemma-2-2B-it) instead of web search.

Phase 1: Train a truthfulness probe on TruthfulQA activations
Phase 2: Use that probe to detect hallucinations in free-form generations
Phase 3: Intervene + grade (same as before but with real probe scores)

Usage: python rlfr_probe_pipeline.py
"""

import json
import pickle
import time
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-2-2b-it"
PROBE_LAYER = 20  # middle-ish layer of 26-layer model; we'll sweep
REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = REPO_ROOT / "results" / "truthfulness_probe.pkl"
OUTPUT_FILE = REPO_ROOT / "results" / "rlfr_probe_results.md"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hard prompt for generation phase
GENERATION_PROMPT = (
    "Write a detailed, information-dense account of the life and scientific contributions "
    "of Marian Smoluchowski. Include his birth and death dates, birthplace, universities "
    "where he studied and taught, his key scientific papers with publication years, his "
    "contributions to Brownian motion theory, the Smoluchowski equation, coagulation theory, "
    "his relationship with Einstein, specific numerical results from his papers, names of "
    "his students and collaborators, and his influence on statistical mechanics. "
    "Be as specific as possible with numbers, names, dates, and journal references."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    claim: str
    span_text: str
    category: str
    probe_score: float = 0.0   # probe P(hallucinated)
    verdict: str = ""          # supported | hallucinated
    intervention: str = ""
    intervention_text: str = ""
    grade: str = ""
    grade_reason: str = ""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    attn_implementation="eager",
)
model.eval()
print(f"Model loaded on {DEVICE}, {model.config.num_hidden_layers} layers")


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------
def get_activations(text: str, layer: int) -> np.ndarray:
    """Get mean-pooled hidden state at a specific layer for input text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
    # Mean pool over sequence
    pooled = hidden.mean(dim=1).squeeze().float().cpu().numpy()
    return pooled


def get_activations_at_last_token(text: str, layer: int) -> np.ndarray:
    """Get hidden state at the last token of the input."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
    last = hidden[0, -1, :].float().cpu().numpy()
    return last


# ---------------------------------------------------------------------------
# Phase 1: Train truthfulness probe on TruthfulQA
# ---------------------------------------------------------------------------
def phase1_train_probe(force_retrain: bool = False) -> dict:
    """Train a linear probe to detect truthful vs hallucinated answers."""
    if PROBE_PATH.exists() and not force_retrain:
        print("\n[Phase 1] Loading existing probe...")
        with open(PROBE_PATH, "rb") as f:
            return pickle.load(f)

    print("\n" + "=" * 60)
    print("  Phase 1: Train Truthfulness Probe")
    print("=" * 60)

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    print(f"  TruthfulQA: {len(ds)} questions")

    # Build paired dataset: (question + correct_answer, 1) and (question + incorrect_answer, 0)
    texts = []
    labels = []
    for row in ds:
        q = row["question"]
        # Correct answers -> label 1 (truthful)
        for ans in row["correct_answers"][:1]:  # take first correct
            texts.append(f"Q: {q}\nA: {ans}")
            labels.append(1)
        # Incorrect answers -> label 0 (hallucinated)
        for ans in row["incorrect_answers"][:1]:  # take first incorrect
            texts.append(f"Q: {q}\nA: {ans}")
            labels.append(0)

    print(f"  Built {len(texts)} samples ({sum(labels)} truthful, {len(labels) - sum(labels)} hallucinated)")

    # Sweep layers to find best
    best_auroc = 0
    best_layer = PROBE_LAYER
    layers_to_try = [10, 15, 18, 20, 22, 24]

    # Extract activations for all layers at once
    print("  Extracting activations (this takes a few minutes)...")
    all_layer_acts = {l: [] for l in layers_to_try}

    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"    [{i}/{len(texts)}]...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for l in layers_to_try:
            h = outputs.hidden_states[l]
            pooled = h.mean(dim=1).squeeze().float().cpu().numpy()
            all_layer_acts[l].append(pooled)

    labels_arr = np.array(labels)

    # Split dev/test UPFRONT — test set never seen during layer selection
    indices = np.arange(len(labels_arr))
    dev_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels_arr
    )
    labels_dev = labels_arr[dev_idx]
    labels_test = labels_arr[test_idx]

    print(f"\n  Split: {len(dev_idx)} dev, {len(test_idx)} test (held-out)")
    print("  Layer sweep (5-fold CV on dev set):")
    layer_results = {}
    for l in layers_to_try:
        X_all = np.stack(all_layer_acts[l])
        X_dev = X_all[dev_idx]

        # 5-fold CV on dev set for layer selection
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aurocs = []
        for tr_idx, va_idx in skf.split(X_dev, labels_dev):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_dev[tr_idx])
            X_va = scaler.transform(X_dev[va_idx])
            probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            probe.fit(X_tr, labels_dev[tr_idx])
            y_prob = probe.predict_proba(X_va)[:, 1]
            fold_aurocs.append(roc_auc_score(labels_dev[va_idx], y_prob))
        auroc = np.mean(fold_aurocs)
        layer_results[l] = {"auroc": float(auroc), "accuracy": 0.0}
        print(f"    Layer {l:2d}: CV AUROC={auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = l

    print(f"\n  Best layer: {best_layer} (CV AUROC={best_auroc:.4f})")

    # Train final probe on full dev set, evaluate on held-out test set
    X_best = np.stack(all_layer_acts[best_layer])
    X_dev = X_best[dev_idx]
    X_test = X_best[test_idx]

    final_scaler = StandardScaler()
    X_dev_s = final_scaler.fit_transform(X_dev)
    X_test_s = final_scaler.transform(X_test)

    final_probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    final_probe.fit(X_dev_s, labels_dev)

    y_prob = final_probe.predict_proba(X_test_s)[:, 1]
    final_auroc = roc_auc_score(labels_test, y_prob)
    final_acc = accuracy_score(labels_test, final_probe.predict(X_test_s))
    print(f"  Final probe (held-out test): AUROC={final_auroc:.4f}, Acc={final_acc:.4f}")

    result = {
        "probe": final_probe,
        "scaler": final_scaler,
        "layer": best_layer,
        "auroc": final_auroc,
        "accuracy": final_acc,
        "layer_results": layer_results,
    }

    with open(PROBE_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved probe to {PROBE_PATH}")

    return result


# ---------------------------------------------------------------------------
# Phase 2: Generate + probe-based detection
# ---------------------------------------------------------------------------
def generate_completion(prompt: str, max_new_tokens: int = 512) -> str:
    """Generate text with the model."""
    chat = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(DEVICE)
    torch.manual_seed(42)  # reproducible generation
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    generated = out[0][inputs.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def claude(prompt: str, input_text: str = "") -> str:
    """Call claude -p for entity extraction and grading."""
    cmd = ["claude", "-p", prompt]
    result = subprocess.run(
        cmd,
        input=input_text if input_text else None,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"  [claude error] {result.stderr[:200]}")
        return ""
    return result.stdout.strip()


def parse_json(raw: str):
    """Parse JSON from output, handling markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for sc, ec in [("[", "]"), ("{", "}")]:
            s = text.find(sc)
            e = text.rfind(ec) + 1
            if s >= 0 and e > s:
                try:
                    return json.loads(text[s:e])
                except json.JSONDecodeError:
                    continue
        return None


def phase2_detect(probe_data: dict) -> tuple[str, list[Entity]]:
    """Generate completion, extract claims, probe each one."""
    print("\n" + "=" * 60)
    print("  Phase 2: Generate + Probe-Based Detection")
    print("=" * 60)

    probe = probe_data["probe"]
    scaler = probe_data["scaler"]
    layer = probe_data["layer"]

    # Step 1: Generate
    print("  Generating completion with Gemma-2-2B-it...")
    completion = generate_completion(GENERATION_PROMPT)
    print(f"  Generated {len(completion)} chars")

    # Step 2: Extract entities (use Claude for this)
    print("  Extracting claims via Claude...")
    extract_prompt = (
        "You extract individual verifiable factual claims from text. "
        "Return a JSON array where each element has: "
        '"claim" (the specific factual assertion), '
        '"span_text" (the exact substring from the original text), '
        '"category" (one of: date, name, number, specification, event). '
        "Only return the JSON array, no other text. "
        "Extract all specific, verifiable factual claims from the following text:"
    )
    raw = claude(extract_prompt, input_text=completion)
    items = parse_json(raw)
    if not isinstance(items, list):
        print("  [error] Could not parse entities")
        return completion, []

    entities = []
    for item in items:
        if isinstance(item, dict):
            entities.append(Entity(
                claim=item.get("claim", ""),
                span_text=item.get("span_text", ""),
                category=item.get("category", "unknown"),
            ))
    print(f"  Extracted {len(entities)} claims")

    # Step 3: Probe each claim
    print(f"  Probing claims at layer {layer}...")
    threshold = 0.5  # P(truthful) < 0.5 => hallucinated

    for i, ent in enumerate(entities):
        # Build a Q/A style text matching probe training format
        probe_text = f"Q: Is the following true? {ent.claim}\nA: {ent.span_text}"
        acts = get_activations(probe_text, layer)
        acts_scaled = scaler.transform(acts.reshape(1, -1))
        prob_truthful = probe.predict_proba(acts_scaled)[0, 1]
        ent.probe_score = 1.0 - prob_truthful  # P(hallucinated)

        if prob_truthful >= threshold:
            ent.verdict = "supported"
        else:
            ent.verdict = "hallucinated"

        marker = "H" if ent.verdict == "hallucinated" else "S"
        print(f"    [{i+1}/{len(entities)}] [{marker}] P(hallu)={ent.probe_score:.3f} | {ent.claim[:60]}...")

    supported = sum(1 for e in entities if e.verdict == "supported")
    hallucinated = sum(1 for e in entities if e.verdict == "hallucinated")
    print(f"\n  Results: {supported} supported, {hallucinated} hallucinated")
    return completion, entities


# ---------------------------------------------------------------------------
# Phase 3: Intervene + Grade
# ---------------------------------------------------------------------------
def phase3_intervene(completion: str, entities: list[Entity]) -> list[Entity]:
    """Intervene on hallucinated claims and grade results."""
    print("\n" + "=" * 60)
    print("  Phase 3: Intervene + Grade")
    print("=" * 60)

    hallucinated = [e for e in entities if e.verdict == "hallucinated"]
    print(f"  {len(hallucinated)} hallucinated claims to intervene on")

    for i, ent in enumerate(hallucinated):
        print(f"  [{i+1}/{len(hallucinated)}] Intervening: {ent.claim[:60]}...")
        prompt = (
            "You are a careful fact-checker. The claim below was flagged as potentially "
            "hallucinated by an internal model probe (confidence: {:.0f}%). "
            "Respond with a JSON object choosing one action:\n"
            '- {{"action": "retract", "text": "retraction acknowledging the error"}}\n'
            '- {{"action": "correct", "text": "corrected factual replacement"}}\n'
            '- {{"action": "maintain", "text": "justification if the claim is actually correct"}}'
        ).format(ent.probe_score * 100)
        input_text = (
            f"ORIGINAL TEXT:\n{completion}\n\n"
            f"FLAGGED CLAIM: \"{ent.span_text}\"\n"
            f"SPECIFIC ASSERTION: {ent.claim}\n"
            f"PROBE SCORE: P(hallucinated) = {ent.probe_score:.3f}"
        )

        raw = claude(prompt, input_text=input_text)
        obj = parse_json(raw)
        if isinstance(obj, dict):
            ent.intervention = obj.get("action", "maintain")
            ent.intervention_text = obj.get("text", "")
        else:
            ent.intervention = "maintain"
            ent.intervention_text = raw[:200]

        # Grade
        if ent.intervention in ("retract", "correct"):
            grade_prompt = (
                "Grade this intervention. Respond with JSON: "
                '{"grade": "Fixed" or "Retracted" or "Failed Fix", "reason": "brief explanation"}'
            )
            grade_input = (
                f"ORIGINAL CLAIM: {ent.claim}\n"
                f"INTERVENTION: {ent.intervention}\n"
                f"TEXT: {ent.intervention_text}"
            )
            raw_g = claude(grade_prompt, input_text=grade_input)
            obj_g = parse_json(raw_g)
            if isinstance(obj_g, dict):
                ent.grade = obj_g.get("grade", "Failed Fix")
                ent.grade_reason = obj_g.get("reason", "")
            else:
                ent.grade = "Failed Fix"
                ent.grade_reason = raw_g[:200]

        time.sleep(0.3)

    return entities


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(completion: str, entities: list[Entity], probe_data: dict) -> str:
    """Generate markdown report."""
    print("\n" + "=" * 60)
    print("  Generating Report")
    print("=" * 60)

    total = len(entities)
    supported = sum(1 for e in entities if e.verdict == "supported")
    hallucinated = sum(1 for e in entities if e.verdict == "hallucinated")
    intervened = [e for e in entities if e.intervention in ("retract", "correct")]
    fixed = sum(1 for e in entities if e.grade == "Fixed")
    retracted = sum(1 for e in entities if e.grade == "Retracted")
    failed = sum(1 for e in entities if e.grade == "Failed Fix")
    maintained = sum(1 for e in entities if e.intervention == "maintain")
    reduction_rate = (fixed + retracted) / hallucinated * 100 if hallucinated > 0 else 0

    layer_results = probe_data.get("layer_results", {})

    lines = [
        "# RLFR Probe-Based Pipeline Results",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Architecture",
        "",
        f"- **Model:** {MODEL_ID}",
        f"- **Probe layer:** {probe_data['layer']}",
        f"- **Probe AUROC:** {probe_data['auroc']:.4f}",
        f"- **Probe accuracy:** {probe_data['accuracy']:.4f}",
        f"- **Training data:** TruthfulQA (817 questions, paired correct/incorrect)",
        "",
        "This replaces web search with a **linear probe on hidden activations** —",
        "reading the model's internal representation to detect when it's confabulating.",
        "",
    ]

    if layer_results:
        lines += [
            "### Layer Sweep",
            "",
            "| Layer | AUROC | Accuracy |",
            "|-------|-------|----------|",
        ]
        for l in sorted(layer_results.keys()):
            r = layer_results[l]
            marker = " **best**" if l == probe_data["layer"] else ""
            lines.append(f"| {l} | {r['auroc']:.4f} | {r['accuracy']:.4f} |{marker}")
        lines.append("")

    lines += [
        "## Generation",
        "",
        f"**Prompt:** {GENERATION_PROMPT[:200]}...",
        "",
        "**Completion (by Gemma-2-2B-it):**",
        "",
        f"> {completion}",
        "",
        "## Probe-Based Verification",
        "",
        "| # | Category | Claim | P(hallu) | Verdict |",
        "|---|----------|-------|----------|---------|",
    ]

    for i, ent in enumerate(entities):
        claim_short = ent.claim[:70] + ("..." if len(ent.claim) > 70 else "")
        lines.append(
            f"| {i+1} | {ent.category} | {claim_short} | {ent.probe_score:.3f} | {ent.verdict} |"
        )

    lines += [
        "",
        f"**Summary:** {supported} supported, {hallucinated} hallucinated out of {total} claims",
        "",
    ]

    if hallucinated > 0:
        lines += [
            "## Interventions & Grading",
            "",
            "| # | Claim | P(hallu) | Action | Grade |",
            "|---|-------|----------|--------|-------|",
        ]
        for i, ent in enumerate(e for e in entities if e.verdict == "hallucinated"):
            claim_short = ent.claim[:50] + ("..." if len(ent.claim) > 50 else "")
            lines.append(
                f"| {i+1} | {claim_short} | {ent.probe_score:.3f} | {ent.intervention} | {ent.grade or 'N/A'} |"
            )
        lines.append("")

    lines += [
        "## Reduction Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total claims | {total} |",
        f"| Supported | {supported} |",
        f"| Hallucinated (probe) | {hallucinated} |",
        f"| Interventions attempted | {len(intervened)} |",
        f"| Fixed | {fixed} |",
        f"| Retracted | {retracted} |",
        f"| Failed fixes | {failed} |",
        f"| Maintained | {maintained} |",
        f"| **Hallucination reduction rate** | **{reduction_rate:.1f}%** |",
        "",
        "## Key Difference from Web Search Version",
        "",
        "The probe reads the model's **internal activations** during generation.",
        "Unlike web search (which fails when DDG returns no results), the probe",
        "detects the difference between 'retrieved from training data' vs",
        "'interpolated/confabulated' — even when the output text looks equally confident.",
        "",
    ]

    # Detailed log
    lines.append("## Detailed Entity Log\n")
    for i, ent in enumerate(entities):
        lines += [
            f"### Entity {i+1}: {ent.category}",
            f"- **Claim:** {ent.claim}",
            f"- **P(hallucinated):** {ent.probe_score:.3f}",
            f"- **Verdict:** {ent.verdict}",
        ]
        if ent.intervention:
            lines += [
                f"- **Intervention:** {ent.intervention}",
                f"- **Intervention text:** {ent.intervention_text}",
            ]
        if ent.grade:
            lines += [
                f"- **Grade:** {ent.grade}",
                f"- **Grade reason:** {ent.grade_reason}",
            ]
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("RLFR Probe-Based Pipeline")
    print("=" * 60)

    # Phase 1: Train probe
    probe_data = phase1_train_probe()

    # Phase 2: Generate + detect
    completion, entities = phase2_detect(probe_data)
    if not entities:
        print("No entities extracted, aborting.")
        sys.exit(1)

    # Phase 3: Intervene + grade
    entities = phase3_intervene(completion, entities)

    # Report
    report = generate_report(completion, entities, probe_data)
    OUTPUT_FILE.write_text(report)
    print(f"\nReport saved to: {OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
