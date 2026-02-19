#!/usr/bin/env python3
"""
Standalone activation extraction for exp9.
Reads completions JSON files, extracts per-token hidden states at layers 20 and 30,
saves as pickle. Needs 40GB A100 (12B model + hidden states in memory).

Usage: python exp9_extract_activations.py
Reads: results/exp9_completions_box2.json + exp9_completions_500.json
Writes: results/exp9_activations.pkl
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-12b-it"
DEVICE = "cuda"
LOC_LAYER = 20
CLS_LAYERS = [20, 30]
ALL_LAYERS = sorted(set([LOC_LAYER] + CLS_LAYERS))  # [20, 30]

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "results" / "exp9_activations.pkl"

def load_all_completions():
    """Load and merge all completion files."""
    all_completions = []
    for fname in ["exp9_completions_box2.json", "exp9_completions_500.json", "exp9_completions.json"]:
        path = REPO_ROOT / "results" / fname
        if path.exists():
            data = json.loads(path.read_text())
            print(f"  Loaded {len(data)} from {fname}")
            # Re-id to avoid collisions
            for d in data:
                d["source"] = fname
            all_completions.extend(data)
    # Deduplicate by (prompt_idx, seed)
    seen = set()
    unique = []
    for c in all_completions:
        key = (c["prompt_idx"], c["seed"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    # Re-assign sequential IDs
    for i, c in enumerate(unique):
        c["id"] = i
    print(f"  Total unique completions: {len(unique)}")
    return unique


def extract_per_token_activations(text, tokenizer, model, layers, max_len=512):
    """Extract per-token hidden states at specified layers."""
    encoding = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=max_len, return_offsets_mapping=True,
    )
    offsets = encoding.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(DEVICE) for k, v in encoding.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    result = {}
    for layer in layers:
        h = out.hidden_states[layer][0].float().cpu().numpy()
        if not np.isfinite(h).all():
            n_bad = (~np.isfinite(h)).sum()
            print(f"    [WARN] Layer {layer}: {n_bad} non-finite values!")
        result[layer] = h
    return result, offsets


def main():
    if OUTPUT_FILE.exists():
        print(f"Output already exists: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"  Contains {len(data)} completions")
        return

    completions = load_all_completions()
    if not completions:
        print("ERROR: No completions found!")
        return

    print(f"\nLoading model for activation extraction...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE,
        attn_implementation="sdpa",
    )
    model.eval()
    _cfg = getattr(model.config, "text_config", model.config)
    print(f"Loaded {MODEL_ID}: {_cfg.num_hidden_layers}L, hidden={_cfg.hidden_size} ({time.time()-t0:.0f}s)")

    print(f"\nExtracting activations at layers {ALL_LAYERS} for {len(completions)} completions...")
    activation_data = {}
    t1 = time.time()

    for i, comp in enumerate(completions):
        cid = comp["id"]
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(completions) - i - 1) / max(rate, 0.01)
            print(f"  [{i+1}/{len(completions)}] {elapsed:.0f}s elapsed, {rate:.1f}/s, ETA {eta:.0f}s", flush=True)

        acts, offsets = extract_per_token_activations(comp["text"], tokenizer, model, ALL_LAYERS)
        activation_data[cid] = {"activations": acts, "offsets": offsets}

        # Checkpoint every 100
        if (i + 1) % 100 == 0:
            tmp = REPO_ROOT / "results" / "exp9_activations_partial.pkl"
            with open(tmp, "wb") as f:
                pickle.dump(activation_data, f)
            print(f"  [checkpoint] {len(activation_data)} saved")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(activation_data, f)

    total_time = time.time() - t0
    size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"\nDone! {len(activation_data)} completions, {size_mb:.0f}MB, {total_time:.0f}s total")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
