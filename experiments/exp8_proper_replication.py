#!/usr/bin/env python3
"""
Experiment 8: Proper Goodfire Replication (Features as Rewards)
================================================================
Faithful replication of the Features as Rewards methodology (Goodfire, Feb 2026):
  - Attention-based probes (not linear logistic regression)
  - Two-stage pipeline: localization + classification
  - Training on model's own generations, verified by Claude CLI
  - Per-token activation extraction
  - Evaluation at threshold 0.7 matching paper metrics

Key methodological fixes from exp1-6 (which got the methodology wrong):
  1. Probes: Transformer localization + Attention classification (paper arch)
  2. Data: Model-generated completions, Claude-verified (not TruthfulQA QA pairs)
  3. Pipeline: Two-stage localize->classify (not single-score)
  4. Eval: Precision/recall at threshold 0.7 (paper's protocol)

Paper results (Gemma-3-12B-IT, 20K+ prompts, ~5M entities):
  - Localization AUC: 0.88
  - Classification AUC: 0.94
  - At tau=0.7: Precision 0.85, Recall 0.56
  - Hallucination reduction: 58%

Our setup (adapted for hardware):
  - Model: Gemma-2-2B-it (26 layers, hidden_dim=2304) -- paper uses 12B
  - Verifier: Claude CLI (paper: Gemini 2.5 Pro + web search)
  - Data: 50 prompts x 2 completions = 100 completions (~500-1000 entities)
  - Layer mapping: paper L20->L11, paper L30->L16

Stages:
  8.1: Generate completions on hard factual prompts
  8.2: Extract entities + verify via Claude CLI
  8.3: Extract per-token activations at target layers
  8.4: Prepare training/test datasets
  8.5: Train localization probe (Transformer) + classification probe (Attention)
  8.6: End-to-end evaluation (AUROC, P/R at tau=0.7, bootstrap CI)
  8.7: Ablation -- linear probe on same data (isolate architecture effect)

Usage: python exp8_proper_replication.py
Output: results/exp8_results.md + results/exp8_results.json
"""

import json
import math
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_BOOTSTRAP = 1000
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = REPO_ROOT / "results" / "exp8_results.json"
OUTPUT_MD = REPO_ROOT / "results" / "exp8_results.md"
CACHE_COMPLETIONS = REPO_ROOT / "results" / "exp8_completions.json"
CACHE_ENTITIES = REPO_ROOT / "results" / "exp8_entities.json"
CACHE_ACTIVATIONS = REPO_ROOT / "results" / "exp8_activations.pkl"

# Layer mapping: paper uses 48-layer model, we use 26-layer
# Paper L20/48 ~ 0.42 -> L11/26, Paper L30/48 ~ 0.63 -> L16/26
LOC_LAYER = 11
CLS_LAYERS = [11, 16]
HIDDEN_DIM = 2304

# Localization probe (paper B.3.1)
LOC_EMBED = 128
LOC_HEADS = 8
LOC_DEPTH = 4
LOC_WINDOW = 256

# Classification probe (paper B.3.2, scaled for 2B)
CLS_EMBED = 1024   # paper: 2048 for 3840-dim model
CLS_HEADS = 8      # paper: 16

# Training hyperparams (paper B.2)
LOC_LR = 1e-3
LOC_WD = 0.1
LOC_EPOCHS = 5
CLS_LR = 5e-2
CLS_WD = 0.1
CLS_EPOCHS = 8
WARMUP_FRAC = 0.1
BATCH_SIZE = 16

# Generation params (paper Table 3)
GEN_TEMP = 1.0
GEN_TOP_P = 0.95
GEN_TOP_K = 64
GEN_MAX_TOKENS = 512  # paper: 4096, reduced for 2B model
N_COMPLETIONS = 2     # paper: 4 per prompt

CLAUDE_MODEL = "claude-sonnet-4-6"

np.random.seed(SEED)
torch.manual_seed(SEED)

# RMSNorm with fallback for older PyTorch
try:
    _RMSNorm = nn.RMSNorm
except AttributeError:
    class _RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8) * self.scale

# ---------------------------------------------------------------------------
# Prompts: 50 hard factual prompts across 8 Longfact++ categories
# ---------------------------------------------------------------------------
PROMPTS = [
    # Biography (8)
    "Write a detailed biography of Marian Smoluchowski including specific dates, institutions, and numerical results from his physics work.",
    "Describe Paul Ehrenfest's contributions to statistical mechanics, his key papers with dates, and his academic career.",
    "Detail Lise Meitner's role in discovering nuclear fission, including dates, energy calculations, and collaborators.",
    "Write about Subrahmanyan Chandrasekhar's work on stellar structure, including the exact Chandrasekhar limit value.",
    "Describe Emmy Noether's mathematical contributions, institutions, and key dates in her career.",
    "Detail Srinivasa Ramanujan's contributions to partition theory, his formulas, and his time at Cambridge.",
    "Write about Rosalind Franklin's X-ray crystallography work, including Photo 51 and its significance.",
    "Describe Évariste Galois's mathematical results, his final letter, and the circumstances of his death.",
    # Science (7)
    "Explain the Standard Model of particle physics, including the specific masses of fundamental particles in MeV/c².",
    "Describe the discovery of CRISPR-Cas9, including key papers, authors, and dates of publication.",
    "Detail the Michelson-Morley experiment, including specific measurements, dates, and apparatus specifications.",
    "Write about the discovery of cosmic microwave background radiation by Penzias and Wilson, including dates and measurements.",
    "Describe the Stern-Gerlach experiment, including specific results, dates, and physical parameters.",
    "Detail the double-slit experiment variations from Young to modern quantum versions, with dates.",
    "Explain the photoelectric effect experiment results that led to Einstein's Nobel Prize, with numerical values.",
    # Medical (6)
    "Describe the development of the polio vaccine by Salk and Sabin, including trial dates, participant numbers, and efficacy rates.",
    "Detail the discovery of penicillin by Alexander Fleming, including the specific date, circumstances, and subsequent development.",
    "Write about the Human Genome Project, including start date, completion date, cost, number of base pairs sequenced.",
    "Describe the first heart transplant by Christiaan Barnard, including specific dates, patient names, and outcomes.",
    "Detail the development of insulin treatment for diabetes, including Banting and Best's experiments and dates.",
    "Write about the eradication of smallpox, including WHO campaign dates, last natural case, and vaccination statistics.",
    # History (7)
    "Describe the Treaty of Westphalia (1648), including specific dates, signatories, key articles, and territorial changes.",
    "Detail the Battle of Borodino (1812), including casualty figures, commanders, troop numbers, and tactical movements.",
    "Write about the construction of the Panama Canal, including specific dates, costs, engineering challenges, and death toll.",
    "Describe the Apollo 11 mission timeline, including specific times, coordinates, and crew activities.",
    "Detail the Manhattan Project, including sites, key scientists, budget figures, and timeline to Trinity test.",
    "Write about the Congress of Vienna (1814-1815), including dates, decisions, participating nations, and outcomes.",
    "Describe the fall of Constantinople in 1453, including dates, troop numbers, weapons, and political consequences.",
    # Geography (5)
    "Describe Lake Baikal, including maximum depth, volume, age, endemic species count, and UNESCO designation.",
    "Detail the Mariana Trench, including maximum depth, coordinates, exploration history, and species discovered.",
    "Write about the Amazon River, including length, discharge rate, number of tributaries, and basin area.",
    "Describe the Galápagos Islands, including formation date, number of islands, endemic species, and Darwin's visit.",
    "Detail Mount Everest, including exact height, first ascent date, climbers, death toll, and geological formation.",
    # Citations (6)
    "Provide exact publication details for Shannon's 'A Mathematical Theory of Communication' (journal, volume, pages, year).",
    "Cite Einstein's four 1905 papers with exact journal names, volumes, page numbers, and submission dates.",
    "Provide publication details for Watson and Crick's 1953 DNA structure paper and Franklin's crystallography paper.",
    "Cite the original Gödel incompleteness theorems publication with exact journal, volume, and page numbers.",
    "Provide exact citations for the RSA cryptosystem paper by Rivest, Shamir, and Adleman.",
    "Cite Hubble's 1929 expanding universe paper and Penzias & Wilson's 1965 CMB discovery paper with full details.",
    # Legal (5)
    "Describe the Magna Carta, including the exact date, location, key clauses, number of articles, and signatories.",
    "Detail the Treaty of Tordesillas, including date, signatories, specific longitude line, and papal involvement.",
    "Write about Brown v. Board of Education, including case number, date, justices, vote count, and legal arguments.",
    "Describe the Nuremberg Trials, including specific dates, defendants, charges, verdicts, and legal precedents set.",
    "Detail the Geneva Conventions, including dates of each convention, number of articles, and signatory counts.",
    # Other (6)
    "Describe the specifications of the Hubble Space Telescope's instruments, including wavelength ranges and resolution.",
    "Detail the Voyager 1 spacecraft's instruments, including mass, power consumption, and current distance from Earth.",
    "Write about the Large Hadron Collider's specifications: beam energy, luminosity, circumference, and magnet details.",
    "Describe the International Space Station's dimensions, mass, orbital parameters, and power generation capacity.",
    "Detail the James Webb Space Telescope's specifications: mirror diameter, wavelength range, and orbit.",
    "Write about the GPS satellite system specifications: orbital altitude, number of satellites, and timing precision.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def claude(prompt, input_text="", timeout=120):
    """Call Claude CLI."""
    cmd = ["claude", "-p", prompt, "--model", CLAUDE_MODEL]
    try:
        result = subprocess.run(
            cmd, input=input_text or None, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return ""


def parse_json(raw):
    """Parse JSON from Claude output, handling markdown fences."""
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


def find_entity_token_indices(text, entity_text, offsets, search_start=0):
    """Map entity text to token indices using character offsets."""
    idx = text.find(entity_text, search_start)
    if idx == -1:
        idx = text.find(entity_text)
    if idx == -1:
        return [], 0
    char_start, char_end = idx, idx + len(entity_text)
    token_indices = []
    for tok_idx, (ts, te) in enumerate(offsets):
        if te > char_start and ts < char_end:
            token_indices.append(tok_idx)
    return token_indices, char_end


def bootstrap_ci(y_true, y_scores, n=N_BOOTSTRAP):
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(SEED)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_scores[idx]))
    if not scores:
        return 0.5, 0.5
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="eager",
)
model.eval()
NUM_LAYERS = model.config.num_hidden_layers
print(f"Loaded {MODEL_ID} on {DEVICE}, {NUM_LAYERS} layers, hidden_dim={model.config.hidden_size}")


def generate(prompt, seed=SEED, max_new_tokens=GEN_MAX_TOKENS):
    """Generate a completion using the paper's sampling parameters."""
    suffix = ("\n\nPlease provide a detailed, fact-dense response with specific names, "
              "dates, numbers, and citations. Be as precise as possible.")
    chat = [{"role": "user", "content": prompt + suffix}]
    inputs = tokenizer.apply_chat_template(
        chat, return_tensors="pt", add_generation_prompt=True,
    ).to(DEVICE)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=GEN_TEMP, top_p=GEN_TOP_P, top_k=GEN_TOP_K,
        )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def extract_per_token_activations(text, layers, max_len=512):
    """Extract per-token hidden states at specified layers.
    Returns dict[layer] -> np.array(seq_len, hidden_dim) and offset_mapping.
    """
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
        result[layer] = out.hidden_states[layer][0].float().cpu().numpy()
    return result, offsets


# ---------------------------------------------------------------------------
# Probe Architectures
# ---------------------------------------------------------------------------
class TransformerLocProbe(nn.Module):
    """Localization probe: per-token entity detection.

    Paper B.3.1: L=4 layer Transformer, E=128, Nh=8, w=256, sigmoid output.
    Simplified: standard TransformerEncoder with pre-norm and learned positions.
    (Paper uses gated SWA + RoPE + GeGLU; we use standard attention + GELU.)
    """

    def __init__(self, input_dim=HIDDEN_DIM, embed_dim=LOC_EMBED, n_heads=LOC_HEADS,
                 n_layers=LOC_DEPTH, max_len=LOC_WINDOW, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, x, padding_mask=None):
        """x: (B, T, D). padding_mask: (B, T) True=padded."""
        B, T, _ = x.shape
        x = self.in_proj(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.out_norm(x)
        return self.out_proj(x).squeeze(-1)  # (B, T)


class AttentionClassProbe(nn.Module):
    """Classification probe: per-entity hallucination detection.

    Paper B.3.2 / Algorithm 2: Noncausal attention probe with learned query.
    Multi-layer input concatenated along sequence dimension.
    Single learned query per head attends over key/value, sigmoid output.
    """

    def __init__(self, input_dim=HIDDEN_DIM, embed_dim=CLS_EMBED, n_heads=CLS_HEADS,
                 n_input_layers=2, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_input_layers = n_input_layers
        self.embed_dim = embed_dim
        # Per-layer RMS normalization (paper Algorithm 2)
        self.norms = nn.ModuleList([_RMSNorm(input_dim) for _ in range(n_input_layers)])
        # Shared k,v projection
        self.wkv = nn.Linear(input_dim, 2 * embed_dim, bias=False)
        # Learned query per head (paper: self.query)
        self.query = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        # Output
        self.wout = nn.Linear(embed_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_layers, mask=None):
        """x_layers: list of (B, T, D) tensors. mask: (B, T) True=padded."""
        normed = [self.norms[i](x_layers[i]) for i in range(self.n_input_layers)]
        x = torch.stack(normed, dim=2)  # (B, T, L, D)
        B, T, L, D = x.shape
        x = x.reshape(B, T * L, D)

        if mask is not None:
            mask = mask.unsqueeze(2).expand(-1, -1, L).reshape(B, T * L)

        kv = self.wkv(x).reshape(B, T * L, 2, self.n_heads, self.head_dim)
        k = kv[:, :, 0].permute(0, 2, 1, 3)  # (B, H, T*L, hd)
        v = kv[:, :, 1].permute(0, 2, 1, 3)

        query = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, H, hd)
        logits = torch.einsum('bhd,bhtd->bht', query, k) / (self.head_dim ** 0.5)

        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(1), float('-inf'))

        scores = torch.softmax(logits.float(), dim=-1).to(x.dtype)
        scores = self.dropout(scores)

        value = torch.einsum('bht,bhtd->bhd', scores, v)
        value = value.reshape(B, self.embed_dim)
        return self.wout(value).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class LocDataset(Dataset):
    """Per-completion windowed dataset for localization probe."""

    def __init__(self, completions_data, max_len=LOC_WINDOW):
        self.samples = []
        for comp in completions_data:
            acts = comp["activations"][LOC_LAYER]
            labels = comp["token_entity_labels"]
            seq_len = acts.shape[0]
            for start in range(0, seq_len, max_len):
                end = min(start + max_len, seq_len)
                chunk_a = acts[start:end]
                chunk_l = labels[start:end]
                real_len = end - start
                pad = max_len - real_len
                if pad > 0:
                    chunk_a = np.pad(chunk_a, ((0, pad), (0, 0)))
                    chunk_l = np.pad(chunk_l, (0, pad), constant_values=-1)
                self.samples.append((chunk_a, chunk_l, real_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a, l, r = self.samples[idx]
        return torch.tensor(a, dtype=torch.float32), torch.tensor(l, dtype=torch.float32), r


class ClsDataset(Dataset):
    """Per-entity dataset for classification probe."""

    def __init__(self, entities_data, max_tokens=32):
        self.samples = []
        self.max_tokens = max_tokens
        for ent in entities_data:
            layer_acts = ent["layer_activations"]
            label = ent["label"]
            n_tok = layer_acts[0].shape[0]
            real_len = min(n_tok, max_tokens)
            padded = []
            for la in layer_acts:
                if la.shape[0] > max_tokens:
                    la = la[:max_tokens]
                elif la.shape[0] < max_tokens:
                    la = np.pad(la, ((0, max_tokens - la.shape[0]), (0, 0)))
                padded.append(la)
            self.samples.append((padded, label, real_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        las, label, rl = self.samples[idx]
        return ([torch.tensor(la, dtype=torch.float32) for la in las],
                torch.tensor(label, dtype=torch.float32), rl)


def cls_collate_fn(batch):
    """Custom collate for classification dataset (variable-count layer tensors)."""
    n_layers = len(batch[0][0])
    all_las = [[] for _ in range(n_layers)]
    labels, lens = [], []
    for layer_acts, label, rl in batch:
        for i, la in enumerate(layer_acts):
            all_las[i].append(la)
        labels.append(label)
        lens.append(rl)
    return [torch.stack(la) for la in all_las], torch.stack(labels), lens


# ---------------------------------------------------------------------------
# Stage 8.1: Generate completions
# ---------------------------------------------------------------------------
def stage_generate():
    if CACHE_COMPLETIONS.exists():
        print(f"  Loading cached completions from {CACHE_COMPLETIONS}")
        return json.loads(CACHE_COMPLETIONS.read_text())

    print("\n" + "=" * 60)
    print("  Stage 8.1: Generate Completions")
    print("=" * 60)

    completions = []
    total = len(PROMPTS) * N_COMPLETIONS
    for i, prompt in enumerate(PROMPTS):
        for j in range(N_COMPLETIONS):
            seed = SEED + i * N_COMPLETIONS + j
            idx = i * N_COMPLETIONS + j
            print(f"  [{idx+1}/{total}] {prompt[:50]}... (seed={seed})")
            text = generate(prompt, seed=seed)
            completions.append({
                "id": idx, "prompt_idx": i, "prompt": prompt,
                "seed": seed, "text": text,
            })
    CACHE_COMPLETIONS.write_text(json.dumps(completions, indent=2))
    print(f"  Saved {len(completions)} completions")
    return completions


# ---------------------------------------------------------------------------
# Stage 8.2: Extract + verify entities
# ---------------------------------------------------------------------------
def stage_extract_verify(completions):
    if CACHE_ENTITIES.exists():
        print(f"  Loading cached entities from {CACHE_ENTITIES}")
        return json.loads(CACHE_ENTITIES.read_text())

    print("\n" + "=" * 60)
    print("  Stage 8.2: Extract & Verify Entities")
    print("=" * 60)

    all_entities = []
    for comp in completions:
        print(f"\n  Completion {comp['id']} ({comp['prompt'][:40]}...)...")

        raw = claude(
            "Extract ALL specific factual entities from this text. Target: people, "
            "organizations, locations, dates, numbers, citations, measurements. "
            "For each entity provide:\n"
            '- "entity": the exact text as it appears\n'
            '- "type": person/date/number/place/org/citation/measurement/other\n'
            '- "claim": a self-contained factual claim about this entity from the text\n'
            "Return a JSON array. Be thorough. Only return the JSON array.",
            input_text=comp["text"], timeout=60,
        )
        items = parse_json(raw)
        if not isinstance(items, list):
            print(f"    [warn] Could not parse entities")
            continue

        entities = []
        for item in items:
            if not isinstance(item, dict) or not item.get("entity"):
                continue
            entities.append({
                "entity": item["entity"],
                "type": item.get("type", "other"),
                "claim": item.get("claim", item["entity"]),
                "completion_id": comp["id"],
            })

        print(f"    Extracted {len(entities)} entities, verifying...")

        # Verify in batches of 10 (paper A.2.2)
        for bs in range(0, len(entities), 10):
            batch = entities[bs:bs + 10]
            claims = "\n".join(f"{k+1}. {e['claim']}" for k, e in enumerate(batch))
            raw = claude(
                "Verify each factual claim. For each, determine if it is:\n"
                "- SUPPORTED: factually correct and used correctly in context\n"
                "- NOT_SUPPORTED: factually incorrect or used incorrectly\n"
                "- INSUFFICIENT_INFO: cannot be verified\n"
                'Return a JSON array: [{"verdict": "...", "reason": "brief"}, ...]',
                input_text=claims, timeout=60,
            )
            results = parse_json(raw)
            if not isinstance(results, list):
                results = []
            for k, ent in enumerate(batch):
                if k < len(results) and isinstance(results[k], dict):
                    v = results[k].get("verdict", "INSUFFICIENT_INFO").upper()
                    if v not in ("SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_INFO"):
                        v = "INSUFFICIENT_INFO"
                    ent["verdict"] = v
                    ent["reason"] = results[k].get("reason", "")
                else:
                    ent["verdict"] = "INSUFFICIENT_INFO"
                    ent["reason"] = "parse error"
            time.sleep(0.3)

        n_s = sum(1 for e in entities if e["verdict"] == "SUPPORTED")
        n_ns = sum(1 for e in entities if e["verdict"] == "NOT_SUPPORTED")
        n_ii = sum(1 for e in entities if e["verdict"] == "INSUFFICIENT_INFO")
        print(f"    {n_s} supported, {n_ns} not supported, {n_ii} insufficient")
        all_entities.extend(entities)

    total = len(all_entities)
    s = sum(1 for e in all_entities if e["verdict"] == "SUPPORTED")
    ns = sum(1 for e in all_entities if e["verdict"] == "NOT_SUPPORTED")
    print(f"\n  Total: {total} entities ({s} supported, {ns} not supported)")

    CACHE_ENTITIES.write_text(json.dumps(all_entities, indent=2))
    return all_entities


# ---------------------------------------------------------------------------
# Stage 8.3: Extract per-token activations
# ---------------------------------------------------------------------------
def stage_extract_activations(completions):
    if CACHE_ACTIVATIONS.exists():
        print(f"  Loading cached activations from {CACHE_ACTIVATIONS}")
        with open(CACHE_ACTIVATIONS, "rb") as f:
            return pickle.load(f)

    print("\n" + "=" * 60)
    print("  Stage 8.3: Extract Per-Token Activations")
    print("=" * 60)

    all_layers = sorted(set([LOC_LAYER] + CLS_LAYERS))
    activation_data = {}

    for comp in completions:
        cid = comp["id"]
        if cid % 10 == 0:
            print(f"  [{cid}/{len(completions)}]...")
        acts, offsets = extract_per_token_activations(comp["text"], all_layers)
        activation_data[cid] = {"activations": acts, "offsets": offsets}

    with open(CACHE_ACTIVATIONS, "wb") as f:
        pickle.dump(activation_data, f)
    print(f"  Saved activations for {len(activation_data)} completions")
    return activation_data


# ---------------------------------------------------------------------------
# Stage 8.4: Prepare training/test data
# ---------------------------------------------------------------------------
def stage_prepare_data(completions, entities, activation_data):
    print("\n" + "=" * 60)
    print("  Stage 8.4: Prepare Training Data")
    print("=" * 60)

    comp_entities = {}
    for ent in entities:
        comp_entities.setdefault(ent["completion_id"], []).append(ent)

    loc_data = []
    cls_data = []

    for comp in completions:
        cid = comp["id"]
        if cid not in activation_data:
            continue
        acts_dict = activation_data[cid]["activations"]
        offsets = activation_data[cid]["offsets"]
        text = comp["text"]
        seq_len = acts_dict[LOC_LAYER].shape[0]

        token_labels = np.zeros(seq_len, dtype=np.int32)
        search_start = 0

        for ent in comp_entities.get(cid, []):
            tok_idx, new_start = find_entity_token_indices(
                text, ent["entity"], offsets, search_start,
            )
            ent["token_indices"] = tok_idx
            if tok_idx:
                search_start = new_start
                for ti in tok_idx:
                    if ti < seq_len:
                        token_labels[ti] = 1

                if ent["verdict"] in ("SUPPORTED", "NOT_SUPPORTED"):
                    valid_idx = [ti for ti in tok_idx if ti < seq_len]
                    if valid_idx:
                        layer_acts = [acts_dict[l][valid_idx] for l in CLS_LAYERS]
                        cls_data.append({
                            "layer_activations": layer_acts,
                            "label": 1 if ent["verdict"] == "NOT_SUPPORTED" else 0,
                            "entity": ent["entity"],
                            "completion_id": cid,
                        })

        loc_data.append({
            "activations": acts_dict,
            "token_entity_labels": token_labels,
            "completion_id": cid,
        })

    n_ent_tok = sum(d["token_entity_labels"].sum() for d in loc_data)
    n_tot_tok = sum(d["token_entity_labels"].shape[0] for d in loc_data)
    n_hallu = sum(1 for d in cls_data if d["label"] == 1)
    n_supp = sum(1 for d in cls_data if d["label"] == 0)
    print(f"  Localization: {n_tot_tok} tokens, {n_ent_tok} entity ({n_ent_tok/max(n_tot_tok,1)*100:.1f}%)")
    print(f"  Classification: {len(cls_data)} entities ({n_hallu} hallucinated, {n_supp} supported)")

    comp_ids = list(set(d["completion_id"] for d in loc_data))
    train_ids, test_ids = train_test_split(comp_ids, test_size=0.2, random_state=SEED)
    train_set, test_set = set(train_ids), set(test_ids)

    result = {
        "loc_train": [d for d in loc_data if d["completion_id"] in train_set],
        "loc_test": [d for d in loc_data if d["completion_id"] in test_set],
        "cls_train": [d for d in cls_data if d["completion_id"] in train_set],
        "cls_test": [d for d in cls_data if d["completion_id"] in test_set],
    }
    print(f"  Split: {len(train_ids)} train / {len(test_ids)} test completions")
    print(f"  Cls train: {len(result['cls_train'])}, Cls test: {len(result['cls_test'])}")
    return result


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def cosine_schedule(optimizer, num_steps, warmup_frac=WARMUP_FRAC):
    warmup = int(num_steps * warmup_frac)
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, num_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Stage 8.5a: Train localization probe
# ---------------------------------------------------------------------------
def train_localization(loc_train, loc_test):
    print("\n" + "=" * 60)
    print("  Stage 8.5a: Train Localization Probe")
    print("=" * 60)

    train_ds = LocDataset(loc_train)
    test_ds = LocDataset(loc_test)

    if len(train_ds) == 0:
        print("  No training data, skipping")
        return None, 0.5

    all_labels = np.concatenate([d["token_entity_labels"] for d in loc_train])
    valid = all_labels >= 0
    n_pos = (all_labels[valid] == 1).sum()
    n_neg = (all_labels[valid] == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    print(f"  {n_pos} pos, {n_neg} neg tokens, pos_weight={pos_weight.item():.1f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    probe = TransformerLocProbe().to(DEVICE)
    optim = torch.optim.AdamW(probe.parameters(), lr=LOC_LR, weight_decay=LOC_WD)
    sched = cosine_schedule(optim, LOC_EPOCHS * len(train_loader))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    best_auroc = 0
    for epoch in range(LOC_EPOCHS):
        probe.train()
        total_loss, n_batch = 0, 0
        for acts, labels, real_lens in train_loader:
            acts, labels = acts.to(DEVICE), labels.to(DEVICE)
            B, T = labels.shape
            pad_mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
            for b in range(B):
                pad_mask[b, real_lens[b]:] = True

            logits = probe(acts, padding_mask=pad_mask)
            valid_mask = labels >= 0
            loss = (criterion(logits, labels.clamp(0, 1)) * valid_mask.float()).sum()
            loss = loss / valid_mask.float().sum()

            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            total_loss += loss.item()
            n_batch += 1

        # Eval
        probe.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for acts, labels, real_lens in DataLoader(test_ds, batch_size=BATCH_SIZE):
                acts = acts.to(DEVICE)
                B, T = labels.shape
                pad_mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
                for b in range(B):
                    pad_mask[b, real_lens[b]:] = True
                probs = torch.sigmoid(probe(acts, padding_mask=pad_mask)).cpu().numpy()
                lbl = labels.numpy()
                for b in range(B):
                    v = lbl[b] >= 0
                    all_p.extend(probs[b][v].tolist())
                    all_l.extend(lbl[b][v].tolist())

        try:
            auroc = roc_auc_score(np.array(all_l), np.array(all_p))
        except ValueError:
            auroc = 0.5
        best_auroc = max(best_auroc, auroc)
        print(f"    Epoch {epoch+1}/{LOC_EPOCHS}: loss={total_loss/max(n_batch,1):.4f}, AUROC={auroc:.4f}")

    return probe, best_auroc


# ---------------------------------------------------------------------------
# Stage 8.5b: Train classification probe
# ---------------------------------------------------------------------------
def train_classification(cls_train, cls_test):
    print("\n" + "=" * 60)
    print("  Stage 8.5b: Train Classification Probe")
    print("=" * 60)

    if len(cls_train) < 10:
        print("  Too few entities, skipping")
        return None, 0.5

    train_ds = ClsDataset(cls_train)
    test_ds = ClsDataset(cls_test)

    n_h = sum(1 for d in cls_train if d["label"] == 1)
    n_s = sum(1 for d in cls_train if d["label"] == 0)
    print(f"  Train: {n_h} hallucinated, {n_s} supported")

    bs = min(BATCH_SIZE, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=cls_collate_fn)

    probe = AttentionClassProbe(
        input_dim=HIDDEN_DIM, embed_dim=CLS_EMBED, n_heads=CLS_HEADS,
        n_input_layers=len(CLS_LAYERS),
    ).to(DEVICE)

    optim = torch.optim.AdamW(probe.parameters(), lr=CLS_LR, weight_decay=CLS_WD)
    sched = cosine_schedule(optim, CLS_EPOCHS * len(train_loader))
    pos_weight = torch.tensor([n_s / max(n_h, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc = 0
    for epoch in range(CLS_EPOCHS):
        probe.train()
        total_loss, n_batch = 0, 0
        for layer_acts, labels, real_lens in train_loader:
            layer_acts = [la.to(DEVICE) for la in layer_acts]
            labels = labels.to(DEVICE)
            B, max_t = labels.shape[0], layer_acts[0].shape[1]
            mask = torch.zeros(B, max_t, dtype=torch.bool, device=DEVICE)
            for b in range(B):
                mask[b, real_lens[b]:] = True

            logits = probe(layer_acts, mask=mask)
            loss = criterion(logits, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            total_loss += loss.item()
            n_batch += 1

        # Eval
        probe.eval()
        all_p, all_l = [], []
        test_loader = DataLoader(
            test_ds, batch_size=min(bs, len(test_ds)), collate_fn=cls_collate_fn,
        )
        with torch.no_grad():
            for layer_acts, labels, real_lens in test_loader:
                layer_acts = [la.to(DEVICE) for la in layer_acts]
                B, max_t = labels.shape[0], layer_acts[0].shape[1]
                mask = torch.zeros(B, max_t, dtype=torch.bool, device=DEVICE)
                for b in range(B):
                    mask[b, real_lens[b]:] = True
                probs = torch.sigmoid(probe(layer_acts, mask=mask)).cpu().numpy()
                all_p.extend(probs.tolist())
                all_l.extend(labels.numpy().tolist())

        try:
            auroc = roc_auc_score(np.array(all_l), np.array(all_p))
        except ValueError:
            auroc = 0.5
        best_auroc = max(best_auroc, auroc)
        print(f"    Epoch {epoch+1}/{CLS_EPOCHS}: loss={total_loss/max(n_batch,1):.4f}, AUROC={auroc:.4f}")

    return probe, best_auroc


# ---------------------------------------------------------------------------
# Stage 8.6: End-to-end evaluation
# ---------------------------------------------------------------------------
def stage_evaluate(loc_probe, cls_probe, data, threshold=0.7):
    print("\n" + "=" * 60)
    print("  Stage 8.6: End-to-End Evaluation")
    print("=" * 60)

    results = {}

    # Classification probe
    if cls_probe is not None and len(data["cls_test"]) > 0:
        test_ds = ClsDataset(data["cls_test"])
        cls_probe.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for layer_acts, labels, real_lens in DataLoader(
                test_ds, batch_size=BATCH_SIZE, collate_fn=cls_collate_fn,
            ):
                layer_acts = [la.to(DEVICE) for la in layer_acts]
                B, max_t = labels.shape[0], layer_acts[0].shape[1]
                mask = torch.zeros(B, max_t, dtype=torch.bool, device=DEVICE)
                for b in range(B):
                    mask[b, real_lens[b]:] = True
                probs = torch.sigmoid(cls_probe(layer_acts, mask=mask)).cpu().numpy()
                all_p.extend(probs.tolist())
                all_l.extend(labels.numpy().tolist())

        preds, labels = np.array(all_p), np.array(all_l)
        try:
            cls_auroc = float(roc_auc_score(labels, preds))
        except ValueError:
            cls_auroc = 0.5

        ci_lo, ci_hi = bootstrap_ci(labels, preds)
        binary = (preds >= threshold).astype(int)
        prec = float(precision_score(labels, binary, zero_division=0))
        rec = float(recall_score(labels, binary, zero_division=0))

        # Optimal threshold
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            b = (preds >= t).astype(int)
            f = f1_score(labels, b, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t

        results["classification"] = {
            "auroc": cls_auroc, "auroc_ci": [ci_lo, ci_hi],
            "threshold_0.7": {
                "precision": prec, "recall": rec,
                "f1": float(f1_score(labels, binary, zero_division=0)),
            },
            "optimal": {"threshold": float(best_t), "f1": float(best_f1)},
            "n_test": len(labels),
            "n_hallucinated": int(labels.sum()),
            "n_supported": int(len(labels) - labels.sum()),
        }
        print(f"  Classification AUROC: {cls_auroc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  At tau=0.7: P={prec:.3f}, R={rec:.3f}")
        print(f"  Optimal: tau={best_t:.2f}, F1={best_f1:.3f}")

    # Localization probe
    if loc_probe is not None and len(data["loc_test"]) > 0:
        test_ds = LocDataset(data["loc_test"])
        loc_probe.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for acts, labels, real_lens in DataLoader(test_ds, batch_size=BATCH_SIZE):
                acts = acts.to(DEVICE)
                B, T = labels.shape
                pad_mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
                for b in range(B):
                    pad_mask[b, real_lens[b]:] = True
                probs = torch.sigmoid(loc_probe(acts, padding_mask=pad_mask)).cpu().numpy()
                lbl = labels.numpy()
                for b in range(B):
                    v = lbl[b] >= 0
                    all_p.extend(probs[b][v].tolist())
                    all_l.extend(lbl[b][v].tolist())

        preds, labels = np.array(all_p), np.array(all_l)
        try:
            loc_auroc = float(roc_auc_score(labels, preds))
        except ValueError:
            loc_auroc = 0.5

        results["localization"] = {
            "auroc": loc_auroc,
            "n_tokens": len(labels),
            "n_entity_tokens": int(labels.sum()),
        }
        print(f"  Localization AUROC: {loc_auroc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Stage 8.7: Ablation -- linear probe on same data
# ---------------------------------------------------------------------------
def stage_ablation(data):
    print("\n" + "=" * 60)
    print("  Stage 8.7: Ablation - Linear Probe on Same Data")
    print("=" * 60)

    if len(data["cls_train"]) < 10 or len(data["cls_test"]) < 5:
        print("  Too few entities for ablation")
        return {}

    def build_features(entities):
        X, y = [], []
        for ent in entities:
            features = np.concatenate([la.mean(axis=0) for la in ent["layer_activations"]])
            X.append(features)
            y.append(ent["label"])
        return np.array(X), np.array(y)

    X_train, y_train = build_features(data["cls_train"])
    X_test, y_test = build_features(data["cls_test"])

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    lr = LogisticRegression(C=0.01, max_iter=1000, random_state=SEED)
    lr.fit(X_tr, y_train)
    lr_preds = lr.predict_proba(X_te)[:, 1]

    try:
        lr_auroc = float(roc_auc_score(y_test, lr_preds))
    except ValueError:
        lr_auroc = 0.5

    lr_b = (lr_preds >= 0.5).astype(int)
    result = {
        "linear_same_data": {
            "auroc": lr_auroc,
            "precision": float(precision_score(y_test, lr_b, zero_division=0)),
            "recall": float(recall_score(y_test, lr_b, zero_division=0)),
            "f1": float(f1_score(y_test, lr_b, zero_division=0)),
            "n_train": len(y_train), "n_test": len(y_test),
        },
    }
    print(f"  Linear probe (same data): AUROC={lr_auroc:.4f}")
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(results, ablation, data_stats, elapsed):
    cls = results.get("classification", {})
    loc = results.get("localization", {})
    lr = ablation.get("linear_same_data", {})

    lines = [
        "# Experiment 8: Proper Goodfire Replication",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Runtime: {elapsed:.0f}s*",
        "",
        "## Methodology",
        "",
        "Faithful replication of Features as Rewards (Goodfire, Feb 2026):",
        f"- **Model:** {MODEL_ID} ({NUM_LAYERS} layers, hidden_dim={HIDDEN_DIM})",
        f"- **Probes:** Transformer localization (L={LOC_DEPTH}, E={LOC_EMBED}, Nh={LOC_HEADS})"
        f" + Attention classification (E={CLS_EMBED}, Nh={CLS_HEADS})",
        f"- **Data:** {len(PROMPTS)} prompts x {N_COMPLETIONS} completions = "
        f"{len(PROMPTS) * N_COMPLETIONS} (paper: 20K x 4 = 84K)",
        f"- **Verifier:** Claude CLI (paper: Gemini 2.5 Pro + web search)",
        f"- **Layers:** Loc=L{LOC_LAYER}, Cls=L{CLS_LAYERS} (paper: L20, L[20,30])",
        "",
        "### Key Differences from Paper",
        "",
        "| Aspect | Paper | Ours |",
        "|--------|-------|------|",
        f"| Model | Gemma-3-12B-IT (48L, 3840D) | Gemma-2-2B-it ({NUM_LAYERS}L, {HIDDEN_DIM}D) |",
        f"| Data | 20K prompts, ~5M entities | {len(PROMPTS)} prompts, "
        f"~{data_stats.get('n_entities', '?')} entities |",
        "| Verifier | Gemini 2.5 Pro + web search | Claude CLI (no web search) |",
        "| Loc probe | Gated SWA + RoPE + GeGLU | Standard Transformer + learned pos |",
        "| RL training | 360 steps ScaleRL/CISPO | None (probes only) |",
        "",
        "## Data Statistics",
        "",
        f"- **Completions:** {data_stats.get('n_completions', '?')}",
        f"- **Entities:** {data_stats.get('n_entities', '?')} "
        f"({data_stats.get('n_supported', '?')} supported, "
        f"{data_stats.get('n_not_supported', '?')} not supported, "
        f"{data_stats.get('n_insufficient', '?')} insufficient)",
        "",
    ]

    if cls:
        ci = cls.get("auroc_ci", [0, 0])
        t7 = cls.get("threshold_0.7", {})
        opt = cls.get("optimal", {})
        lines += [
            "## Classification Probe Results",
            "",
            "| Metric | Paper (12B) | Ours (2B) |",
            "|--------|-------------|-----------|",
            f"| AUROC | 0.94 | **{cls['auroc']:.4f}** [{ci[0]:.3f}, {ci[1]:.3f}] |",
            f"| Precision (tau=0.7) | 0.85 | **{t7.get('precision', 0):.3f}** |",
            f"| Recall (tau=0.7) | 0.56 | **{t7.get('recall', 0):.3f}** |",
            f"| F1 (tau=0.7) | -- | **{t7.get('f1', 0):.3f}** |",
            f"| Optimal tau | -- | {opt.get('threshold', 0):.2f} "
            f"(F1={opt.get('f1', 0):.3f}) |",
            f"| Test entities | ~243K | {cls.get('n_test', '?')} |",
            "",
        ]

    if loc:
        lines += [
            "## Localization Probe Results",
            "",
            f"| Metric | Paper (12B) | Ours (2B) |",
            "|--------|-------------|-----------|",
            f"| AUROC | 0.88 | **{loc['auroc']:.4f}** |",
            f"| Test tokens | -- | {loc.get('n_tokens', '?')} "
            f"({loc.get('n_entity_tokens', '?')} entity) |",
            "",
        ]

    if lr:
        cls_auroc = cls.get("auroc", 0.5)
        lines += [
            "## Ablation: Architecture vs Data",
            "",
            "| Probe | AUROC | Notes |",
            "|-------|-------|-------|",
            f"| Attention (paper arch) | **{cls_auroc:.4f}** | Proper Goodfire replication |",
            f"| Linear (exp1 arch) | {lr['auroc']:.4f} | Same data, simpler architecture |",
            "| Linear (TruthfulQA, exp4 OOD) | 0.592 | Wrong data + wrong architecture |",
            "",
            f"Architecture effect: {cls_auroc - lr['auroc']:+.3f} AUROC",
            f"Data effect: {lr['auroc'] - 0.592:+.3f} AUROC (linear probe, new data vs TruthfulQA)",
            "",
        ]

    lines += [
        "## Comparison to Previous Experiments",
        "",
        "| Exp | Approach | AUROC |",
        "|-----|----------|-------|",
        "| 1 | Linear, TruthfulQA (in-dist) | 0.877 |",
        "| 4 | Linear, TruthfulQA (OOD free-form) | 0.592 |",
        "| 6 | Linear + diverse training | 0.574 |",
        f"| **8** | **Attention, model-generated** | **{cls.get('auroc', '?')}** |",
        "",
        "## Interpretation",
        "",
    ]

    cls_auroc = cls.get("auroc", 0.5)
    if cls_auroc > 0.8:
        lines.append("The attention probe with model-generated data substantially outperforms "
                      "the linear probe approach. The Goodfire methodology works.")
    elif cls_auroc > 0.65:
        lines.append("Moderate improvement. Model scale (2B vs 12B) may be the main limitation.")
    else:
        lines.append("Attention probes do NOT substantially help on Gemma-2-2B-it. "
                      "The 2B model may lack sufficient truthfulness representations.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("  Experiment 8: Proper Goodfire Replication")
    print("=" * 60)

    completions = stage_generate()
    entities = stage_extract_verify(completions)
    activation_data = stage_extract_activations(completions)
    data = stage_prepare_data(completions, entities, activation_data)

    loc_probe, loc_auroc = train_localization(data["loc_train"], data["loc_test"])
    cls_probe, cls_auroc = train_classification(data["cls_train"], data["cls_test"])

    results = stage_evaluate(loc_probe, cls_probe, data)
    ablation = stage_ablation(data)

    elapsed = time.time() - t0

    total = len(entities)
    n_s = sum(1 for e in entities if e.get("verdict") == "SUPPORTED")
    n_ns = sum(1 for e in entities if e.get("verdict") == "NOT_SUPPORTED")
    data_stats = {
        "n_completions": len(completions), "n_entities": total,
        "n_supported": n_s, "n_not_supported": n_ns, "n_insufficient": total - n_s - n_ns,
    }

    json_results = {
        "model": MODEL_ID, "n_prompts": len(PROMPTS),
        "n_completions": len(completions), "n_entities": total,
        "data_stats": data_stats,
        "localization": results.get("localization", {}),
        "classification": results.get("classification", {}),
        "ablation": ablation,
        "config": {
            "loc_layer": LOC_LAYER, "cls_layers": CLS_LAYERS,
            "loc_embed": LOC_EMBED, "loc_heads": LOC_HEADS, "loc_depth": LOC_DEPTH,
            "cls_embed": CLS_EMBED, "cls_heads": CLS_HEADS,
            "loc_lr": LOC_LR, "cls_lr": CLS_LR, "threshold": 0.7,
        },
        "paper_comparison": {
            "paper_cls_auroc": 0.94, "paper_loc_auroc": 0.88,
            "paper_precision_0.7": 0.85, "paper_recall_0.7": 0.56,
            "exp1_in_dist": 0.877, "exp4_ood": 0.592, "exp6_diverse": 0.574,
        },
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    OUTPUT_JSON.write_text(json.dumps(json_results, indent=2))
    print(f"\nSaved: {OUTPUT_JSON}")

    report = generate_report(results, ablation, data_stats, elapsed)
    OUTPUT_MD.write_text(report)
    print(f"Saved: {OUTPUT_MD}")

    probe_path = REPO_ROOT / "results" / "exp8_probes.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump({
            "loc_probe": loc_probe.state_dict() if loc_probe else None,
            "cls_probe": cls_probe.state_dict() if cls_probe else None,
        }, f)
    print(f"Saved: {probe_path}")

    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
