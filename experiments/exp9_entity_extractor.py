#!/usr/bin/env python3
"""
Standalone entity extraction for exp9.
Reads exp9_completions.json, runs Claude CLI to extract + verify entities,
writes exp9_entities.json. Can run on any machine with Claude CLI.

Usage:
  python exp9_entity_extractor.py [--start N] [--end M]

  --start N: start at completion index N (default 0)
  --end M: stop at completion index M (default all)

This allows splitting across machines:
  Machine A: python exp9_entity_extractor.py --start 0 --end 300
  Machine B: python exp9_entity_extractor.py --start 300 --end 600
Then merge: python exp9_entity_extractor.py --merge
"""

import argparse
import json
import subprocess
import time
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPLETIONS_FILE = REPO_ROOT / "results" / "exp9_completions.json"
ENTITIES_FILE = REPO_ROOT / "results" / "exp9_entities.json"

CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_BIN = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")


def claude(prompt, input_text="", timeout=120):
    cmd = [CLAUDE_BIN, "-p", prompt, "--model", CLAUDE_MODEL]
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


def extract_and_verify(completions, start=0, end=None):
    if end is None:
        end = len(completions)

    subset = completions[start:end]
    all_entities = []
    checkpoint_file = REPO_ROOT / "results" / f"exp9_entities_partial_{start}_{end}.json"

    # Resume from checkpoint if exists
    done_ids = set()
    if checkpoint_file.exists():
        existing = json.loads(checkpoint_file.read_text())
        all_entities = existing
        done_ids = set(e["completion_id"] for e in existing)
        print(f"  Resuming from checkpoint: {len(done_ids)} completions already done")

    for i, comp in enumerate(subset):
        cid = comp["id"]
        if cid in done_ids:
            continue

        print(f"\n  [{start + i + 1}/{end}] Completion {cid} ({comp['prompt'][:40]}...)...")

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
                "completion_id": cid,
            })

        print(f"    Extracted {len(entities)} entities, verifying...")

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

        # Checkpoint every 10 completions
        if (i + 1) % 10 == 0:
            checkpoint_file.write_text(json.dumps(all_entities, indent=2))
            print(f"    [checkpoint] {len(all_entities)} entities saved")

    # Final save
    checkpoint_file.write_text(json.dumps(all_entities, indent=2))

    total = len(all_entities)
    s = sum(1 for e in all_entities if e["verdict"] == "SUPPORTED")
    ns = sum(1 for e in all_entities if e["verdict"] == "NOT_SUPPORTED")
    print(f"\n  Total: {total} entities ({s} supported, {ns} not supported)")
    print(f"  Saved to: {checkpoint_file}")
    return all_entities


def merge_partials():
    """Merge all partial entity files into exp9_entities.json."""
    partials = sorted(REPO_ROOT.glob("results/exp9_entities_partial_*.json"))
    if not partials:
        print("No partial files found")
        return

    all_entities = []
    seen = set()
    for p in partials:
        data = json.loads(p.read_text())
        for ent in data:
            key = (ent["completion_id"], ent["entity"])
            if key not in seen:
                seen.add(key)
                all_entities.append(ent)
        print(f"  {p.name}: {len(data)} entities")

    ENTITIES_FILE.write_text(json.dumps(all_entities, indent=2))
    total = len(all_entities)
    s = sum(1 for e in all_entities if e["verdict"] == "SUPPORTED")
    ns = sum(1 for e in all_entities if e["verdict"] == "NOT_SUPPORTED")
    print(f"\n  Merged: {total} entities ({s} supported, {ns} not supported)")
    print(f"  Saved to: {ENTITIES_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_partials()
        return

    if not COMPLETIONS_FILE.exists():
        print(f"ERROR: {COMPLETIONS_FILE} not found. Run generation first.")
        return

    completions = json.loads(COMPLETIONS_FILE.read_text())
    print(f"Loaded {len(completions)} completions")
    print(f"Processing range [{args.start}:{args.end or len(completions)}]")

    extract_and_verify(completions, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
