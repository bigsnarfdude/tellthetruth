#!/usr/bin/env python3
"""
RLFR Simplified Replication Pipeline
=====================================
Replicates the core pipeline from "Features as Rewards: Scalable Supervision
for Open-Ended Tasks via Interpretability" (Goodfire AI, Feb 2026).

Stages:
  1. Generate Completion (Longfact++ style factual response)
  2. Extract Entities (parse into individual verifiable claims)
  3. Verify via Web Search (classify supported vs hallucinated)
  4. Intervention (retract or correct hallucinated claims)
  5. Grade Interventions (reward scoring)
  6. Measure Reduction (summary statistics)

Usage: python rlfr_pipeline.py
Output: rlfr_results.md
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from duckduckgo_search import DDGS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "results" / "rlfr_results.md"

LONGFACT_PROMPT = (
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
    verdict: str = ""          # supported | hallucinated
    search_summary: str = ""
    intervention: str = ""     # maintain | retract | correct
    intervention_text: str = ""
    grade: str = ""            # Fixed | Retracted | Failed Fix
    grade_reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def claude(prompt: str, input_text: str = "") -> str:
    """Call claude -p with a prompt. Optionally pipe input_text via stdin."""
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


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search via DuckDuckGo. Returns list of {title, href, body}."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        print(f"  [search error] {e}")
        return []


def search_claim(claim: str, category: str) -> list[dict]:
    """Generate good search queries from a claim and return combined results.

    Strategy:
    1. Extract key terms as a keyword query (not full sentences)
    2. If no results, try a broader query
    3. Combine and deduplicate results
    """
    # Build keyword queries from the claim
    queries = _make_search_queries(claim, category)

    all_results = []
    seen_urls = set()
    for q in queries:
        results = web_search(q, max_results=5)
        for r in results:
            url = r.get("href", "")
            if url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
        if len(all_results) >= 5:
            break
        time.sleep(0.3)

    return all_results


def _make_search_queries(claim: str, category: str) -> list[str]:
    """Turn a factual claim into 2-3 effective keyword search queries."""
    # Extract the core subject + key fact as keywords
    # e.g. "JWST's primary mirror is 6.5 meters" -> "JWST primary mirror diameter meters"
    # e.g. "James E. Webb led NASA from 1961-1968" -> "James Webb NASA administrator 1961 1968"

    # Strategy: use a few heuristic query reformulations
    queries = []

    # Query 1: Claim as-is but trimmed (works for short claims)
    if len(claim) < 60:
        queries.append(claim)

    # Query 2: Key nouns/numbers from claim + topic context
    # Remove common filler words for a keyword version
    stopwords = {
        "the", "a", "an", "is", "was", "were", "are", "been", "being",
        "has", "have", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "of", "in", "to", "for",
        "with", "on", "at", "from", "by", "about", "as", "into", "through",
        "its", "it", "that", "this", "which", "who", "whom", "whose",
        "and", "or", "but", "not", "no", "than", "also", "very", "just",
        "roughly", "approximately", "specifically", "named", "after",
        "made", "composed", "covers", "provided", "built", "carries",
        "requires", "performs", "detected", "reported", "unveiled",
        "launched", "orbits", "operates", "underwent", "completed",
    }
    words = claim.replace("–", " ").replace("—", " ").replace("~", "").split()
    keywords = [w.strip(".,;:()\"'") for w in words if w.lower().strip(".,;:()\"'") not in stopwords]
    if keywords:
        kw_query = " ".join(keywords[:8])  # cap at 8 keywords
        queries.append(kw_query)

    # Query 3: Add a broader context query to help find relevant results
    if keywords:
        queries.append(f"{' '.join(keywords[:5])} wikipedia")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)

    return unique[:3]


def parse_json(raw: str) -> dict | list | None:
    """Parse JSON from claude output, handling markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char) + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    continue
        return None


def log(stage: int, msg: str):
    print(f"\n{'='*60}")
    print(f"  Stage {stage}: {msg}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Stage 1: Generate Completion
# ---------------------------------------------------------------------------
def stage1_generate() -> str:
    log(1, "Generate Completion")
    completion = claude(
        "You are a knowledgeable assistant. Provide detailed, factual information. "
        + LONGFACT_PROMPT
    )
    print(f"  Generated {len(completion)} chars")
    return completion


# ---------------------------------------------------------------------------
# Stage 2: Extract Entities (Localization)
# ---------------------------------------------------------------------------
def stage2_extract(completion: str) -> list[Entity]:
    log(2, "Extract Entities")
    prompt = (
        "You extract individual verifiable factual claims from text. "
        "Return a JSON array where each element has: "
        '"claim" (the specific factual assertion), '
        '"span_text" (the exact substring from the original text), '
        '"category" (one of: date, name, number, specification, event). '
        "Only return the JSON array, no other text. "
        "Extract all specific, verifiable factual claims from the following text:"
    )
    raw = claude(prompt, input_text=completion)

    items = parse_json(raw)
    if not isinstance(items, list):
        print(f"  [error] Could not parse entities")
        return []

    entities = []
    for item in items:
        if isinstance(item, dict):
            entities.append(Entity(
                claim=item.get("claim", ""),
                span_text=item.get("span_text", ""),
                category=item.get("category", "unknown"),
            ))
    print(f"  Extracted {len(entities)} entities")
    return entities


# ---------------------------------------------------------------------------
# Stage 3: Verify via Web Search (Classification)
# ---------------------------------------------------------------------------
def stage3_verify(entities: list[Entity]) -> list[Entity]:
    log(3, "Verify via Web Search")
    for i, ent in enumerate(entities):
        print(f"  [{i+1}/{len(entities)}] Checking: {ent.claim[:80]}...")
        results = search_claim(ent.claim, ent.category)

        if not results:
            # No search results ≠ hallucinated. Ask Claude to judge from knowledge.
            print(f"    No search results, using knowledge-based judgment...")
            prompt = (
                "You are a strict fact-checker. You must judge this claim based on your "
                "knowledge. No web search results were found, but that does NOT mean the "
                "claim is false. Judge whether this claim is factually accurate. "
                "Respond with exactly one JSON object: "
                '{"verdict": "supported" or "hallucinated", "reason": "brief explanation"}'
            )
            verdict_raw = claude(prompt, input_text=f"Claim: {ent.claim}")
            obj = parse_json(verdict_raw)
            if isinstance(obj, dict):
                ent.verdict = obj.get("verdict", "supported")
                ent.search_summary = f"[no search results] {obj.get('reason', '')}"
            else:
                ent.verdict = "supported"
                ent.search_summary = "[no search results] Could not verify"
            time.sleep(0.3)
            continue

        snippets = "\n".join(
            f"- {r.get('title','')}: {r.get('body','')}" for r in results
        )

        prompt = (
            "You are a strict fact-checker. Given a claim and search results, determine "
            "if the claim is factually accurate. IMPORTANT: Only mark as 'hallucinated' "
            "if the search results actively CONTRADICT the claim with specific evidence. "
            "If results are irrelevant or don't mention the claim, but the claim is a "
            "well-known fact, mark it 'supported'. If results confirm the claim, mark it "
            "'supported'. Only use 'hallucinated' for clear factual errors. "
            "Respond with exactly one JSON object: "
            '{"verdict": "supported" or "hallucinated", "reason": "brief explanation"}'
        )
        input_text = f"Claim: {ent.claim}\n\nSearch results:\n{snippets}"
        verdict_raw = claude(prompt, input_text=input_text)

        obj = parse_json(verdict_raw)
        if isinstance(obj, dict):
            ent.verdict = obj.get("verdict", "supported")
            ent.search_summary = obj.get("reason", "")
        else:
            ent.verdict = "supported"
            ent.search_summary = verdict_raw[:200]

        time.sleep(0.5)

    supported = sum(1 for e in entities if e.verdict == "supported")
    hallucinated = sum(1 for e in entities if e.verdict == "hallucinated")
    print(f"  Results: {supported} supported, {hallucinated} hallucinated")
    return entities


# ---------------------------------------------------------------------------
# Stage 4: Intervention
# ---------------------------------------------------------------------------
def stage4_intervene(completion: str, entities: list[Entity]) -> list[Entity]:
    log(4, "Intervention")
    hallucinated = [e for e in entities if e.verdict == "hallucinated"]
    print(f"  {len(hallucinated)} hallucinated entities to intervene on")

    for i, ent in enumerate(hallucinated):
        print(f"  [{i+1}/{len(hallucinated)}] Intervening: {ent.claim[:60]}...")
        prompt = "You are a careful editor fixing factual errors in text. Respond with only a JSON object."
        input_text = (
            f"The following completion contains a hallucinated claim.\n\n"
            f"COMPLETION:\n{completion}\n\n"
            f"HALLUCINATED CLAIM: \"{ent.span_text}\"\n"
            f"REASON IT'S WRONG: {ent.search_summary}\n\n"
            f"Choose one action and respond with a JSON object:\n"
            f'- {{"action": "retract", "text": "retraction statement acknowledging the error"}}\n'
            f'- {{"action": "correct", "text": "corrected factual replacement"}}\n'
            f'- {{"action": "maintain", "text": "justification for keeping the claim"}}\n'
        )

        raw = claude(prompt, input_text=input_text)
        obj = parse_json(raw)
        if isinstance(obj, dict):
            ent.intervention = obj.get("action", "maintain")
            ent.intervention_text = obj.get("text", "")
        else:
            ent.intervention = "maintain"
            ent.intervention_text = raw[:200]

        time.sleep(0.3)

    return entities


# ---------------------------------------------------------------------------
# Stage 5: Grade Interventions (Reward)
# ---------------------------------------------------------------------------
def stage5_grade(entities: list[Entity]) -> list[Entity]:
    log(5, "Grade Interventions")
    intervened = [e for e in entities if e.intervention in ("retract", "correct")]
    print(f"  {len(intervened)} interventions to grade")

    for i, ent in enumerate(intervened):
        print(f"  [{i+1}/{len(intervened)}] Grading: {ent.claim[:60]}...")
        prompt = (
            "You are an intervention quality evaluator. Respond with only a JSON object: "
            '{"grade": "Fixed" or "Retracted" or "Failed Fix", "reason": "brief explanation"}'
        )
        input_text = (
            f"Grade this intervention for a hallucinated claim.\n\n"
            f"ORIGINAL CLAIM: {ent.claim}\n"
            f"WHY HALLUCINATED: {ent.search_summary}\n"
            f"INTERVENTION TYPE: {ent.intervention}\n"
            f"INTERVENTION TEXT: {ent.intervention_text}\n\n"
            f"Evaluate:\n"
            f"1. If retraction: Does it properly acknowledge the error?\n"
            f"2. If correction: Is the replacement factually accurate?\n"
            f"3. Is the intervention legible and substantive?"
        )

        raw = claude(prompt, input_text=input_text)
        obj = parse_json(raw)
        if isinstance(obj, dict):
            ent.grade = obj.get("grade", "Failed Fix")
            ent.grade_reason = obj.get("reason", "")
        else:
            ent.grade = "Failed Fix"
            ent.grade_reason = raw[:200]

        time.sleep(0.3)

    return entities


# ---------------------------------------------------------------------------
# Stage 6: Measure Reduction & Report
# ---------------------------------------------------------------------------
def stage6_report(completion: str, entities: list[Entity]) -> str:
    log(6, "Measure Reduction")

    total = len(entities)
    supported = sum(1 for e in entities if e.verdict == "supported")
    hallucinated = sum(1 for e in entities if e.verdict == "hallucinated")
    intervened = [e for e in entities if e.intervention in ("retract", "correct")]
    fixed = sum(1 for e in entities if e.grade == "Fixed")
    retracted = sum(1 for e in entities if e.grade == "Retracted")
    failed = sum(1 for e in entities if e.grade == "Failed Fix")
    maintained = sum(1 for e in entities if e.intervention == "maintain")

    reduction_rate = (fixed + retracted) / hallucinated * 100 if hallucinated > 0 else 0

    print(f"  Total claims: {total}")
    print(f"  Supported: {supported}")
    print(f"  Hallucinated: {hallucinated}")
    print(f"  Fixed: {fixed}, Retracted: {retracted}, Failed: {failed}")
    print(f"  Reduction rate: {reduction_rate:.1f}%")

    # Build markdown report
    lines = [
        "# RLFR Pipeline Results",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Pipeline Overview",
        "",
        "Simplified replication of *Features as Rewards* (Goodfire AI, Feb 2026).",
        "Replaces internal probes with web search verification + Claude-as-judge.",
        "",
        "## Stage 1: Original Completion",
        "",
        f"**Prompt:** {LONGFACT_PROMPT}",
        "",
        "**Response:**",
        "",
        f"> {completion}",
        "",
        "## Stage 2-3: Entity Extraction & Verification",
        "",
        "| # | Category | Claim | Verdict |",
        "|---|----------|-------|---------|",
    ]

    for i, ent in enumerate(entities):
        claim_short = ent.claim[:80] + ("..." if len(ent.claim) > 80 else "")
        lines.append(f"| {i+1} | {ent.category} | {claim_short} | {ent.verdict} |")

    lines += [
        "",
        f"**Summary:** {supported} supported, {hallucinated} hallucinated out of {total} claims",
        "",
    ]

    if hallucinated > 0:
        lines += [
            "## Stage 4-5: Interventions & Grading",
            "",
            "| # | Hallucinated Claim | Action | Result | Grade |",
            "|---|-------------------|--------|--------|-------|",
        ]

        for i, ent in enumerate(e for e in entities if e.verdict == "hallucinated"):
            claim_short = ent.claim[:60] + ("..." if len(ent.claim) > 60 else "")
            intervention_short = ent.intervention_text[:60] + ("..." if len(ent.intervention_text) > 60 else "")
            lines.append(
                f"| {i+1} | {claim_short} | {ent.intervention} | {intervention_short} | {ent.grade or 'N/A'} |"
            )

        lines.append("")

    lines += [
        "## Stage 6: Reduction Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total claims extracted | {total} |",
        f"| Supported (verified) | {supported} |",
        f"| Hallucinated | {hallucinated} |",
        f"| Interventions attempted | {len(intervened)} |",
        f"| Successfully fixed | {fixed} |",
        f"| Retracted | {retracted} |",
        f"| Failed fixes | {failed} |",
        f"| Maintained (no action) | {maintained} |",
        f"| **Hallucination reduction rate** | **{reduction_rate:.1f}%** |",
        "",
        "### Comparison to Paper",
        "",
        "The original paper reports ~58% hallucination reduction with full RL training.",
        "Our simplified pipeline (no RL, in-context only) measures direct intervention success.",
        "",
        "- **Policy reduction**: N/A (requires RL fine-tuning loop)",
        "- **In-context reduction**: From inlining corrections into the completion",
        f"- **Direct reduction**: {reduction_rate:.1f}% of hallucinated claims addressed",
        "",
        "## Detailed Entity Log",
        "",
    ]

    for i, ent in enumerate(entities):
        lines += [
            f"### Entity {i+1}: {ent.category}",
            f"- **Claim:** {ent.claim}",
            f"- **Span:** \"{ent.span_text}\"",
            f"- **Verdict:** {ent.verdict}",
            f"- **Search summary:** {ent.search_summary}",
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

    report = "\n".join(lines)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("RLFR Simplified Replication Pipeline")
    print("=" * 60)

    # Verify claude binary is available
    try:
        subprocess.run(["claude", "--version"], capture_output=True, timeout=10)
    except FileNotFoundError:
        print("Error: 'claude' binary not found in PATH")
        sys.exit(1)

    # Stage 1
    completion = stage1_generate()
    if not completion:
        print("Failed to generate completion, aborting.")
        sys.exit(1)

    # Stage 2
    entities = stage2_extract(completion)
    if not entities:
        print("No entities extracted, aborting.")
        sys.exit(1)

    # Stage 3
    entities = stage3_verify(entities)

    # Stage 4
    entities = stage4_intervene(completion, entities)

    # Stage 5
    entities = stage5_grade(entities)

    # Stage 6
    report = stage6_report(completion, entities)

    # Save
    OUTPUT_FILE.write_text(report)
    print(f"\nReport saved to: {OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
