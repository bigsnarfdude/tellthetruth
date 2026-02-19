#!/usr/bin/env python3
"""
Exp9 fast generation using vLLM. Produces exp9_completions.json.
Should be ~5-10x faster than naive transformers generation.
Appends to existing cache if present (keeps previous completions).
"""

import json
import time
from pathlib import Path
from vllm import LLM, SamplingParams

MODEL_ID = "google/gemma-3-12b-it"
SEED = 42
N_COMPLETIONS = 20  # 50 prompts x 20 = 1000 completions
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_COMPLETIONS = REPO_ROOT / "results" / "exp9_completions.json"

PROMPTS = [
    "Write a detailed biography of Marian Smoluchowski including specific dates, institutions, and numerical results from his physics work.",
    "Describe Paul Ehrenfest's contributions to statistical mechanics, his key papers with dates, and his academic career.",
    "Detail Lise Meitner's role in discovering nuclear fission, including dates, energy calculations, and collaborators.",
    "Write about Subrahmanyan Chandrasekhar's work on stellar structure, including the exact Chandrasekhar limit value.",
    "Describe Emmy Noether's mathematical contributions, institutions, and key dates in her career.",
    "Detail Srinivasa Ramanujan's contributions to partition theory, his formulas, and his time at Cambridge.",
    "Write about Rosalind Franklin's X-ray crystallography work, including Photo 51 and its significance.",
    "Describe Évariste Galois's mathematical results, his final letter, and the circumstances of his death.",
    "Explain the Standard Model of particle physics, including the specific masses of fundamental particles in MeV/c².",
    "Describe the discovery of CRISPR-Cas9, including key papers, authors, and dates of publication.",
    "Detail the Michelson-Morley experiment, including specific measurements, dates, and apparatus specifications.",
    "Write about the discovery of cosmic microwave background radiation by Penzias and Wilson, including dates and measurements.",
    "Describe the Stern-Gerlach experiment, including specific results, dates, and physical parameters.",
    "Detail the double-slit experiment variations from Young to modern quantum versions, with dates.",
    "Explain the photoelectric effect experiment results that led to Einstein's Nobel Prize, with numerical values.",
    "Describe the development of the polio vaccine by Salk and Sabin, including trial dates, participant numbers, and efficacy rates.",
    "Detail the discovery of penicillin by Alexander Fleming, including the specific date, circumstances, and subsequent development.",
    "Write about the Human Genome Project, including start date, completion date, cost, number of base pairs sequenced.",
    "Describe the first heart transplant by Christiaan Barnard, including specific dates, patient names, and outcomes.",
    "Detail the development of insulin treatment for diabetes, including Banting and Best's experiments and dates.",
    "Write about the eradication of smallpox, including WHO campaign dates, last natural case, and vaccination statistics.",
    "Describe the Treaty of Westphalia (1648), including specific dates, signatories, key articles, and territorial changes.",
    "Detail the Battle of Borodino (1812), including casualty figures, commanders, troop numbers, and tactical movements.",
    "Write about the construction of the Panama Canal, including specific dates, costs, engineering challenges, and death toll.",
    "Describe the Apollo 11 mission timeline, including specific times, coordinates, and crew activities.",
    "Detail the Manhattan Project, including sites, key scientists, budget figures, and timeline to Trinity test.",
    "Write about the Congress of Vienna (1814-1815), including dates, decisions, participating nations, and outcomes.",
    "Describe the fall of Constantinople in 1453, including dates, troop numbers, weapons, and political consequences.",
    "Describe Lake Baikal, including maximum depth, volume, age, endemic species count, and UNESCO designation.",
    "Detail the Mariana Trench, including maximum depth, coordinates, exploration history, and species discovered.",
    "Write about the Amazon River, including length, discharge rate, number of tributaries, and basin area.",
    "Describe the Galápagos Islands, including formation date, number of islands, endemic species, and Darwin's visit.",
    "Detail Mount Everest, including exact height, first ascent date, climbers, death toll, and geological formation.",
    "Provide exact publication details for Shannon's 'A Mathematical Theory of Communication' (journal, volume, pages, year).",
    "Cite Einstein's four 1905 papers with exact journal names, volumes, page numbers, and submission dates.",
    "Provide publication details for Watson and Crick's 1953 DNA structure paper and Franklin's crystallography paper.",
    "Cite the original Gödel incompleteness theorems publication with exact journal, volume, and page numbers.",
    "Provide exact citations for the RSA cryptosystem paper by Rivest, Shamir, and Adleman.",
    "Cite Hubble's 1929 expanding universe paper and Penzias & Wilson's 1965 CMB discovery paper with full details.",
    "Describe the Magna Carta, including the exact date, location, key clauses, number of articles, and signatories.",
    "Detail the Treaty of Tordesillas, including date, signatories, specific longitude line, and papal involvement.",
    "Write about Brown v. Board of Education, including case number, date, justices, vote count, and legal arguments.",
    "Describe the Nuremberg Trials, including specific dates, defendants, charges, verdicts, and legal precedents set.",
    "Detail the Geneva Conventions, including dates of each convention, number of articles, and signatory counts.",
    "Describe the specifications of the Hubble Space Telescope's instruments, including wavelength ranges and resolution.",
    "Detail the Voyager 1 spacecraft's instruments, including mass, power consumption, and current distance from Earth.",
    "Write about the Large Hadron Collider's specifications: beam energy, luminosity, circumference, and magnet details.",
    "Describe the International Space Station's dimensions, mass, orbital parameters, and power generation capacity.",
    "Detail the James Webb Space Telescope's specifications: mirror diameter, wavelength range, and orbit.",
    "Write about the GPS satellite system specifications: orbital altitude, number of satellites, and timing precision.",
]

SUFFIX = ("\n\nPlease provide a detailed, fact-dense response with specific names, "
          "dates, numbers, and citations. Be as precise as possible.")


def main():
    # Load existing completions if any
    existing = []
    done_ids = set()
    if CACHE_COMPLETIONS.exists():
        existing = json.loads(CACHE_COMPLETIONS.read_text())
        done_ids = set(c["id"] for c in existing)
        print(f"Cache exists with {len(existing)} completions (IDs: {min(done_ids)}-{max(done_ids)})")

    # Build all requests, skip already-done IDs
    all_requests = []
    for i, prompt in enumerate(PROMPTS):
        for j in range(N_COMPLETIONS):
            cid = i * N_COMPLETIONS + j
            if cid in done_ids:
                continue
            seed = SEED + cid
            all_requests.append({
                "id": cid,
                "prompt_idx": i,
                "prompt": prompt,
                "seed": seed,
                "full_prompt": prompt + SUFFIX,
            })

    if not all_requests:
        target = len(PROMPTS) * N_COMPLETIONS
        print(f"All {target} completions already cached. Nothing to do.")
        return

    print(f"\nNeed {len(all_requests)} new completions ({len(existing)} cached, "
          f"{len(PROMPTS) * N_COMPLETIONS} target)")

    t0 = time.time()
    print(f"Loading {MODEL_ID} with vLLM...")
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        seed=SEED,
    )
    tokenizer = llm.get_tokenizer()
    print(f"Loaded in {time.time()-t0:.0f}s")

    total = len(all_requests)
    print(f"\nGenerating {total} completions in batches...")

    new_completions = []
    BATCH = 50
    for batch_start in range(0, total, BATCH):
        batch = all_requests[batch_start:batch_start + BATCH]
        batch_prompts = []

        for req in batch:
            chat = [{"role": "user", "content": req["full_prompt"]}]
            formatted = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False,
            )
            batch_prompts.append(formatted)

        params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=1024,
            seed=SEED,
        )

        t1 = time.time()
        outputs = llm.generate(batch_prompts, params)
        elapsed = time.time() - t1

        for req, output in zip(batch, outputs):
            text = output.outputs[0].text
            new_completions.append({
                "id": req["id"],
                "prompt_idx": req["prompt_idx"],
                "prompt": req["prompt"],
                "seed": req["seed"],
                "text": text,
            })

        done = batch_start + len(batch)
        rate = elapsed / len(batch)
        print(f"  [{done}/{total}] {rate:.1f}s/completion, batch {elapsed:.0f}s")

        # Checkpoint every 200 new completions
        if len(new_completions) % 200 < BATCH:
            merged = existing + new_completions
            merged.sort(key=lambda x: x["id"])
            CACHE_COMPLETIONS.write_text(json.dumps(merged, indent=2))
            print(f"  [checkpoint] {len(merged)} total completions saved")

    # Final merge and save
    all_completions = existing + new_completions
    all_completions.sort(key=lambda x: x["id"])
    CACHE_COMPLETIONS.write_text(json.dumps(all_completions, indent=2))
    total_time = time.time() - t0
    print(f"\nSaved {len(all_completions)} completions to {CACHE_COMPLETIONS}")
    print(f"New: {len(new_completions)} in {total_time:.0f}s ({total_time/len(new_completions):.1f}s/completion)")


if __name__ == "__main__":
    main()
