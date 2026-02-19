#!/usr/bin/env python3
"""
Exp9 Stage 9.1 only: Generate 600 completions from Gemma-3-12B-IT.
Writes results/exp9_completions.json then exits.
Run entity extraction separately with exp9_entity_extractor.py.
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-12b-it"
DEVICE = "cuda"
SEED = 42
N_COMPLETIONS = 4  # paper uses 4 per prompt; 50 x 4 = 200 completions
GEN_TEMP = 1.0
GEN_TOP_P = 0.95
GEN_TOP_K = 64
GEN_MAX_TOKENS = 1024

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

torch.manual_seed(SEED)

print("Loading model...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
)
model.eval()
_cfg = getattr(model.config, "text_config", model.config)
print(f"Loaded {MODEL_ID}: {_cfg.num_hidden_layers} layers, hidden={_cfg.hidden_size} ({time.time()-t0:.0f}s)")


def generate(prompt, seed=SEED, max_new_tokens=GEN_MAX_TOKENS):
    suffix = ("\n\nPlease provide a detailed, fact-dense response with specific names, "
              "dates, numbers, and citations. Be as precise as possible.")
    chat = [{"role": "user", "content": prompt + suffix}]
    result = tokenizer.apply_chat_template(
        chat, return_tensors="pt", add_generation_prompt=True,
    )
    if isinstance(result, torch.Tensor):
        input_ids = result.to(DEVICE)
        gen_kwargs = {"input_ids": input_ids}
        prompt_len = input_ids.shape[1]
    else:
        inputs = {k: v.to(DEVICE) for k, v in result.items()}
        gen_kwargs = inputs
        prompt_len = inputs["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **gen_kwargs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=GEN_TEMP, top_p=GEN_TOP_P, top_k=GEN_TOP_K,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def main():
    if CACHE_COMPLETIONS.exists():
        print(f"Cache exists: {CACHE_COMPLETIONS}")
        existing = json.loads(CACHE_COMPLETIONS.read_text())
        print(f"  {len(existing)} completions already cached, skipping.")
        return

    print(f"\nGenerating {len(PROMPTS)} x {N_COMPLETIONS} = {len(PROMPTS)*N_COMPLETIONS} completions")
    total = len(PROMPTS) * N_COMPLETIONS
    completions = []

    for i, prompt in enumerate(PROMPTS):
        for j in range(N_COMPLETIONS):
            seed = SEED + i * N_COMPLETIONS + j
            idx = i * N_COMPLETIONS + j
            t1 = time.time()
            print(f"  [{idx+1}/{total}] {prompt[:50]}... (seed={seed})", end="", flush=True)
            text = generate(prompt, seed=seed)
            elapsed = time.time() - t1
            print(f" [{elapsed:.1f}s, {len(text)} chars]")
            completions.append({
                "id": idx, "prompt_idx": i, "prompt": prompt,
                "seed": seed, "text": text,
            })

            # Checkpoint every 50
            if (idx + 1) % 50 == 0:
                tmp = REPO_ROOT / "results" / "exp9_completions_partial.json"
                tmp.write_text(json.dumps(completions, indent=2))
                print(f"  [checkpoint] {len(completions)} completions")

    CACHE_COMPLETIONS.write_text(json.dumps(completions, indent=2))
    print(f"\nSaved {len(completions)} completions to {CACHE_COMPLETIONS}")
    print(f"Total generation time: {time.time()-t0:.0f}s")
    print("\nNext: run exp9_entity_extractor.py to extract and verify entities")


if __name__ == "__main__":
    main()
