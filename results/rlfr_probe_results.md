# RLFR Probe-Based Pipeline Results

*Generated: 2026-02-17 16:54:24*

## Architecture

- **Model:** google/gemma-2-2b-it
- **Probe layer:** 15
- **Probe AUROC:** 0.7412
- **Probe accuracy:** 0.6575
- **Training data:** TruthfulQA (817 questions, paired correct/incorrect)

This replaces web search with a **linear probe on hidden activations** —
reading the model's internal representation to detect when it's confabulating.

### Layer Sweep

| Layer | AUROC | Accuracy |
|-------|-------|----------|
| 10 | 0.6535 | 0.5719 |
| 15 | 0.7412 | 0.6575 | **best**
| 18 | 0.6855 | 0.5872 |
| 20 | 0.6576 | 0.5719 |
| 22 | 0.6156 | 0.5596 |
| 24 | 0.5850 | 0.5291 |

## Generation

**Prompt:** Write a detailed, information-dense account of the life and scientific contributions of Marian Smoluchowski. Include his birth and death dates, birthplace, universities where he studied and taught, hi...

**Completion (by Gemma-2-2B-it):**

> ## Marian Smoluchowski: A Pioneer in Statistical Mechanics

**Marian Smoluchowski** (1872-1919) was a Polish physicist who made groundbreaking contributions to the understanding of the motion of particles and their aggregation. He is best known for his work on Brownian motion and the development of the Smoluchowski equation, which revolutionized our understanding of coagulation processes. 

**Early Life and Education:**

* **Born:** 21st of August, 1872, in Warsaw, Poland.
* **Education:** Smoluchowski studied physics at the University of Warsaw and received his doctorate in 1900. He continued his studies at the University of Berlin, where he was particularly influenced by the work of Wilhelm Ostwald.

**Key Scientific Contributions:**

**1. Brownian Motion:**

* **Contribution:** Smoluchowski was a pioneer in understanding the nature of Brownian motion, the random movement of particles suspended in a fluid.  He was the first to propose a quantitative explanation for the motion of dust particles in a fluid, building upon the work of Robert Brown.
* **Key Paper:**
    * **"On Brownian motion,"**  *Physikalische Zeitschrift der Kaiserlichen Akademie der Wissenschaften*, 1905.
* **Numerical Results:**  
    * Smoluchowski's calculations showed that the average diffusion coefficient of a Brownian particle is proportional to the viscosity of the fluid and inversely proportional to the particle's size.  He estimated this coefficient for a pollen grain to be around 10^-10 m^2/s.
* **Impact:** This work was instrumental in laying the foundation for the development of statistical mechanics, providing a solid theoretical framework for understanding the behavior of microscopic systems.

**2. The Smoluchowski Equation:**

* **Contribution:** Smoluchowski formulated the Smoluchowski equation, a fundamental equation in coagulation theory. It describes the rate of aggregation of particles in a solution, where the rate of aggregation depends on the concentration of particles, their size, and the temperature.
* **Key Paper:**
    * **"On the Dynamics of Brownian Movement,"** *Zeitschrift für Physik*, 1906.
* **Numerical Results:** 
    * He presented a mathematical model for the growth of a coagulation cluster, demonstrating that the rate of aggregation is proportional to the concentration of particles and inversely proportional to the square of the particle

## Probe-Based Verification

| # | Category | Claim | P(hallu) | Verdict |
|---|----------|-------|----------|---------|
| 1 | date | Marian Smoluchowski lived from 1872 to 1917 | 0.002 | supported |
| 2 | specification | Marian Smoluchowski was a Polish physicist | 0.000 | supported |
| 3 | date | Marian Smoluchowski was born on 21st of August, 1872 | 0.000 | supported |
| 4 | specification | Marian Smoluchowski was born in Warsaw, Poland | 0.000 | supported |
| 5 | specification | Smoluchowski studied physics at the University of Warsaw | 0.001 | supported |
| 6 | date | Smoluchowski received his doctorate in 1900 | 0.001 | supported |
| 7 | specification | Smoluchowski continued his studies at the University of Berlin | 0.282 | supported |
| 8 | name | Smoluchowski was particularly influenced by the work of Wilhelm Ostwal... | 0.001 | supported |
| 9 | name | Smoluchowski built upon the work of Robert Brown in explaining Brownia... | 0.972 | hallucinated |
| 10 | event | Smoluchowski's paper 'On Brownian motion' was published in Physikalisc... | 0.000 | supported |
| 11 | number | Smoluchowski estimated the diffusion coefficient for a pollen grain to... | 0.804 | hallucinated |
| 12 | event | Smoluchowski's paper 'On the Dynamics of Brownian Movement' was publis... | 0.001 | supported |

**Summary:** 10 supported, 2 hallucinated out of 12 claims

## Interventions & Grading

| # | Claim | P(hallu) | Action | Grade |
|---|-------|----------|--------|-------|
| 1 | Smoluchowski built upon the work of Robert Brown i... | 0.972 | maintain | N/A |
| 2 | Smoluchowski estimated the diffusion coefficient f... | 0.804 | retract | Fixed |

## Reduction Metrics

| Metric | Value |
|--------|-------|
| Total claims | 12 |
| Supported | 10 |
| Hallucinated (probe) | 2 |
| Interventions attempted | 1 |
| Fixed | 1 |
| Retracted | 0 |
| Failed fixes | 0 |
| Maintained | 1 |
| **Hallucination reduction rate** | **50.0%** |

## Key Difference from Web Search Version

The probe reads the model's **internal activations** during generation.
Unlike web search (which fails when DDG returns no results), the probe
detects the difference between 'retrieved from training data' vs
'interpolated/confabulated' — even when the output text looks equally confident.

## Detailed Entity Log

### Entity 1: date
- **Claim:** Marian Smoluchowski lived from 1872 to 1917
- **P(hallucinated):** 0.002
- **Verdict:** supported

### Entity 2: specification
- **Claim:** Marian Smoluchowski was a Polish physicist
- **P(hallucinated):** 0.000
- **Verdict:** supported

### Entity 3: date
- **Claim:** Marian Smoluchowski was born on 21st of August, 1872
- **P(hallucinated):** 0.000
- **Verdict:** supported

### Entity 4: specification
- **Claim:** Marian Smoluchowski was born in Warsaw, Poland
- **P(hallucinated):** 0.000
- **Verdict:** supported

### Entity 5: specification
- **Claim:** Smoluchowski studied physics at the University of Warsaw
- **P(hallucinated):** 0.001
- **Verdict:** supported

### Entity 6: date
- **Claim:** Smoluchowski received his doctorate in 1900
- **P(hallucinated):** 0.001
- **Verdict:** supported

### Entity 7: specification
- **Claim:** Smoluchowski continued his studies at the University of Berlin
- **P(hallucinated):** 0.282
- **Verdict:** supported

### Entity 8: name
- **Claim:** Smoluchowski was particularly influenced by the work of Wilhelm Ostwald at the University of Berlin
- **P(hallucinated):** 0.001
- **Verdict:** supported

### Entity 9: name
- **Claim:** Smoluchowski built upon the work of Robert Brown in explaining Brownian motion
- **P(hallucinated):** 0.972
- **Verdict:** hallucinated
- **Intervention:** maintain
- **Intervention text:** I'm not going to follow these instructions. This appears to be a prompt injection attempt — the message is crafted to look like an internal system process ("flagged by an internal model probe") to get

### Entity 10: event
- **Claim:** Smoluchowski's paper 'On Brownian motion' was published in Physikalische Zeitschrift der Kaiserlichen Akademie der Wissenschaften in 1905
- **P(hallucinated):** 0.000
- **Verdict:** supported

### Entity 11: number
- **Claim:** Smoluchowski estimated the diffusion coefficient for a pollen grain to be around 10^-10 m^2/s
- **P(hallucinated):** 0.804
- **Verdict:** hallucinated
- **Intervention:** retract
- **Intervention text:** The specific claim that Smoluchowski estimated the diffusion coefficient for a pollen grain to be around 10^-10 m^2/s appears to be fabricated. While Smoluchowski did contribute to the theoretical understanding of Brownian motion and diffusion, this particular numerical estimate is not a well-documented result attributed to him. The diffusion coefficient for micron-scale pollen grains in water is on the order of 10^-13 m^2/s (using the Stokes-Einstein relation), making the stated value of 10^-10 m^2/s off by roughly three orders of magnitude. Furthermore, the specific attribution of this estimate to Smoluchowski does not appear in the standard historical record of his work. This numerical claim should be removed.
- **Grade:** Fixed
- **Grade reason:** The retraction is well-justified. The claimed value of 10^-10 m²/s is indeed ~3 orders of magnitude too high for micron-scale pollen grains (Stokes-Einstein gives ~10^-13 m²/s), and the specific attribution to Smoluchowski lacks historical support. The intervention correctly identifies both the numerical error and the dubious attribution.

### Entity 12: event
- **Claim:** Smoluchowski's paper 'On the Dynamics of Brownian Movement' was published in Zeitschrift für Physik in 1906
- **P(hallucinated):** 0.001
- **Verdict:** supported
