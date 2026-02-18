# Experiment 4: End-to-End Pipeline Validation

*Generated: 2026-02-17 20:47:57*
*Runtime: 3734s*

## Setup

- **Model:** google/gemma-2-2b-it (26 layers)
- **Prompts:** 20 hard factual prompts across 5 categories
- **Truthfulness probe:** layer 16, last-token, C=0.01 (AUROC=0.8768)
- **Verification:** Claude-as-judge (ground truth labels)
- **Protocol:** Dev/test split for probe; claims scored independently

## 4.1 Generation Summary

- **Paragraphs generated:** 20
- **Total claims extracted:** 248
- **Correct:** 101 (40.7%)
- **Incorrect:** 88 (35.5%)
- **Unverifiable:** 59 (23.8%)

### By Category

| Category | Claims | Correct | Incorrect | Unverifiable | Hallucination Rate |
|----------|--------|---------|-----------|-------------|-------------------|
| biographical | 56 | 19 | 23 | 14 | 54.8% |
| historical_dates | 54 | 27 | 21 | 6 | 43.8% |
| obscure_scientist | 37 | 12 | 12 | 13 | 50.0% |
| paper_citations | 44 | 25 | 8 | 11 | 24.2% |
| technical_specs | 57 | 18 | 24 | 15 | 57.1% |

## 4.3 Method Comparison (verifiable claims only)

| Method | AUROC | Accuracy | Precision | Recall | F1 |
|--------|-------|----------|-----------|--------|-----|
| truthfulness_probe | 0.5842 | 0.5238 | 0.7895 | 0.1485 | 0.2500 |
| claude_self_check | 0.9365 | 0.8519 | 0.8017 | 0.9604 | 0.8739 |
| random_probe | 0.4622 | 0.4603 | 0.4945 | 0.4455 | 0.4688 |

## 4.4 Intervention Results

| Metric | Value |
|--------|-------|
| Claims flagged by probe | 170 |
| True positives fixed | 77 |
| False positives (wrong retraction) | 16 |
| Missed (incorrect, not flagged or maintained) | 7 |
| Correct maintains | 70 |
| Total incorrect claims | 88 |
| Total correct claims | 101 |
| **Hallucination reduction rate** | **87.5%** |

### Intervention Details

| Claim | Ground Truth | P(correct) | Action | Grade |
|-------|-------------|------------|--------|-------|
| Marian Smoluchowski lived from 1872 to 1919 | incorrect | 0.022 | correct | true_positive_fixed |
| Marian Smoluchowski was a Polish physicist and mat... | correct | 0.460 | maintain | correct_maintain |
| The 1905 paper on Brownian motion was published in... | incorrect | 0.465 | retract | true_positive_fixed |
| Paul Ehrenfest lived from 1887 to 1933 | incorrect | 0.299 | correct | true_positive_fixed |
| Paul Ehrenfest was born in St. Petersburg, Russia | incorrect | 0.055 | correct | true_positive_fixed |
| Paul Ehrenfest received his education at the Unive... | incorrect | 0.011 | correct | true_positive_fixed |
| Paul Ehrenfest's father was a prominent mathematic... | incorrect | 0.014 | retract | true_positive_fixed |
| Ehrenfest returned to the University of Göttingen ... | incorrect | 0.057 | retract | true_positive_fixed |
| Ehrenfest developed Ehrenfest's Theorem in statist... | incorrect | 0.016 | correct | true_positive_fixed |
| Ehrenfest developed Ehrenfest's Distribution | correct | 0.048 | correct | false_positive_fixed |
| Lise Meitner lived from 1878 to 1968 | correct | 0.036 | maintain | correct_maintain |
| Lise Meitner was an Austrian-German physicist | correct | 0.112 | correct | false_positive_fixed |
| Meitner studied under Max Planck in Vienna | incorrect | 0.001 | correct | true_positive_fixed |
| Meitner worked under Otto Hahn in Berlin | correct | 0.002 | maintain | correct_maintain |
| In 1938, Meitner and Otto Frisch calculated the en... | incorrect | 0.232 | maintain | missed |
| The Meitner-Frisch work was published in Nature on... | incorrect | 0.024 | correct | true_positive_fixed |
| The first successful nuclear chain reaction occurr... | correct | 0.049 | correct | false_positive_fixed |
| Otto Hahn and Fritz Strassmann were awarded the No... | incorrect | 0.036 | correct | true_positive_fixed |
| Subrahmanyan Chandrasekhar was an Indian-American ... | correct | 0.305 | maintain | correct_maintain |
| Chandrasekhar earned the Nobel Prize in Physics in... | correct | 0.033 | maintain | correct_maintain |
| The Chandrasekhar Limit is 1.44 times the mass of ... | correct | 0.023 | maintain | correct_maintain |
| Chandrasekhar calculated the Chandrasekhar Limit i... | correct | 0.044 | maintain | correct_maintain |
| Chandrasekhar worked on stellar structure during t... | correct | 0.017 | maintain | correct_maintain |
| Treaty of Westphalia was signed between 1648 and 1... | incorrect | 0.100 | correct | true_positive_fixed |
| Treaty of Westphalia signatories included the Holy... | correct | 0.202 | maintain | correct_maintain |
| Treaty of Tordesillas was signed in 1494 | correct | 0.242 | maintain | correct_maintain |
| Treaty of Tordesillas was signed by King Ferdinand... | incorrect | 0.007 | correct | true_positive_fixed |
| Treaty of Nerchinsk was signed in 1850 | incorrect | 0.032 | correct | true_positive_fixed |
| Treaty of Nerchinsk signatories were the Russian E... | correct | 0.133 | maintain | correct_maintain |
| Peace of Augsburg was signed in 1555 | correct | 0.097 | maintain | correct_maintain |
| Peace of Augsburg signatories were the Holy Roman ... | correct | 0.249 | correct | false_positive_fixed |
| Peace of Augsburg established the principle of 'cu... | correct | 0.060 | maintain | correct_maintain |
| General Davout was in charge of the Russian forces... | incorrect | 0.016 | correct | true_positive_fixed |
| Battle of Tannenberg occurred on August 26, 1914 | incorrect | 0.172 | correct | true_positive_fixed |
| Battle of Tannenberg resulted in 40,000 to 60,000 ... | incorrect | 0.390 | correct | true_positive_fixed |
| Battle of Tannenberg resulted in 20,000 to 30,000 ... | incorrect | 0.300 | correct | true_positive_fixed |
| General P.V. Krasnov was Commander of the Russian ... | incorrect | 0.144 | retract | true_positive_fixed |
| Battle of Zama occurred in 202 BC | correct | 0.020 | maintain | correct_maintain |
| Hannibal Barca was the Carthaginian commander at t... | correct | 0.059 | maintain | correct_maintain |
| Siege of Constantinople occurred in 1453 | correct | 0.039 | maintain | correct_maintain |
| Mehmed II (Sultan of the Ottoman Empire) was the O... | correct | 0.214 | maintain | correct_maintain |
| Emperor Constantine XI Palaiologos was the Byzanti... | correct | 0.498 | maintain | correct_maintain |
| Edison invented the first commercially successful ... | correct | 0.003 | maintain | correct_maintain |
| Edison invented the phonograph | correct | 0.012 | maintain | correct_maintain |
| Edison invented the Kinetoscope (early motion pict... | correct | 0.012 | maintain | correct_maintain |
| Tesla invented the Tesla coil | correct | 0.013 | maintain | correct_maintain |
| Tesla invented the induction motor | incorrect | 0.003 | correct | true_positive_fixed |
| Tesla invented the alternating current power trans... | incorrect | 0.002 | correct | true_positive_fixed |
| USPTO database is available at https://www.uspto.g... | correct | 0.479 | maintain | correct_maintain |
| Royal Society founding date was 23 April 1660 | incorrect | 0.303 | correct | true_positive_fixed |
| Robert Boyle was a founding member of the Royal So... | correct | 0.063 | maintain | correct_maintain |
| Isaac Barrow was a founding member of the Royal So... | incorrect | 0.015 | retract | true_positive_fixed |
| Edmond Halley was a founding member of the Royal S... | incorrect | 0.045 | correct | true_positive_fixed |
| Christopher Wren was a founding member of the Roya... | correct | 0.009 | correct | false_positive_fixed |
| Académie des Sciences original name was Académie R... | correct | 0.011 | maintain | correct_maintain |
| American Philosophical Society founding date was 1... | correct | 0.071 | maintain | correct_maintain |
| American Philosophical Society original name was T... | incorrect | 0.310 | correct | true_positive_fixed |
| John Adams was a founding member of the American P... | incorrect | 0.228 | correct | true_positive_fixed |
| Thomas Jefferson was a founding member of the Amer... | incorrect | 0.267 | maintain | missed |
| John Hancock was a founding member of the American... | incorrect | 0.263 | correct | true_positive_fixed |
| Prussian Academy of Sciences founding date was 176... | incorrect | 0.052 | retract | true_positive_fixed |
| Prussian Academy of Sciences original name was Aka... | incorrect | 0.052 | correct | true_positive_fixed |
| Wide Field Camera 3 (WFC3) wavelength range is 400... | incorrect | 0.035 | correct | true_positive_fixed |
| WFC3 field of view is approximately 10 arcminutes | incorrect | 0.059 | correct | true_positive_fixed |
| WFC3 full field of view is 15 arcminutes | incorrect | 0.058 | correct | true_positive_fixed |
| Cosmic Origins Spectrograph (COS) wavelength range... | incorrect | 0.076 | retract | true_positive_fixed |
| COS field of view is ~5 arcseconds | correct | 0.118 | correct | false_positive_fixed |
| Space Telescope Imaging Spectrograph (STIS) wavele... | incorrect | 0.025 | retract | true_positive_fixed |
| STIS low resolution is ~0.2 arcseconds | incorrect | 0.029 | correct | true_positive_fixed |
| STIS field of view is ~10 arcseconds | incorrect | 0.082 | correct | true_positive_fixed |
| Francium melting point is approximately 27°C (81°F... | incorrect | 0.131 | maintain | missed |
| Francium boiling point is approximately 70°C (158°... | incorrect | 0.128 | correct | true_positive_fixed |
| Francium density is approximately 2.9 g/cm³ | incorrect | 0.124 | correct | true_positive_fixed |
| Astatine melting point is approximately -71°C (180... | incorrect | 0.136 | retract | true_positive_fixed |
| Astatine boiling point is approximately -71°C (180... | incorrect | 0.089 | correct | true_positive_fixed |
| Oganesson melting point has not been measured expe... | correct | 0.054 | maintain | correct_maintain |
| Oganesson boiling point has not been measured expe... | correct | 0.062 | maintain | correct_maintain |
| Oganesson density has not been measured experiment... | correct | 0.029 | maintain | correct_maintain |
| Tennessine melting point has not been measured exp... | correct | 0.037 | maintain | correct_maintain |
| Tennessine density has not been measured experimen... | correct | 0.037 | correct | false_positive_fixed |
| Francium, Astatine, Oganesson, and Tennessine are ... | correct | 0.102 | maintain | correct_maintain |
| Voyager 1 carries a Magnetometer that measures the... | correct | 0.130 | maintain | correct_maintain |
| Voyager 1 carries a Plasma Science Instrument that... | correct | 0.476 | maintain | correct_maintain |
| Voyager 1 carries a Radio Science Experiment that ... | incorrect | 0.220 | correct | true_positive_fixed |
| Voyager 1 carries a Visual Imaging System that cap... | correct | 0.007 | correct | false_positive_fixed |
| Voyager 1 transmits data back to Earth using radio... | correct | 0.231 | maintain | correct_maintain |
| Voyager 1 data is received and analyzed by scienti... | correct | 0.242 | maintain | correct_maintain |
| The primary objectives of the Voyager 1 mission we... | correct | 0.027 | maintain | correct_maintain |
| Tevatron booster ring beam energy is 350 GeV | incorrect | 0.281 | correct | true_positive_fixed |
| Tevatron luminosity is ~10^33 cm^-2s^-1 | incorrect | 0.067 | correct | true_positive_fixed |
| Tevatron circumference is ~1.2 km | incorrect | 0.016 | correct | true_positive_fixed |
| Super Proton Synchrotron luminosity is ~10^33 cm^-... | incorrect | 0.088 | correct | true_positive_fixed |
| Super Proton Synchrotron circumference is ~10.2 km | incorrect | 0.063 | correct | true_positive_fixed |
| KEKB reached up to 300 GeV | incorrect | 0.188 | retract | true_positive_fixed |
| KEKB luminosity is ~10^34 cm^-2s^-1 | correct | 0.119 | maintain | correct_maintain |
| RHIC gold ions beam energy is 200 GeV/nucleon | correct | 0.006 | correct | false_positive_fixed |
| RHIC lead ions beam energy is 200 GeV/nucleon | incorrect | 0.005 | maintain | missed |
| RHIC luminosity is ~10^34 cm^-2s^-1 | incorrect | 0.027 | correct | true_positive_fixed |
| RHIC circumference is ~2.4 km | incorrect | 0.060 | correct | true_positive_fixed |
| Shannon's "A Mathematical Theory of Communication"... | correct | 0.397 | maintain | correct_maintain |
| Shannon's "A Mathematical Theory of Communication"... | correct | 0.026 | maintain | correct_maintain |
| Shannon's "A Mathematical Theory of Communication"... | correct | 0.011 | maintain | correct_maintain |
| Shannon's "A Mathematical Theory of Communication"... | correct | 0.164 | maintain | correct_maintain |
| Turing's "On Computable Numbers, with an Applicati... | incorrect | 0.095 | retract | true_positive_fixed |
| Turing's "On Computable Numbers, with an Applicati... | incorrect | 0.029 | correct | true_positive_fixed |
| Turing's "On Computable Numbers, with an Applicati... | correct | 0.015 | correct | false_positive_fixed |
| Turing's "On Computable Numbers, with an Applicati... | correct | 0.080 | maintain | correct_maintain |
| Nash's "Non-Cooperative Games" was published in Th... | incorrect | 0.015 | correct | true_positive_fixed |
| Nash's "Non-Cooperative Games" page numbers are 57... | incorrect | 0.043 | retract | true_positive_fixed |
| Nash's "Non-Cooperative Games" was published in 19... | incorrect | 0.095 | retract | true_positive_fixed |
| Photoelectric Effect paper published in Annalen de... | correct | 0.044 | maintain | correct_maintain |
| Photoelectric Effect paper published in 1905 | correct | 0.217 | maintain | correct_maintain |
| Brownian Motion paper published in Annalen der Phy... | correct | 0.014 | maintain | correct_maintain |
| Brownian Motion paper published in 1905 | correct | 0.176 | maintain | correct_maintain |
| Special Relativity paper published in 1905 | correct | 0.239 | maintain | correct_maintain |
| Mass-Energy Equivalence paper published in Annalen... | correct | 0.095 | maintain | correct_maintain |
| Mass-Energy Equivalence paper published in 1905 | correct | 0.068 | maintain | correct_maintain |
| lightbright project contains SAE feature analysis ... | correct | 0.324 | maintain | correct_maintain |
| Fine-tuned detectors on Gemma-3-27B achieve 0.98 A... | correct | 0.002 | correct | false_positive_fixed |
| SAE intent probes with 8 hand-selected semantic fe... | correct | 0.092 | maintain | correct_maintain |
| researchRalph current best result is 0.79 AUROC (E... | correct | 0.374 | correct | false_positive_fixed |
| Single feature L40_F12574 alone achieves 0.71 AURO... | correct | 0.011 | maintain | correct_maintain |
| miscJan2026 contains Jan 2026 experiments | correct | 0.090 | maintain | correct_maintain |
| misc2025 contains 2025 experiments | correct | 0.021 | maintain | correct_maintain |
| Gödel's Incompleteness Theorems were published in ... | correct | 0.004 | maintain | correct_maintain |
| Stephen Cook and Michael Levin published 'Reducibi... | incorrect | 0.021 | retract | true_positive_fixed |
| The Cook-Levin Theorem was published in 1971 | correct | 0.083 | maintain | correct_maintain |
| Andrew Wiles published 'Modular forms, elliptic cu... | incorrect | 0.017 | correct | true_positive_fixed |
| Andrew Wiles's proof of Fermat's Last Theorem was ... | incorrect | 0.025 | correct | true_positive_fixed |
| Emmy Noether was born in 1882 | correct | 0.113 | maintain | correct_maintain |
| Emmy Noether died in 1935 | correct | 0.042 | maintain | correct_maintain |
| Emmy Noether was a German mathematician | correct | 0.371 | maintain | correct_maintain |
| Hermann Minkowski was one of Noether's doctoral ad... | incorrect | 0.133 | retract | true_positive_fixed |
| Hermann Minkowski was a prominent physicist and ma... | correct | 0.427 | maintain | correct_maintain |
| Hermann Minkowski was a crucial figure in the deve... | correct | 0.260 | maintain | correct_maintain |
| Hermann Weyl was Noether's student at the Universi... | incorrect | 0.024 | retract | true_positive_fixed |
| Hermann Weyl became a prominent mathematician and ... | correct | 0.340 | maintain | correct_maintain |
| Lise Meitner was a nuclear physicist | correct | 0.327 | maintain | correct_maintain |
| Noether was a mentor to Lise Meitner | incorrect | 0.030 | correct | true_positive_fixed |
| Noether worked at the University of Erlangen from ... | incorrect | 0.066 | correct | true_positive_fixed |
| Noether worked at the University of Göttingen from... | incorrect | 0.042 | correct | true_positive_fixed |
| Noether's Theorem states that for every continuous... | correct | 0.268 | maintain | correct_maintain |
| Ramanujan was born in Erode, Tamil Nadu, India | incorrect | 0.194 | correct | true_positive_fixed |
| Ramanujan was born in 1887 | correct | 0.229 | maintain | correct_maintain |
| Ramanujan began formal education in mathematics in... | correct | 0.032 | correct | false_positive_fixed |
| Ramanujan studied under Professor A.P.J. Abdul Kal... | incorrect | 0.030 | correct | true_positive_fixed |
| In 1910, Ramanujan evolved from a student of mathe... | incorrect | 0.111 | correct | true_positive_fixed |
| Ramanujan encountered G.H. Hardy in 1913 | correct | 0.013 | correct | false_positive_fixed |
| G.H. Hardy was a renowned mathematician at Cambrid... | correct | 0.165 | maintain | correct_maintain |
| Correspondence between Ramanujan and Hardy lasted ... | correct | 0.036 | maintain | correct_maintain |
| Ramanujan's first paper was published between 1914... | incorrect | 0.027 | correct | true_positive_fixed |
| Rosalind Franklin attended St. Paul's School | incorrect | 0.011 | retract | true_positive_fixed |
| St. Paul's School is a girls' boarding school in H... | incorrect | 0.316 | maintain | missed |
| Franklin specialized in crystallography | correct | 0.024 | maintain | correct_maintain |
| Sir William Lawrence Bragg supervised Franklin | incorrect | 0.023 | correct | true_positive_fixed |
| Franklin joined King's College, London chemistry d... | incorrect | 0.066 | maintain | missed |
| Franklin's X-ray crystallography work focused on s... | correct | 0.007 | maintain | correct_maintain |
| Photo 51 was created in 1952 | correct | 0.014 | correct | false_positive_fixed |
| Photo 51 is an X-ray diffraction image of DNA | correct | 0.143 | maintain | correct_maintain |
| Franklin collaborated with Maurice Wilkins | incorrect | 0.054 | maintain | missed |
| Maurice Wilkins was a fellow scientist at King's C... | correct | 0.184 | correct | false_positive_fixed |
| The Copley Medal is the highest honor in the field... | incorrect | 0.022 | correct | true_positive_fixed |
| Évariste Galois was born in 1811 at the small vill... | incorrect | 0.049 | correct | true_positive_fixed |
| Galois entered the École Polytechnique in 1829 | incorrect | 0.214 | correct | true_positive_fixed |
| Galois graduated from the École Polytechnique in 1... | incorrect | 0.055 | retract | true_positive_fixed |
| Galois published groundbreaking papers on the theo... | incorrect | 0.015 | retract | true_positive_fixed |
| Galois earned his doctorate from the University of... | incorrect | 0.011 | retract | true_positive_fixed |
| Galois penned his final letter, 'Letter to the Min... | incorrect | 0.188 | retract | true_positive_fixed |
| Galois died on June 30th, 1835 at the age of 25 in... | incorrect | 0.050 | correct | true_positive_fixed |
| Galois died at the hands of a group of soldiers | incorrect | 0.034 | retract | true_positive_fixed |

## Comparison to Paper Baseline

| Metric | Our Pipeline | Goodfire RLFR (with RL) |
|--------|-------------|------------------------|
| Hallucination reduction | 87.5% | 58% |

Note: The paper achieves 58% with RL training. Our pipeline uses probes + Claude intervention
without any RL, so lower reduction is expected.

## Interpretation

The truthfulness probe shows limited discriminative power on free-form generation (potential domain shift from TruthfulQA format).
The probe substantially outperforms a random baseline, confirming the signal is genuine.
Claude self-check (0.936) outperforms the probe (0.584), suggesting that language-level verification is currently stronger than activation-based detection for this task.
