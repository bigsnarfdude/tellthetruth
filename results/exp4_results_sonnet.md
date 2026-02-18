# Experiment 4: End-to-End Pipeline Validation

*Generated: 2026-02-17 21:56:15*
*Runtime: 3987s*

## Setup

- **Model:** google/gemma-2-2b-it (26 layers)
- **Prompts:** 20 hard factual prompts across 5 categories
- **Truthfulness probe:** layer 16, last-token, C=0.01 (AUROC=0.8768)
- **Verification:** Claude-as-judge (ground truth labels)
- **Protocol:** Dev/test split for probe; claims scored independently

## 4.1 Generation Summary

- **Paragraphs generated:** 20
- **Total claims extracted:** 271
- **Correct:** 106 (39.1%)
- **Incorrect:** 147 (54.2%)
- **Unverifiable:** 18 (6.6%)

### By Category

| Category | Claims | Correct | Incorrect | Unverifiable | Hallucination Rate |
|----------|--------|---------|-----------|-------------|-------------------|
| biographical | 59 | 17 | 40 | 2 | 70.2% |
| historical_dates | 56 | 29 | 25 | 2 | 46.3% |
| obscure_scientist | 44 | 12 | 22 | 10 | 64.7% |
| paper_citations | 50 | 23 | 24 | 3 | 51.1% |
| technical_specs | 62 | 25 | 36 | 1 | 59.0% |

## 4.3 Method Comparison (verifiable claims only)

| Method | AUROC | Accuracy | Precision | Recall | F1 |
|--------|-------|----------|-----------|--------|-----|
| truthfulness_probe | 0.5918 | 0.6008 | 0.7273 | 0.0755 | 0.1368 |
| claude_self_check | 0.9857 | 0.8893 | 0.7910 | 1.0000 | 0.8833 |
| random_probe | 0.5086 | 0.5375 | 0.4567 | 0.5472 | 0.4979 |

## 4.4 Intervention Results

| Metric | Value |
|--------|-------|
| Claims flagged by probe | 242 |
| True positives fixed | 135 |
| False positives (wrong retraction) | 6 |
| Missed (incorrect, not flagged or maintained) | 9 |
| Correct maintains | 92 |
| Total incorrect claims | 147 |
| Total correct claims | 106 |
| **Hallucination reduction rate** | **91.8%** |

### Intervention Details

| Claim | Ground Truth | P(correct) | Action | Grade |
|-------|-------------|------------|--------|-------|
| Marian Smoluchowski was born in 1872 | correct | 0.116 | correct | false_positive_fixed |
| Marian Smoluchowski died in 1919 | incorrect | 0.020 | correct | true_positive_fixed |
| Marian Smoluchowski was Polish | correct | 0.105 | maintain | correct_maintain |
| Smoluchowski published a paper on Brownian motion ... | incorrect | 0.032 | correct | true_positive_fixed |
| The paper was titled 'On the Brownian Motion of Pa... | incorrect | 0.050 | correct | true_positive_fixed |
| Smoluchowski published a paper on coagulation in 1... | incorrect | 0.073 | correct | true_positive_fixed |
| Paul Ehrenfest was born in 1887 | incorrect | 0.187 | correct | true_positive_fixed |
| Paul Ehrenfest died in 1933 | correct | 0.010 | maintain | correct_maintain |
| Paul Ehrenfest was born in St. Petersburg, Russia | incorrect | 0.055 | correct | true_positive_fixed |
| Paul Ehrenfest received his education at the Unive... | incorrect | 0.011 | correct | true_positive_fixed |
| Paul Ehrenfest worked as a research assistant at t... | incorrect | 0.109 | correct | true_positive_fixed |
| Paul Ehrenfest moved to the University of Berlin f... | incorrect | 0.032 | correct | true_positive_fixed |
| Paul Ehrenfest held a professorship at the Univers... | incorrect | 0.133 | correct | true_positive_fixed |
| Paul Ehrenfest became a professor at the Universit... | incorrect | 0.029 | correct | true_positive_fixed |
| Paul Ehrenfest returned to the University of Götti... | incorrect | 0.082 | correct | true_positive_fixed |
| Paul Ehrenfest's father was a prominent mathematic... | incorrect | 0.014 | correct | true_positive_fixed |
| Ehrenfest taught at the University of Berlin from ... | incorrect | 0.030 | correct | true_positive_fixed |
| Ehrenfest taught at the University of Zurich, Swit... | incorrect | 0.079 | correct | true_positive_fixed |
| Ehrenfest taught at the University of Göttingen fr... | incorrect | 0.119 | correct | true_positive_fixed |
| Lise Meitner was born in 1878 | correct | 0.014 | maintain | correct_maintain |
| Lise Meitner died in 1968 | correct | 0.023 | maintain | correct_maintain |
| Lise Meitner was Austrian-German | correct | 0.011 | maintain | correct_maintain |
| Meitner studied under Max Planck in Vienna | incorrect | 0.001 | correct | true_positive_fixed |
| Meitner worked under Otto Hahn in Berlin | incorrect | 0.002 | correct | true_positive_fixed |
| In 1938, Meitner and Otto Frisch calculated the en... | incorrect | 0.232 | correct | true_positive_fixed |
| The Meitner-Frisch paper was published in the jour... | correct | 0.003 | maintain | correct_maintain |
| The Meitner-Frisch paper was published on 28th Jul... | incorrect | 0.145 | correct | true_positive_fixed |
| The first successful nuclear chain reaction occurr... | correct | 0.049 | maintain | correct_maintain |
| The Nobel Prize for nuclear fission was awarded to... | incorrect | 0.100 | correct | true_positive_fixed |
| Lise Meitner was never nominated for the Nobel Pri... | incorrect | 0.061 | correct | true_positive_fixed |
| Otto Frisch was a collaborator and friend of Meitn... | correct | 0.111 | maintain | correct_maintain |
| Subrahmanyan Chandrasekhar was an Indian-American ... | correct | 0.305 | maintain | correct_maintain |
| Chandrasekhar won the Nobel Prize in Physics in 19... | correct | 0.027 | maintain | correct_maintain |
| The Chandrasekhar Limit is 1.44 times the mass of ... | correct | 0.023 | maintain | correct_maintain |
| Treaty of Westphalia was signed between 1648 and 1... | correct | 0.100 | correct | false_positive_fixed |
| Main signatories of the Treaty of Westphalia inclu... | incorrect | 0.048 | correct | true_positive_fixed |
| Treaty of Tordesillas was signed in 1494 | correct | 0.242 | maintain | correct_maintain |
| Treaty of Tordesillas was signed by King Ferdinand... | incorrect | 0.006 | correct | true_positive_fixed |
| Treaty of Tordesillas was signed by Queen Isabella... | correct | 0.019 | correct | false_positive_fixed |
| Treaty of Tordesillas divided the New World betwee... | correct | 0.064 | maintain | correct_maintain |
| Treaty of Nerchinsk was signed in 1850 | incorrect | 0.032 | correct | true_positive_fixed |
| Treaty of Nerchinsk was signed by the Russian Empi... | incorrect | 0.093 | correct | true_positive_fixed |
| Treaty of Nerchinsk was primarily focused on borde... | correct | 0.326 | maintain | correct_maintain |
| Peace of Augsburg was signed in 1555 | correct | 0.097 | maintain | correct_maintain |
| Peace of Augsburg was signed by Holy Roman Emperor... | incorrect | 0.129 | correct | true_positive_fixed |
| Peace of Augsburg established the principle of cui... | correct | 0.003 | maintain | correct_maintain |
| Marshal Soult commanded the French forces at the B... | incorrect | 0.053 | correct | true_positive_fixed |
| General Davout was in charge of the Russian forces... | incorrect | 0.016 | correct | true_positive_fixed |
| Battle of Tannenberg had an estimated 20,000 to 30... | incorrect | 0.369 | correct | true_positive_fixed |
| Battle of Tannenberg took place on August 26, 1914 | incorrect | 0.143 | correct | true_positive_fixed |
| General Paul von Hindenburg commanded the German f... | correct | 0.279 | maintain | correct_maintain |
| General P.V. Krasnov commanded the Russian forces ... | incorrect | 0.204 | correct | true_positive_fixed |
| Battle of Zama took place in 202 BC | correct | 0.018 | maintain | correct_maintain |
| Scipio Africanus commanded the Roman forces at the... | correct | 0.295 | maintain | correct_maintain |
| The Siege of Constantinople occurred in 1453 | correct | 0.058 | maintain | correct_maintain |
| Emperor Constantine XI Palaiologos commanded the B... | correct | 0.317 | maintain | correct_maintain |
| The Royal Society was founded on 23 April 1660 | incorrect | 0.194 | correct | true_positive_fixed |
| The Royal Society's original name was 'The Royal S... | incorrect | 0.344 | maintain | missed |
| Robert Boyle was a founding member of the Royal So... | correct | 0.063 | maintain | correct_maintain |
| Robert Boyle was a physicist and chemist | correct | 0.169 | maintain | correct_maintain |
| Isaac Barrow was a founding member of the Royal So... | incorrect | 0.015 | correct | true_positive_fixed |
| Isaac Barrow was a mathematician and physicist | incorrect | 0.057 | correct | true_positive_fixed |
| Edmond Halley was a founding member of the Royal S... | incorrect | 0.045 | correct | true_positive_fixed |
| Edmond Halley was an astronomer | correct | 0.099 | maintain | correct_maintain |
| Henry Oldenburg was a founding member of the Royal... | correct | 0.123 | maintain | correct_maintain |
| Christopher Wren was a founding member of the Roya... | correct | 0.009 | maintain | correct_maintain |
| Christopher Wren was an architect | correct | 0.334 | maintain | correct_maintain |
| The Académie des Sciences was founded on 20 August... | incorrect | 0.128 | correct | true_positive_fixed |
| The Académie des Sciences' original name was 'Acad... | correct | 0.144 | maintain | correct_maintain |
| The American Philosophical Society was founded in ... | correct | 0.077 | maintain | correct_maintain |
| John Adams was a founding member of the American P... | incorrect | 0.228 | correct | true_positive_fixed |
| John Adams became president of the United States | correct | 0.406 | maintain | correct_maintain |
| Thomas Jefferson was a founding member of the Amer... | incorrect | 0.267 | correct | true_positive_fixed |
| Thomas Jefferson became president of the United St... | correct | 0.416 | maintain | correct_maintain |
| William Smith was a founding member of the America... | incorrect | 0.044 | correct | true_positive_fixed |
| William Smith was a surveyor and cartographer | correct | 0.418 | maintain | correct_maintain |
| John Hancock was a founding member of the American... | incorrect | 0.263 | correct | true_positive_fixed |
| John Hancock was president of the Continental Cong... | correct | 0.046 | maintain | correct_maintain |
| The Prussian Academy of Sciences was officially es... | incorrect | 0.181 | correct | true_positive_fixed |
| The Prussian Academy of Sciences' original name wa... | incorrect | 0.434 | correct | true_positive_fixed |
| The Prussian Academy of Sciences' first members we... | incorrect | 0.078 | correct | true_positive_fixed |
| Frederick the Great was the King of Prussia | correct | 0.492 | maintain | correct_maintain |
| WFC3 wavelength range is 400-700 nm | incorrect | 0.299 | correct | true_positive_fixed |
| WFC3 resolution is 0.1 arcseconds | incorrect | 0.014 | correct | true_positive_fixed |
| WFC3 field of view is approximately 10 arcminutes | incorrect | 0.059 | correct | true_positive_fixed |
| WFC3 full field of view is 15 arcminutes | incorrect | 0.058 | correct | true_positive_fixed |
| WFC3 spectral resolution is approximately 30% per ... | incorrect | 0.012 | correct | true_positive_fixed |
| COS wavelength range is 0.3-1.7 microns | incorrect | 0.034 | correct | true_positive_fixed |
| COS high resolution is approximately 0.01 arcsecon... | incorrect | 0.032 | maintain | missed |
| COS field of view is approximately 5 arcseconds | incorrect | 0.134 | correct | true_positive_fixed |
| COS spectral resolution is approximately 20% per p... | incorrect | 0.024 | correct | true_positive_fixed |
| STIS wavelength range is 1100-1700 nm | incorrect | 0.161 | correct | true_positive_fixed |
| STIS high resolution is approximately 0.05 arcseco... | correct | 0.096 | maintain | correct_maintain |
| STIS low resolution is approximately 0.2 arcsecond... | incorrect | 0.073 | correct | true_positive_fixed |
| STIS field of view is approximately 10 arcseconds | incorrect | 0.159 | correct | true_positive_fixed |
| STIS spectral resolution high is approximately 10%... | incorrect | 0.083 | correct | true_positive_fixed |
| Francium has a melting point of approximately 27°C... | correct | 0.230 | maintain | correct_maintain |
| Francium has a boiling point of approximately 70°C... | incorrect | 0.197 | correct | true_positive_fixed |
| Francium has a density of approximately 2.9 g/cm³ | incorrect | 0.203 | correct | true_positive_fixed |
| Francium has an electronegativity estimated around... | correct | 0.271 | maintain | correct_maintain |
| Astatine has a melting point of approximately -71°... | incorrect | 0.179 | correct | true_positive_fixed |
| Astatine has a boiling point of approximately -71°... | incorrect | 0.161 | correct | true_positive_fixed |
| Astatine has a density of approximately 1.4 g/cm³ | incorrect | 0.142 | correct | true_positive_fixed |
| Astatine has an electronegativity estimated around... | incorrect | 0.240 | correct | true_positive_fixed |
| Oganesson's melting point has not been measured ex... | correct | 0.026 | maintain | correct_maintain |
| Oganesson's boiling point has not been measured ex... | correct | 0.040 | maintain | correct_maintain |
| Oganesson's density has not been measured experime... | correct | 0.022 | maintain | correct_maintain |
| Oganesson has an electronegativity estimated aroun... | incorrect | 0.368 | maintain | missed |
| Tennessine's melting point has not been measured e... | correct | 0.022 | maintain | correct_maintain |
| Tennessine's boiling point has not been measured e... | correct | 0.031 | maintain | correct_maintain |
| Tennessine's density has not been measured experim... | correct | 0.038 | maintain | correct_maintain |
| Tennessine has an electronegativity estimated arou... | incorrect | 0.285 | correct | true_positive_fixed |
| Francium is highly radioactive | correct | 0.103 | maintain | correct_maintain |
| Astatine is highly radioactive | correct | 0.113 | maintain | correct_maintain |
| Oganesson is highly radioactive | correct | 0.023 | maintain | correct_maintain |
| Tennessine is highly radioactive | correct | 0.030 | maintain | correct_maintain |
| Voyager 1 carries a Magnetometer instrument | correct | 0.243 | maintain | correct_maintain |
| Voyager 1 carries a Plasma Science Instrument | correct | 0.180 | maintain | correct_maintain |
| Voyager 1 carries a Cosmic Ray Detector | correct | 0.245 | maintain | correct_maintain |
| Voyager 1 carries a Radio Science Experiment | correct | 0.250 | maintain | correct_maintain |
| Voyager 1 carries a Visual Imaging System | incorrect | 0.065 | correct | true_positive_fixed |
| Voyager 1 transmits data back to Earth using radio... | correct | 0.231 | maintain | correct_maintain |
| Voyager 1 data is received and analyzed by scienti... | correct | 0.242 | maintain | correct_maintain |
| The RTG converts heat from radioactive decay into ... | correct | 0.251 | maintain | correct_maintain |
| The primary objectives of the Voyager 1 mission we... | incorrect | 0.040 | maintain | missed |
| The Voyager 1 mission has been ongoing for decades | correct | 0.099 | maintain | correct_maintain |
| Tevatron main ring beam energy is 1.02 TeV | incorrect | 0.346 | correct | true_positive_fixed |
| Tevatron booster ring beam energy is 350 GeV | incorrect | 0.281 | correct | true_positive_fixed |
| Tevatron luminosity is approximately 10^33 cm^-2s^... | correct | 0.093 | correct | false_positive_fixed |
| Tevatron circumference is approximately 1.2 km | incorrect | 0.020 | correct | true_positive_fixed |
| Super Proton Synchrotron (SPS) main ring beam ener... | incorrect | 0.091 | maintain | missed |
| SPS luminosity is approximately 10^33 cm^-2s^-1 | incorrect | 0.038 | correct | true_positive_fixed |
| SPS circumference is approximately 10.2 km | incorrect | 0.052 | correct | true_positive_fixed |
| KEKB reached up to 300 GeV beam energy | incorrect | 0.118 | retract | true_positive_fixed |
| KEKB luminosity is approximately 10^34 cm^-2s^-1 | correct | 0.109 | maintain | correct_maintain |
| KEKB circumference is approximately 26 km | incorrect | 0.065 | correct | true_positive_fixed |
| RHIC gold ion beam energy is 200 GeV/nucleon | incorrect | 0.011 | correct | true_positive_fixed |
| RHIC lead ion beam energy is 200 GeV/nucleon | incorrect | 0.010 | correct | true_positive_fixed |
| RHIC luminosity is approximately 10^34 cm^-2s^-1 | incorrect | 0.039 | correct | true_positive_fixed |
| RHIC circumference is approximately 2.4 km | incorrect | 0.075 | maintain | missed |
| Shannon's 'A Mathematical Theory of Communication'... | correct | 0.302 | maintain | correct_maintain |
| Shannon's 'A Mathematical Theory of Communication'... | correct | 0.027 | maintain | correct_maintain |
| Shannon's 'A Mathematical Theory of Communication'... | incorrect | 0.059 | correct | true_positive_fixed |
| Shannon's 'A Mathematical Theory of Communication'... | correct | 0.153 | maintain | correct_maintain |
| Turing's 'On Computable Numbers, with an Applicati... | incorrect | 0.106 | correct | true_positive_fixed |
| Turing's 'On Computable Numbers, with an Applicati... | incorrect | 0.099 | maintain | missed |
| Turing's 'On Computable Numbers, with an Applicati... | correct | 0.062 | correct | false_positive_fixed |
| Turing's 'On Computable Numbers, with an Applicati... | correct | 0.091 | maintain | correct_maintain |
| Nash's 'Non-Cooperative Games' was published in Th... | incorrect | 0.022 | correct | true_positive_fixed |
| Nash's 'Non-Cooperative Games' was published in vo... | incorrect | 0.164 | correct | true_positive_fixed |
| Nash's 'Non-Cooperative Games' spans pages 577-593 | incorrect | 0.086 | correct | true_positive_fixed |
| Nash's 'Non-Cooperative Games' was published in 19... | incorrect | 0.095 | correct | true_positive_fixed |
| Einstein published a paper on the Photoelectric Ef... | correct | 0.234 | maintain | correct_maintain |
| The Photoelectric Effect paper was published in An... | correct | 0.032 | maintain | correct_maintain |
| The Photoelectric Effect paper appeared in Volume ... | incorrect | 0.064 | correct | true_positive_fixed |
| Einstein published a paper on Brownian Motion in 1... | correct | 0.059 | maintain | correct_maintain |
| The Brownian Motion paper was published in Annalen... | correct | 0.009 | maintain | correct_maintain |
| Einstein published a paper on Special Relativity i... | correct | 0.050 | maintain | correct_maintain |
| The Special Relativity paper was published in Anna... | correct | 0.074 | maintain | correct_maintain |
| The Special Relativity paper appeared in Volume 32... | incorrect | 0.057 | correct | true_positive_fixed |
| The Special Relativity paper appeared on pages 639... | incorrect | 0.043 | correct | true_positive_fixed |
| Einstein published a paper on Mass-Energy Equivale... | correct | 0.004 | maintain | correct_maintain |
| The Mass-Energy Equivalence paper was published in... | correct | 0.013 | maintain | correct_maintain |
| The Mass-Energy Equivalence paper appeared in Volu... | incorrect | 0.037 | correct | true_positive_fixed |
| The Mass-Energy Equivalence paper appeared on page... | incorrect | 0.015 | correct | true_positive_fixed |
| Watson and Crick's paper is titled 'Molecular Stru... | correct | 0.158 | maintain | correct_maintain |
| Watson and Crick's paper was authored by James Wat... | correct | 0.489 | maintain | correct_maintain |
| Watson and Crick's paper was published in Nature | correct | 0.009 | maintain | correct_maintain |
| Watson and Crick's paper was published on 28th Apr... | incorrect | 0.153 | correct | true_positive_fixed |
| Watson and Crick's paper appears on pages 715-730 ... | incorrect | 0.030 | correct | true_positive_fixed |
| Rosalind Franklin's paper is titled 'The Constitut... | incorrect | 0.213 | correct | true_positive_fixed |
| Rosalind Franklin's paper was published in Nature | correct | 0.003 | maintain | correct_maintain |
| Rosalind Franklin's paper was published in 1952 | incorrect | 0.019 | correct | true_positive_fixed |
| The Meselson-Stahl experiment paper is titled 'DNA... | incorrect | 0.085 | correct | true_positive_fixed |
| Matthew Meselson is an author of the Meselson-Stah... | correct | 0.148 | maintain | correct_maintain |
| Gödel's paper was published in 1931 | correct | 0.072 | maintain | correct_maintain |
| Gödel's paper title is 'On the consistency of the ... | incorrect | 0.363 | correct | true_positive_fixed |
| Gödel's paper was published in the Journal für die... | incorrect | 0.022 | correct | true_positive_fixed |
| The Cook-Levin theorem paper was published in 1971 | correct | 0.149 | correct | false_positive_fixed |
| The authors of the Cook-Levin theorem paper are St... | incorrect | 0.057 | correct | true_positive_fixed |
| The Cook-Levin theorem paper title is 'Reducibilit... | incorrect | 0.053 | correct | true_positive_fixed |
| The Cook-Levin theorem paper was published in The ... | incorrect | 0.032 | correct | true_positive_fixed |
| Andrew Wiles's proof of Fermat's Last Theorem was ... | correct | 0.025 | maintain | correct_maintain |
| The author of the Fermat's Last Theorem proof is A... | correct | 0.318 | maintain | correct_maintain |
| Wiles's paper title is 'Modular forms, elliptic cu... | incorrect | 0.108 | correct | true_positive_fixed |
| Wiles's paper was published in The Journal of the ... | incorrect | 0.135 | correct | true_positive_fixed |
| Emmy Noether was born in 1882 | correct | 0.113 | maintain | correct_maintain |
| Emmy Noether died in 1935 | correct | 0.042 | maintain | correct_maintain |
| Emmy Noether was German | correct | 0.056 | maintain | correct_maintain |
| Emmy Noether studied mathematics at the University... | correct | 0.055 | maintain | correct_maintain |
| Emmy Noether earned a diploma in mathematics from ... | incorrect | 0.017 | correct | true_positive_fixed |
| Hermann Minkowski was one of Noether's doctoral ad... | incorrect | 0.134 | correct | true_positive_fixed |
| Hermann Weyl was a student of Noether's at the Uni... | incorrect | 0.046 | correct | true_positive_fixed |
| Lise Meitner was mentored by Noether | incorrect | 0.000 | correct | true_positive_fixed |
| Lise Meitner was a nuclear physicist | correct | 0.327 | maintain | correct_maintain |
| Noether was at the University of Göttingen from 19... | incorrect | 0.073 | correct | true_positive_fixed |
| Noether was at the University of Berlin from 1920 ... | incorrect | 0.037 | correct | true_positive_fixed |
| Noether's Theorem states that for every continuous... | correct | 0.236 | maintain | correct_maintain |
| Hermann Minkowski was a key figure in the developm... | correct | 0.281 | maintain | correct_maintain |
| Ramanujan was born in 1887 | correct | 0.229 | maintain | correct_maintain |
| Ramanujan was born in Erode, Tamil Nadu, India | correct | 0.194 | maintain | correct_maintain |
| Ramanujan begins formal education in mathematics i... | correct | 0.024 | maintain | correct_maintain |
| Ramanujan studied under Professor A.P.J. Abdul Kal... | incorrect | 0.030 | correct | true_positive_fixed |
| Ramanujan studied at the Madras Presidency College | incorrect | 0.014 | correct | true_positive_fixed |
| Ramanujan encountered G.H. Hardy in 1913 | incorrect | 0.013 | maintain | missed |
| G.H. Hardy was a mathematician at Cambridge Univer... | correct | 0.267 | maintain | correct_maintain |
| Ramanujan's first paper was titled 'On the Theory ... | incorrect | 0.161 | correct | true_positive_fixed |
| Ramanujan's first paper was published between 1914... | incorrect | 0.051 | correct | true_positive_fixed |
| Ramanujan's Formula for the Partition Function rec... | incorrect | 0.014 | correct | true_positive_fixed |
| Ramanujan's Formula relates the number of partitio... | incorrect | 0.003 | correct | true_positive_fixed |
| Rosalind Franklin attended St. Paul's School from ... | incorrect | 0.011 | correct | true_positive_fixed |
| St. Paul's School is a girls' boarding school in H... | incorrect | 0.316 | correct | true_positive_fixed |
| Rosalind Franklin studied at King's College, Londo... | incorrect | 0.028 | correct | true_positive_fixed |
| Rosalind Franklin graduated with a Bachelor of Sci... | incorrect | 0.004 | correct | true_positive_fixed |
| Rosalind Franklin earned a Master of Science degre... | incorrect | 0.011 | correct | true_positive_fixed |
| Rosalind Franklin specialized in crystallography a... | incorrect | 0.227 | correct | true_positive_fixed |
| Rosalind Franklin worked as a research assistant a... | incorrect | 0.034 | correct | true_positive_fixed |
| Rosalind Franklin worked under the supervision of ... | incorrect | 0.050 | correct | true_positive_fixed |
| Rosalind Franklin joined the King's College, Londo... | incorrect | 0.004 | correct | true_positive_fixed |
| Franklin's X-ray crystallography work focused on s... | incorrect | 0.067 | correct | true_positive_fixed |
| Photo 51 was taken in 1952 | correct | 0.033 | maintain | correct_maintain |
| Photo 51 is an X-ray diffraction image of DNA | correct | 0.143 | maintain | correct_maintain |
| Photo 51 revealed the helical structure of DNA | correct | 0.058 | maintain | correct_maintain |
| Franklin collaborated with Maurice Wilkins at King... | incorrect | 0.191 | correct | true_positive_fixed |
| Franklin was awarded the Copley Medal in 1958 | incorrect | 0.143 | correct | true_positive_fixed |
| The Copley Medal is described as the highest honor... | incorrect | 0.102 | maintain | missed |
| Galois was born in 1811 | correct | 0.114 | maintain | correct_maintain |
| Galois was born in the village of Étampes, France | incorrect | 0.213 | correct | true_positive_fixed |
| Galois entered the École Polytechnique in 1829 | incorrect | 0.214 | correct | true_positive_fixed |
| The École Polytechnique is a prestigious military ... | correct | 0.030 | maintain | correct_maintain |
| Galois graduated from the École Polytechnique in 1... | incorrect | 0.121 | retract | true_positive_fixed |
| Galois graduated at the age of 20 | incorrect | 0.035 | correct | true_positive_fixed |
| Galois graduated with honors | incorrect | 0.012 | correct | true_positive_fixed |
| Galois published papers on theory of equations and... | incorrect | 0.016 | correct | true_positive_fixed |
| Galois earned his doctorate from the University of... | incorrect | 0.031 | correct | true_positive_fixed |
| Galois earned his doctorate at the age of 23 | incorrect | 0.013 | correct | true_positive_fixed |
| Galois penned a final letter titled 'Letter to the... | incorrect | 0.171 | correct | true_positive_fixed |
| Galois died on June 30th | incorrect | 0.114 | correct | true_positive_fixed |
| Galois died in 1835 | incorrect | 0.054 | correct | true_positive_fixed |
| Galois died at the age of 25 | incorrect | 0.004 | correct | true_positive_fixed |
| Galois died in Lyon, France | incorrect | 0.203 | correct | true_positive_fixed |
| Galois died at the hands of a group of soldiers | incorrect | 0.034 | correct | true_positive_fixed |

## Comparison to Paper Baseline

| Metric | Our Pipeline | Goodfire RLFR (with RL) |
|--------|-------------|------------------------|
| Hallucination reduction | 91.8% | 58% |

Note: The paper achieves 58% with RL training. Our pipeline uses probes + Claude intervention
without any RL, so lower reduction is expected.

## Interpretation

The truthfulness probe shows limited discriminative power on free-form generation (potential domain shift from TruthfulQA format).
Claude self-check (0.986) outperforms the probe (0.592), suggesting that language-level verification is currently stronger than activation-based detection for this task.
