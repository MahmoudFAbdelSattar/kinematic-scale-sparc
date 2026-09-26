```markdown
# SPARC Kinematic Scale Analysis (v13.0)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fzenodo.xxxxxxx-blue)](https://doi.org/)

**Complete, self-contained pipeline for analysing the SPARC galaxy sample and testing for a characteristic kinematic scale in the high-\(\phi\) regime.**

This repository contains the exact code used to produce all results, tables, and figures reported in:

> **"Observational Evidence for a Characteristic Kinematic Scale in the High-\(\phi\) Regime of Spiral Galaxies"**  
> *M. F. Abdel‑Sattar (2026)*

---

## Overview

The code implements the **operational definition** of the characteristic kinematic scale

\[
c_0 = \frac{v_{\text{bar}}}{\sqrt{e^{2\phi}-1}} = \frac{v_{\text{obs}}\,e^{-\phi}}{\sqrt{e^{2\phi}-1}},
\qquad \phi = \ln\!\left(\frac{v_{\text{obs}}}{v_{\text{bar}}}\right),
\]

and performs a comprehensive empirical analysis to test whether \(c_0\) converges to a stable value in the high-\(\phi\) regime of spiral galaxies.

The analysis includes:

- **Progressive cuts** on \(c_0\) to examine convergence.
- **Physical cuts** based on galactocentric radius, the dimensionless field \(\phi\), and measurement quality—demonstrating that the convergence is not an artefact of cutting on \(c_0\) itself.
- **Galaxy‑level cross‑validation** (80/20 split, 1 000 iterations) showing that the central \(c_0\) value is predictable.
- **Median analysis** with bootstrap confidence interval.
- **Monte Carlo error simulations** (10 000 realisations, uncorrelated and correlated).
- **Stratified analysis** of scatter as a function of radius and \(\phi\).
- **Tautology test** (spurious convergence check) using synthetic dark‑halo, MOND, and randomised datasets.
- **Automatic generation of LaTeX tables** for direct inclusion in the manuscript.

The pipeline is purely observational: the dimensionless field \(\phi\) and the scale \(c_0\) are constructed directly from the observed quantities \(v_{\text{obs}}\) and \(v_{\text{bar}}\), without reference to any theoretical framework. A possible physical interpretation of the empirical scale \(c_0\) is developed in a companion paper (Paper II); the present analysis does not depend on that interpretation.

---

## Requirements

- Python 3.10 or later
- The following packages (install via `pip`):

```bash
pip install numpy pandas scipy matplotlib tqdm requests astropy
```

All dependencies are pure Python and available on PyPI. No proprietary software is required.

---

Quick Start

1. Clone or download this repository.
2. Run the main script:

```bash
python sparc_analysis_v13.0.py
```

The script will:

1. Download the SPARC data (≈ 270 kB) from the public repository at astroweb.cwru.edu/SPARC.
2. Cache the parsed data as sparc_raw.pkl for faster subsequent runs.
3. Execute the full analysis pipeline (galaxy analysis, statistics, simulations, physical cuts, cross‑validation, tautology test).
4. Save all figures (PDF) and a LaTeX table (table_physical_cuts.tex) in the output directory.

Expected runtime: ≈ 2–5 minutes on a modern laptop (most of the time is spent on the 10 000 Monte‑Carlo and cross‑validation iterations).

---

Output

All outputs are written to the folder sparc_complete_analysis_exact_c0/.

Figures

File Description
figure_1_c0_distribution.pdf Bootstrap‑mean distributions for full and clean samples.
figure_2_progressive_cuts.pdf Progressive‑cuts analysis (mean ± CI and p‑values).
figure_c0_vs_radius_outliers.pdf c_0 vs. galactocentric radius with outliers highlighted.
figure_3_phi_vs_vbar.pdf Dimensionless field \phi vs. baryonic velocity.
figure_4_rar_comparison.pdf Radial Acceleration Relation with empirical, MOND, and unity predictions.
figure_S1_stratified_analysis.pdf Stratified analysis (scatter vs. radius and \phi).
figure_S3_enhanced_error_analysis.pdf Monte‑Carlo error analysis with correlated simulations.
figure_tautology_test.pdf Spurious‑convergence test (synthetic datasets).

Tables

File Description
table_physical_cuts.tex LaTeX table of means under physically independent selection criteria.

Data files

File Description
sparc_raw.pkl Cached raw SPARC data.
all_measurements_*.csv All c_0 measurements with auxiliary quantities.
galaxy_statistics_*.csv Per‑galaxy statistics including regime classification.

---

Code Structure

```
sparc_analysis_v13.0.py
├── Config                          # All tunable parameters and physical constants
├── SPARCDataFetcher                # Download & parse the public SPARC dataset
├── DimensionlessFieldCalculator    # Operational c₀, φ, accelerations, error propagation
├── ErrorEstimator                  # MAD‑based error estimation
├── GalaxyAnalyzer                  # Per‑galaxy and global computation of φ and c₀
├── StatisticalAnalyzer             # Bootstrap CI, t‑test, progressive cuts
├── EnhancedSimulationComparator    # 10 000 Monte‑Carlo error simulations
├── StratifiedAnalyzer              # Scatter as function of radius and φ
├── PhysicalCutsAnalyzer            # Cuts on R, φ, error; cross‑validation; median; LaTeX export
├── TautologyTest                   # Synthetic datasets: dark‑halo, MOND, randomised
├── MainVisualizer                  # Generate all publication‑quality figures
└── SPARCCompleteAnalysisPipeline   # Top‑level orchestrator
```

---

Reproducibility

· The random seed is fixed in Config.RANDOM_SEED = 42.
· The SPARC data are downloaded from a permanent, versioned URL.
· All intermediate results are exported as CSV files.
· The exact software environment can be reproduced with the requirements.txt file (provided in the repository).

---

Citation

If you use this code or its results in your research, please cite the accompanying paper:

```bibtex
@article{AbdelSattar2026,
  title   = {Observational Evidence for a Characteristic Kinematic Scale in the High-$\phi$ Regime of Spiral Galaxies},
  author  = {Abdel-Sattar, Mahmoud F.},
  journal = {to be submitted},
  year    = {2026}
}
```

and the Zenodo archive of this repository (DOI to be inserted upon acceptance).

---

License

This project is licensed under the MIT License – see the LICENSE file for details.

---

Contact

Mahmoud F. Abdel‑Sattar
Department of Astronomy and Meteorology, Faculty of Science, Al‑Azhar University, Cairo, Egypt
📧 m.f.abdel-sattar@azhar.edu.eg
🔗 GitHub repository

```
