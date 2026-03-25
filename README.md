# OptimalBattery

The OptimalBattery project implements the analyses underlying the paper **"Multi-Task Batteries for Precision Functional Mapping"** by Arafat, Nettekoven, Xiang & Diedrichsen (2026).

## Dependencies

**Diedrichsen Lab packages:**
- [Diedrichsenlab/Functional_Fusion](https://github.com/DiedrichsenLab/Functional_Fusion)
- [Diedrichsenlab/PcmPy](https://github.com/DiedrichsenLab/PcmPy)

**Other dependencies:**
- numpy
- pandas
- torch
- scipy
- matplotlib
- seaborn
- nibabel

## Core Modules

| Module | Description |
|--------|-------------|
| `construct.py` | Optimal battery construction using eigenvalue-based criteria (variance, inverse trace, log determinant) |
| `simulate.py` | Simulation functions for parcellation and connectivity modeling |
| `evaluate.py` | Evaluation metrics for assessing battery performance |
| `estimate.py` | Estimation of parcellations (U matrices) and functional profiles (V matrices) |
| `design.py` | Experimental design matrix generation for grouped vs. interspersed designs |
| `util.py` | Utility functions for matrix operations and data preprocessing |
| `plot.py` | Visualization functions for publication-quality figures |

## Scripts / Code to replicate different parts of the paper

### Section 1: Functional Localization (Fig 1)

**Single-contrast vs. multi-task localizers**

| Script | Description |
|--------|-------------|
| `scripts/_1functional_localization/1a.localization_fSNR.py` | Estimate fSNR distribution from MDTB dataset (Fig 1d) |
| `scripts/_1functional_localization/1b.localization_sim.py` | Simulate single-contrast vs multi-task localization (Fig 1e-g) |
| `scripts/_1functional_localization/1c.localization_real_fSNRxSize.py` | fSNR vs ROI size correlation in real data |
| `scripts/_1functional_localization/1d.localization_real.py` | Empirical localization of cerebellar language region (Fig 1h-k) |

**Figures:** `paper_figures/1.localization.ipynb`

---

### Section 2: Optimal Battery Selection (Fig 2)

**Parcellation and connectivity simulations & real data**

| Script | Description |
|--------|-------------|
| `scripts/_2battery_selection/2a.parcellation_sim.py` | Parcellation simulation across battery sizes (Fig 2a) |
| `scripts/_2battery_selection/2b.parcellation_real_cortical.py` | Neocortical parcellation on MDTB (Fig 2c) |
| `scripts/_2battery_selection/2c.parcellation_real_cerebellar.py` | Cerebellar parcellation on MDTB (Fig S2b) |
| `scripts/_2battery_selection/2d.connectivity_sim.py` | Connectivity modeling simulation (Fig 2b) |
| `scripts/_2battery_selection/2e.connectivity_real.py` | Neocortex-cerebellum connectivity on MDTB (Fig 2d) |

**Figures:** `paper_figures/2.battery_selection.ipynb`

---

### Section 3: Experimental Design (Fig 3 & 4)

**Grouped vs. interspersed designs, temporal autocorrelation, carryover effects**

| Script | Description |
|--------|-------------|
| `scripts/_3experimental_design/_1hcp_task_covariance.py` | HCP task covariance matrices & baseline noise estimation (Fig 3a-b) |
| `scripts/_3experimental_design/_2sim_blocked_vs_interspersed.py` | Simulate grouped vs interspersed design reliability (Fig 3c-d) |
| `scripts/_3experimental_design/_3.temporal_autocorrelation.py` | Temporal autocorrelation analysis on MDTB (Fig 4a) |
| `scripts/_3experimental_design/_4.carryover.py` | Task carryover effect estimation (Fig 4b) |

**Figures:** `paper_figures/3.experimental_design.ipynb`

---

### Supplementary Materials

| Script | Description |
|--------|-------------|
| `scripts/z.supp/1.group_vs_indi_cov.py` | Group vs individual covariance matrix comparison (Fig S1) |
| `scripts/z.supp/3.task_library.py` | Build the combined task activation library (79 conditions) |
| `scripts/z.supp/4.region_batteries.ipynb` | Optimal batteries for Motor, PFC, Cerebellum (Table S1) |

---

---

## License

MIT License - Diedrichsen Lab (2024)