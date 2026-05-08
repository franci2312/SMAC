# SMAC

This repository contains the code to reproduce the results presented in the paper:

 *"Simultaneous Monitoring of Shape and Surface Color via 4D Point Clouds: A Registration-free Approach"*

Precomputed feature vectors are provided to allow reproduction of the results without requiring access to a high-performance computing (HPC) cluster. 
Feature vectors are available on Zenodo. Download them [here](https://zenodo.org/records/20084999) and place the `data/` folder in the root of this repository before running any script.

Users who wish to recompute them from scratch using a HPC cluster can refer to the scripts in `hpc/`.


## Repository Structure
```
.
├── utils.py               # Core functions (implementation)
├── environment.yml        # Conda environment specification
├── Table_1.py             # Reproduces Table 1 (sensitivity analysis)
├── Figure_6a.py           # Reproduces Figure 6a
├── Figure_6b.py           # Reproduces Figure 6b
├── Figure_7.py            # Reproduces Figure 7 (all four panels)
├── data/                  # Precomputed feature vectors (downloaded from Zenodo)
└── hpc/                   # HPC scripts to recompute feature vectors from scratch (see hpc/README.md)
```

## Requirements

Create and activate the conda environment:
```bash
conda env create -f smac.yaml
conda activate smac
```

## Reproducing the Results

> **Note:** `Table_1.py` must be run before the figure scripts, as it generates and saves the IC results that are then used by `Figure_6a.py` and `Figure_6b.py`. Similarly, `Figure_6a.py` must be run before running `Figure_7.py`.

**Table 1:**
```bash
python Table_1.py
```

**Figure 6:**
```bash
python Figure_6a.py
python Figure_6b.py
```

**Figure 7** (run once per panel):
```bash
python Figure_7.py --panel 1
python Figure_7.py --panel 2
python Figure_7.py --panel 3
python Figure_7.py --panel 4
```

All scripts accept an optional `--k` argument (default: `0.05`) for the CUSUM parameter; results presented in the Supplementary Material (Section S2) are also reproducible by changing `--k` to `0.1` and `0.2`.

Results (i.e., tables in csv format) and figures are saved to a `results/` folder created automatically.



