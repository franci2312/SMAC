# SMAC

This repository contains the code to reproduce the results presented in the manuscript:

 *"Simultaneous Monitoring of Shape and Surface Color via 4D Point Clouds: A Registration-free Approach"*


## Repository Structure
```
.
├── utils.py               # Core functions (implementation)
├── environment.yml        # Conda environment specification
├── Table_1.py             # Reproduces Table 1 (sensitivity analysis)
└── data-smac/             # Precomputed feature vectors
```

Precomputed feature vectors are provided in `data-smac/` to allow reproduction of the results without requiring access to a high-performance computing cluster. 


## Requirements

Create and activate the conda environment:
```bash
conda env create -f smac.yaml
conda activate smac
```

## Reproducing the Results

**Table 1:**
```bash
python Table_1.py
```

Results (i.e., tables in csv format) and figures are saved to a `results/` folder created automatically.