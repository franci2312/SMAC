# HPC Scripts

These scripts were used to generate the precomputed feature vectors provided in `data/`.
They were run on the Phoenix Cluster provided by the Partnership for an Advanced Computing Environment (PACE) at the Georgia Institute of Technology, Atlanta, Georgia, USA.

## Contents
- `bunny.npy`: nominal Stanford Bunny point cloud used as the reference object
- `scripts/`: SLURM submission scripts (`.sh`) and Python simulation scripts (`.py`)
- `environment.yml`: conda environment specification

Users wishing to reproduce the simulations from scratch should adapt the paths in the `.sh` files to their own HPC environment.

Before running experiments, please create the conda environment in your HPC cluster using

```bash
conda env create --name meshenv -f environment.yml
```


## Usage
Precomputed outputs are available in `data/` and are sufficient to reproduce Table 1 and Figures 6 and 7 of the paper.

## Computational Environment
Details on the computational environment:

- OS: Red Hat Enterprise Linux 9.5

- Hardware: Dual Intel Xeon Gold 6226R processors (2.90GHz, 32 cores total), 187 GB RAM per node