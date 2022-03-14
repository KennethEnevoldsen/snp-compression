#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem 64g
#SBATCH -c 32
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

python src/data/plink_to_xarray.py
