#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem 64g
#SBATCH -c 64
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred


echo 'Saving as xarray: ' $venv
python src/data/plink_to_xarray.py
