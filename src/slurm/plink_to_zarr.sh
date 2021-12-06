#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem 64g
#SBATCH -c 16
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out

python src/data/plink_to_torch.py
