#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out

python src/data/main.py
