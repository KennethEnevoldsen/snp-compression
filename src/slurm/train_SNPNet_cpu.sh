#!/bin/bash
#SBATCH -t 3:0:0
#SBATCH --mem 64g
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out

python src/models/train_SNPNet.py
