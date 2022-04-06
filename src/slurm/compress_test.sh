#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition gpu
#SBATCH --mem 128g
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

which python

echo 'compressing chromosomes'
python src/apply/validate.py
