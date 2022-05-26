#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem 32g
#SBATCH -c 4
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

which python

echo 'combing c snps'
python src/apply/combine.py
