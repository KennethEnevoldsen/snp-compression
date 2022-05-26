#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem 32g
#SBATCH -c 4
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

/home/kce/miniconda3/envs/snpnet/bin/python /home/kce/NLPPred/snp-compression/src/fine_tune/baselines.py