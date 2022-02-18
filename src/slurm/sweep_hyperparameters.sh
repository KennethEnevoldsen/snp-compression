#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition gpu
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --dependency afterok:60842002

NUM=10
SWEEPID="initial_search"
venv="SNPNet"

echo 'Activating virtual environment: ' $venv
conda activate $venv
which python

echo 'Running wandb sweep'
wandb sweep src/train/sweep.yaml --count $NUM $SWEEPID

