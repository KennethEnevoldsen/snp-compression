#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition gpu
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

SWEEPID="kenevoldsen/snp-compression/4ze1zumz"
venv="SNPNet"

echo 'Activating virtual environment: ' $venv
source /faststorage/project/NLPPred/snp-compression/SNPNet/bin/activate
which python

# echo 'Update sweep config based on local'
# wandb sweep src/train/sweep-01.yaml

echo 'Running wandb sweep'
wandb agent $SWEEPID

