#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition gpu
#SBATCH --mem 128g
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred


echo 'Fine-tuning SNPNet'
python src/fine_tune/fine_tune.py
