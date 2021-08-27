#!/bin/bash
#SBATCH -t 3:0:0
#SBATCH --partition gpu
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --gres=gpu:1

python src/models/train_model.py
