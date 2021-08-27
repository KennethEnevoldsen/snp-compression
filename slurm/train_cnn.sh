#!/bin/bash
#SBATCH -t 4:0:0
#SBATCH --partition gpu
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -output ./reports/output.%a.out # STDOUT

python src/models/train_model.py
