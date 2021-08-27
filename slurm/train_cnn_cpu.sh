#!/bin/bash
#SBATCH -t 3:0:0
#SBATCH --mem 64g
#SBATCH -c 32

python src/models/train_model.py
