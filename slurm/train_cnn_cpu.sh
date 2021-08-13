#!/bin/bash
#SBATCH -t 1-0:0:0
#SBATCH --mem 64g
#SBATCH -c 8

python src/models/train_model.py
