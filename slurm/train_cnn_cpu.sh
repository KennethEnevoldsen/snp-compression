#!/bin/bash
#SBATCH --mem 64g
#SBATCH -c 8

python src/models/train_model.py
