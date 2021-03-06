#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition gpu
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

venv="SNPNet"

echo 'Activating virtual environment: ' $venv
source /faststorage/project/NLPPred/snp-compression/SNPNet/bin/activate
which python

echo 'Training SNPNet'
python src/train/train.py --config src/configs/test_SNPNet.yaml
