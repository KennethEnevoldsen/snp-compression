"""
script for running train script using slurm

Example:
python src/models/train_model.py 
"""
import os

init_batch = """#!/bin/bash
#SBATCH --partition gpu
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --gres=gpu:1"""



# make .sh scripts
for g in grid:
    c, d, ff, de = g
    if d is True:
        python_cmd = f"python autoencoder.py -f {ff} -cl {c} -d True -de {de}"
    elif d is False:
        python_cmd = f"python autoencoder.py -f {ff} -cl {c} -de {de}"

    filename = f"run_autoencoder_D{str(d)}_C{str(c)}_F{str(ff)}_DN{str(de)}.sh"
    with open(filename, "w") as f:
        f.write(init_batch + "\n" + python_cmd)
    os.system(f"sbatch {filename} -A NLPPred")
