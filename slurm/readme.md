
# 101 on Slurm


## Run a job

sbatch {filename}.sh -A NLPPred

A is account

## check status of submitted queue
squeue -u USERNAME

# See available nodes
gnodes


# Commands

squeue -u kce

sbatch slurm/train_cnn_cpu.sh -A NLPPred
sbatch slurm/train_cnn.sh -A NLPPred