
# 101 on Slurm


## Run a job

sbatch {filename}.sh -A NLPPred

A is account

## check status of submitted queue
squeue -u USERNAME

# See available nodes
gnodes


# Commands
sbatch slurm/train_cnn_cpu.sh -A NLPPred