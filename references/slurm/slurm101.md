
# 101 on Slurm


# Remember to run a interactive window


```
srun -p normal --pty -c 4 --mem=16g bash
```

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
sbatch references/slurm/train_cnn.sh -A NLPPred