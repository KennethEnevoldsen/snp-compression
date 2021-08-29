
# 101 on Slurm
https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands

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


# Frequently used commands

squeue -u kce

<!-- sbatch -A NLPPred slurm/train_cnn_cpu.sh  -->
sbatch -A NLPPred references/slurm/train_cnn.sh