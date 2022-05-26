
# 101 on Slurm
https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands

# Remember to run a interactive window


```
srun --pty -c 4 --mem=16g bash -A NLPPred
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
sbatch -A NLPPred src/slurm/train_cnn.sh
sbatch -A NLPPred src/slurm/train_SNPNet.sh
sbatch -A NLPPred src/slurm/train_SNPNet_cpu.sh
sbatch -A NLPPred src/slurm/plink_to_zarr.sh

## SSH to with slurm
squeue -u kce
ssh NODEID