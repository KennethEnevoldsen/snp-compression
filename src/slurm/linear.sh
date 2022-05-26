#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem 32g
#SBATCH -c 4
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

which python

echo 'running linear model (height)'
ldak/ldak5.2.linux --linear height --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/height.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (hyper)'
ldak/ldak5.2.linux --linear hyper --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/hyper.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (reaction)'
ldak/ldak5.2.linux --linear reaction --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/reaction.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (snoring)'
ldak/ldak5.2.linux --linear snoring --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/snoring.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (pulse)'
ldak/ldak5.2.linux --linear pulse --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/pulse.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (sbp)'
ldak/ldak5.2.linux --linear sbp --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/sbp.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (quals)'
ldak/ldak5.2.linux --linear quals --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/quals.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (neur)'
ldak/ldak5.2.linux --linear neur --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/neur.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (chron)'
ldak/ldak5.2.linux --linear chron --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/chron.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (fvc)'
ldak/ldak5.2.linux --linear fvc --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/fvc.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (ever)'
ldak/ldak5.2.linux --linear ever --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/ever.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (bmi)'
ldak/ldak5.2.linux --linear bmi --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/bmi.train --mpheno 1 --SNP-data NO  --max-threads 4
echo 'running linear model (awake)'
ldak/ldak5.2.linux --linear awake --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/awake.train --mpheno 1 --SNP-data NO  --max-threads 4

