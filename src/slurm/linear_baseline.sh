#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem 32g
#SBATCH -c 4
#SBATCH --output ./reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

which python

echo 'running linear model (height)'
ldak/ldak5.2.linux --linear ldak_results/height_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/height.train --mpheno 1 --max-threads 4
echo 'running linear model (hyper)'
ldak/ldak5.2.linux --linear ldak_results/hyper_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/hyper.train --mpheno 1 --max-threads 4
echo 'running linear model (reaction)'
ldak/ldak5.2.linux --linear ldak_results/reaction_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/reaction.train --mpheno 1 --max-threads 4
echo 'running linear model (snoring)'
ldak/ldak5.2.linux --linear ldak_results/snoring_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/snoring.train --mpheno 1 --max-threads 4
echo 'running linear model (pulse)'
ldak/ldak5.2.linux --linear ldak_results/pulse_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/pulse.train --mpheno 1 --max-threads 4
echo 'running linear model (sbp)'
ldak/ldak5.2.linux --linear ldak_results/sbp_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/sbp.train --mpheno 1 --max-threads 4
echo 'running linear model (quals)'
ldak/ldak5.2.linux --linear ldak_results/quals_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/quals.train --mpheno 1 --max-threads 4
echo 'running linear model (neur)'
ldak/ldak5.2.linux --linear ldak_results/neur_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/neur.train --mpheno 1 --max-threads 4
echo 'running linear model (chron)'
ldak/ldak5.2.linux --linear ldak_results/chron_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/chron.train --mpheno 1 --max-threads 4
echo 'running linear model (fvc)'
ldak/ldak5.2.linux --linear ldak_results/fvc_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/fvc.train --mpheno 1 --max-threads 4
echo 'running linear model (ever)'
ldak/ldak5.2.linux --linear ldak_results/ever_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/ever.train --mpheno 1 --max-threads 4
echo 'running linear model (bmi)'
ldak/ldak5.2.linux --linear ldak_results/bmi_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/bmi.train --mpheno 1 --max-threads 4
echo 'running linear model (awake)'
ldak/ldak5.2.linux --linear ldak_results/awake_baseline --bfile /home/kce/dsmwpred/data/ukbb/geno --pheno /home/kce/dsmwpred/data/ukbb/awake.train --mpheno 1 --max-threads 4