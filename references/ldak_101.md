# Todo
- [ ] perform single SNP analysis using LDAK --linear (on actual data)
  - [x] on dummy data
  - [ ] on cSNPs
    - [ ] record if any of the SNPs are significant and their effects
- [ ] send data to Doug for heritability
- [ ] Create code for using encoder in a neural prediction model
  - [ ] 
- [ ] Figure out how to do an "all snp prediction model"
  - [ ] Examine MixedLM 

# LDAK Experiments to perform


## Single SNP analysis

[Read more](https://dougspeed.com/single-predictor-analysis/) on single predictor analysis in LDAK

```
ldak/ldak5.2.linux --linear result --bfile ../data-science-exam/mhcabf --pheno ../data-science-exam/mhcabf.fam --mpheno 4
```
Where 4 is the 4th column (where the first 2 is ID) so it is the 6th.

Thus using csnps:
```
ldak/ldak5.2.linux --linear result --sped data/processed/csnp --pheno /home/kce/dsmwpred/data/ukbb/height.train --mpheno 1 --SNP-data NO
```

## Heritability
Send to Doug



## All SNP prediction model
Which model exactly performs this analysis?

Why don't we recommend multiblup? (probably something which should be added?)
https://dougspeed.com/multiblup/

## Merge data
```
ldak/ldak5.2.linux --make-sped csnp.sped --msped combine_list.txt --SNP-data NO
```

Where combine_list.txt is a list of the form:
```
data/processed/csnp_chrom1.sped data/processed/csnp_chrom1.bim data/processed/csnp_chrom1.fam
data/processed/csnp_chrom2.sped data/processed/csnp_chrom2.bim data/processed/csnp_chrom2.fam
data/processed/csnp_chrom6.sped data/processed/csnp_chrom6.bim data/processed/csnp_chrom6.fam
```