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


## Heritability
Send to Doug



## All SNP prediction model
Which model exactly performs this analysis?

Why don't we recommend multiblup? (probably something which should be added?)
https://dougspeed.com/multiblup/

