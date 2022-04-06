# A short description of PLINK file format

for more please see: http://zzz.bwh.harvard.edu/plink/binary.shtml

## bim
The file test.bim is the extended map file, which also includes the names of the alleles: 

```
(chromosome, SNP, cM, base-position, allele 1, allele 2):
     1       snp1    0       1       G       A
     1       snp2    0       2       1       2
     1       snp3    0       3       A       C
```

## fam
The file test.fam contains the following columns:

- Family ID ('FID')
- Within-family ID ('IID'; cannot be '0')
- Within-family ID of father ('0' if father isn't in dataset)
- Within-family ID of mother ('0' if mother isn't in dataset)
- Sex code ('1' = male, '2' = female, '0' = unknown)
- Phenotype value ('1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control)

And could e.g. look like:
```
     1 1 0 0 1 0
     1 2 0 0 1 0
     1 3 1 2 1 2
     2 1 0 0 1 0
     2 2 0 0 1 2
     2 3 1 2 1 2
```

read more on the [.fam](https://www.cog-genomics.org/plink/1.9/formats#fam) format

## bed
The file test.bed is simply the first six columns of test.ped
```
     1 1 0 0 1 0
     1 2 0 0 1 0
     1 3 1 2 1 2
     2 1 0 0 1 0
     2 2 0 0 1 2
     2 3 1 2 1 2
```
