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
The file test.fam is simply the first six columns of test.ped
```
     1 1 0 0 1 0
     1 2 0 0 1 0z
     1 3 1 2 1 2
     2 1 0 0 1 0
     2 2 0 0 1 2
     2 3 1 2 1 2
```

## bed
The file test.bedd is simply the first six columns of test.ped
```
     1 1 0 0 1 0
     1 2 0 0 1 0z
     1 3 1 2 1 2
     2 1 0 0 1 0
     2 2 0 0 1 2
     2 3 1 2 1 2
```
