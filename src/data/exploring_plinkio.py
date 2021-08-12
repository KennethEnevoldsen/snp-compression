"""
"""

import os
import shutil
from pathlib import Path

from plinkio import plinkfile

origin = os.path.join(Path.home(), "NLPPred")
file = os.path.join(origin, "mhcuvps")
plink_file = plinkfile.open(file)


sample_list = plink_file.get_samples()
locus_list = plink_file.get_loci()

locus = locus_list[0]
len(locus_list)

locus.position
locus.chromosome
locus.bp_position
locus.allele1
locus.allele2
locus.name

sample = sample_list[0]
dir(sample)

sample.affection
sample.father_iid
sample.fid
sample.iid
sample.mother_iid
sample.phenotype
sample.sex

from collections import Counter
count = Counter([genotype for row in plink_file for genotype in row])
count

i = 0
for locus, row in zip( locus_list, plink_file ):
    for sample, genotype in zip( sample_list, row ):
        i += 1
        if i > 100:
            continue
        print( "Individual {0} has genotype {1} for snp {2}.".format( sample.iid, genotype, locus.name ) )


row.allele_counts()