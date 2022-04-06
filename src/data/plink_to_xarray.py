"""
A script for reading in PLINK file transpose them and convert them into xarrays 
saved as .zarr files
"""

import os
import sys
from wasabi import msg

sys.path.append(".")
sys.path.append("../../.")

from src.data.dataloaders import PLINKIterableDataset

read_path = os.path.join("/home", "kce", "dsmwpred", "data", "ukbb", "geno.bed")
save_path = os.path.join("/home", "kce", "NLPPred", "snp-compression", "data", "interim", "genotype.zarr")

ds = PLINKIterableDataset(read_path)
ds.to_disk(save_path, overwrite=True)

msg.good("Process complete")
