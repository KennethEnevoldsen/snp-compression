"""
A script for reading in PLINK file and converting them into Dask arrays saved as .zarr files
"""

import os
import time

from wasabi import msg

import dask
from pandas_plink import read_plink


read_path = os.path.join("..", "..", "dsmwpred", "data", "ukbb", "geno")
save_path = os.path.join("data", "interim")

(bim, fam, G) = read_plink(read_path)

msg.info("Read in data")

with dask.config.set(**{"array.slicing.split_large_chunks": True}):
    for chr in bim["chrom"].unique():
        start = time.time()
        G_ = G[bim["chrom"].to_numpy() == chr].rechunk()
        path = os.path.join(save_path, f"chrom_{chr}")
        G_.T.to_zarr(path, overwrite=True)
        e = time.time() - start
        msg.info(f"Saved chromosome {chr} to {path}. Time spent: {e}")
    msg.good("Process complete")
