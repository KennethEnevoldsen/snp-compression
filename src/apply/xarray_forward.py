import os
import xarray as xr
import torch
from pandas_plink import read_plink1_bin, write_plink1_bin
import numpy as np

path = os.path.join("/home", "kce", "NLPPred", "data-science-exam", "mhcuvbo.bed")
geno = read_plink1_bin(path)


def dummy_condense(x):
    return torch.zeros((x.shape[0], 1024))


condensed = dummy_condense(geno.compute().data)

# reconstruct xarray
c_geno = xr.DataArray(
    condensed,
    dims=["sample", "variant"],
    coords={"sample": geno.sample,  # add metadata, including iid, chrom, trait, etc.
            "variant": np.arange(0, condensed.shape[1])  # generate variant id
            },
)

write_plink1_bin(c_geno, "mhcuvbo_condensed.bed")
