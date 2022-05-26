"""
script for combining the data
"""

from typing import Union, Optional
from pathlib import Path
import sys

import xarray as xr
import dask.array as da
import numpy as np

sys.path.append(".")
sys.path.append("../../.")
from src.data.write_to_sped import write_to_sped


def add_metadata(
    compressed: da.Array,
    meta: xr.DataArray,
    chrom: Union[int, np.ndarray] = 0,
    snp: Optional[np.ndarray] = None,
) -> xr.DataArray:
    """Add metadata to a dask array from an existing xarrary.

    Args:
        compressed (da.Array): A dask array
        meta (xr.DataArray): An xarray
        chrom (Union[int, np.ndarray], optional): The chromosome. Defaults to 0.
            Indicating unknown
        snp (Union[str, np.ndarray, None], optional): The snp ID. Defaults to None. In
            which case it is set to "c{chrom}_{i}", where i is the index of the snp in
            the matrix

    Returns:
        xr.DataArray
    """

    if isinstance(chrom, int):
        chrom = np.repeat(chrom, compressed.shape[1])
    if snp is None:
        snp = np.array([f"c{chrom}_{t}" for t in range(compressed.shape[1])])

    coords = {
        "variant": np.arange(0, compressed.shape[1]),  # create variant id
        "chrom": ("variant", chrom),  # add chromosome
        "a0": ("variant", np.repeat("c1", compressed.shape[1])),  # add allele 1
        "a1": ("variant", np.repeat("c2", compressed.shape[1])),  # add allele 2
        "snp": ("variant", snp),  # add SNP id
        "pos": ("variant", np.arange(0, compressed.shape[1])),  # add position
    }

    for k in meta.sample._coords:
        coords[k] = ("sample", meta.sample[k].data)

    c_geno = xr.DataArray(
        compressed,
        dims=["sample", "variant"],
        coords=coords,
    )

    return c_geno


if __name__ == "__main__":
    data_path = Path("/home/kce/NLPPred/snp-compression/data")
    genotype_path = data_path / "interim" / "genotype.zarr"

    geno = xr.open_zarr(genotype_path)

    c_snps_ = []
    for c in [1, 2, 6]:
        chrom_path = data_path / "processed" / f"chrom_{c}" / "chrom.zarr"
        chrom = da.from_zarr(chrom_path)
        c_snps_.append(chrom.squeeze(-1).squeeze(1))

    # combine the dask arrays
    c_snps = da.concatenate(c_snps_, axis=1)
    chrom = np.array(
        [chrom for snps, chrom in zip(c_snps_, [1, 2, 6]) for _ in range(snps.shape[1])]
    )
    snp_id = np.array(
        [
            f"c{chrom}_{snp_id}"
            for snps, chrom in zip(c_snps_, [1, 2, 6])
            for snp_id in range(snps.shape[1])
        ]
    )

    write_path = data_path / "processed" / "csnp.sped"
    c_snps_w_meta = add_metadata(c_snps, meta=geno, chrom=chrom, snp=snp_id)
    print("writing to sped")
    write_to_sped(c_snps_w_meta, sped=write_path)
