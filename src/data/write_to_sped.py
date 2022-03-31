from typing import Union
from pathlib import Path

import xarray as xr
import numpy as np

from pandas_plink._write import _fill_sample, _fill_variant, _write_fam, _write_bim


def write_to_sped(
    G: xr.DataArray,
    sped: Union[str, Path],
    bim: Union[str, Path, None] = None,
    fam: Union[str, Path, None] = None,
) -> None:
    """Writes dataarray to .sped file.

    A .sped binary file set consists of three files:

    - .sped: containing the genotype.
    - .bim: containing variant information.
    - .fam: containing sample information.

    The user must provide the genotype (dosage) via a :class:`xarray.DataArray`.
    That matrix must have two named dimensions: **sample** and **variant**.

    Args:
        G (xr.DataArray): The genotype matrix.
        sped (str): The path to the .sped file.
        bim (Union[str, Path, None], optional): The path to the .bim file. If
            None, the .bim file will be written to the same directory as the
            .sped file.
        fam (Union[str, Path, None], optional): The path to the .fam file. If
            None, the .fam file will be written to the same directory as the
            .sped file.

    Example:
        >>> from xarray import DataArray
        >>> from src.data.write_to_sped import write_to_sped
        >>>
        >>> G = DataArray(
        ...     [[3.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
        ...     dims=["sample", "variant"],
        ...     coords = dict(
        ...         sample  = ["boffy", "jolly"],
        ...         fid     = ("sample", ["humin"] * 2 ),
        ...
        ...         variant = ["not", "sure", "what"],
        ...         snp     = ("variant", ["rs1", "rs2", "rs3"]),
        ...         chrom   = ("variant", ["1", "1", "2"]),
        ...         a0      = ("variant", ['A', 'T', 'G']),
        ...         a1      = ("variant", ['C', 'A', 'T']),
        ...     )
        ... )
        >>> write_to_sped(G, "./test.sped")

    """
    if G.ndim != 2:
        raise ValueError("G has to be bidimensional")

    if set(list(G.dims)) != set(["sample", "variant"]):
        raise ValueError("G has to have both `sample` and `variant` dimensions.")

    G = G.transpose("sample", "variant")

    sped = Path(sped)

    if bim is None:
        bim = sped.with_suffix(".bim")

    if fam is None:
        fam = sped.with_suffix(".fam")

    bim = Path(bim)
    fam = Path(fam)

    G = _fill_sample(G)
    G = _fill_variant(G)
    _write_sped(G, sped)
    _write_fam(fam, G)
    _write_bim(bim, G)


def _write_sped(G: xr.DataArray, sped: Union[str, Path]) -> None:
    """Writes the .sped file.

    Args:
        G (xr.DataArray): The genotype matrix.
        sped (str): The path to the .sped file.

    """
    sped = Path(sped)
    G.data.astype(np.float32).T.tofile(sped)


if __name__ == "__main__":
    import os

    path = os.path.join(
        "/home", "kce", "NLPPred", "snp-compression", "data", "interim", "genotype.zarr"
    )
    G = xr.open_zarr(path).genotype[:11, :]
    l = [i for i in G]

    import random

    random.shuffle(l)
    out = xr.concat(l, dim="sample", coords="all")
    out

    condensed = np.random.normal(size=(out.shape[0], 100))

    out[0, 0].compute()._coords

    coords = {
        "variant": np.arange(0, condensed.shape[1]),  # create variant id
        "chrom": ("variant", np.repeat(1, condensed.shape[1])),  # add chromosome
        "a0": ("variant", np.repeat("A", condensed.shape[1])),  # add allele 1
        "a1": ("variant", np.repeat("B", condensed.shape[1])),  # add allele 2
        "snp": (
            "variant",
            np.array([f"c{t}" for t in range(condensed.shape[1])]),
        ),  # add SNP id
        "pos": ("variant", np.arange(0, condensed.shape[1])),  # add position
    }
    for k in out.sample._coords:
        coords[k] = ("sample", out.sample[k].data)
    c_geno = xr.DataArray(
        condensed,
        dims=["sample", "variant"],
        coords=coords,
    )

    # c_geno.data.astype(np.float32).T.tofile("sample1.sped")

    write_to_sped(c_geno, "sample_testing.sped")
    # add 1 to basepair position
