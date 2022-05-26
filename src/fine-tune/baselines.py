"""
train baselines
- linear classifier
- 2 layer neural network
"""
from pathlib import Path
import sys
import json

import xarray as xr
import dask.array as da
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import pandas as pd

sys.path.append(".")
sys.path.append("../../.")

from src.apply.combine import add_metadata


def load_csnps():
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

    c_snps = add_metadata(c_snps, meta=geno, chrom=chrom, snp=snp_id)
    return c_snps


def load_pheno(path: str, c_snps, split="train"):
    path = Path(path).with_suffix("." + split)

    df = pd.read_csv(path, sep=" ", header=None)
    assert sum(df[0] == df[1]) == len(df[0])  # FID == IID
    df.columns = ["FID", "IID", "PHENO"]
    df["IID"] = df["IID"].astype(int)
    overlapping_ids = c_snps.iid.astype(int).isin(df["IID"]).compute()

    pheno_mapping = {iid: pheno for iid, pheno in zip(df["IID"], df["PHENO"])}
    out = c_snps[overlapping_ids]
    X = out.data.compute()
    y = np.array(
        [pheno_mapping[i] for i in out.coords["iid"].astype(int).compute().data]
    )
    return X, y


c_snps = load_csnps()

phenotypes = [
    "bmi",
    "awake",
    "height",
    "hyper",
    "reaction",
    "snoring",
    "pulse",
    "sbp",
    "quals",
    "neur",
    "chron",
    "fvc",
    "ever",
]

results = {"linear": {}, "MLP": {}}
for pheno in phenotypes:
    data_path = Path("/home/kce/dsmwpred/data/ukbb") / pheno
    X, y = load_pheno(
        "/home/kce/dsmwpred/data/ukbb/awake", c_snps=c_snps, split="train"
    )
    X_test, y_test = load_pheno(
        "/home/kce/dsmwpred/data/ukbb/awake", c_snps=c_snps, split="test"
    )

    mdl = LinearRegression()
    mdl = mdl.fit(X, y)
    results["linear"][pheno] = mdl.score(X_test, y_test)
    print(f"{pheno} linear: {results['linear'][pheno]}")

    mdl = MLPRegressor()
    mdl = mdl.fit(X, y)
    results["MLP"][pheno] = mdl.score(X_test, y_test)
    print(f"{pheno}: MLP {results['MLP'][pheno]}")

save_path = Path("data/results")
save_path.mkdir(exist_ok=True, parents=True)

with open(save_path / "csnps_linear_results.json", "w") as f:
    json.dump(results, f)
