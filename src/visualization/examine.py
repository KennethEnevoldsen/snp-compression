"""
pip install kaleido dash_bio
kaleido is for saving to png
"""
import os
from pathlib import Path

import pandas as pd
import numpy as np
import dash_bio


if __name__ == "__main__":
    project_path = Path("/home/kce/NLPPred/snp-compression")
    save_path = project_path / "reports" / "images" / "single_snp_analysis"
    read_path = project_path / "ldak_results"

    files = [read_path / f for f in os.listdir(read_path) if f.endswith(".assoc")]

    for i, f in enumerate(files):
        title = f.stem
        print(f"{title} - {i} / {len(files)}")
        df = pd.read_csv(f, index_col=None, delimiter=" ")
        df = df[df["Chromosome"].isin({6, 1, 2})]
        df = df.reset_index()

        fig = dash_bio.ManhattanPlot(
            dataframe=df,
            chrm="Chromosome",
            bp="Basepair",
            p="Wald_P",
            snp="Predictor",
            gene=None,
            title=title,
        )

        fig.write_html(save_path / f"{title}.html")
        fig.write_image(save_path / f"{title}.png")

    for i, f in enumerate(sorted(files)):
        title = f.stem
        print(f"{title} - {i} / {len(files)}")
        df = pd.read_csv(f, index_col=None, delimiter=" ")
        df = df[df["Chromosome"].isin({6, 1, 2})]
        df = df.reset_index()

        n_sig = sum(df["Wald_P"] < 5e-08)
        expected = 5e-08 * len(df)
        print("\tN significant (p < 5x10^-8):", n_sig)
        print("\tExpected N given number of SNPs:", expected)
        print("\tN significant / expected: ", n_sig / expected)
        # print("\tMean effect size of significant: ", df["Effect"][df["Wald_P"] < 5e-08].abs().mean())
