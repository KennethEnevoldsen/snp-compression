"""
Convert the "dsmwpred", "data", "ukbb", "geno" to tensors in chunk
"""
import os
from typing import Tuple
from plinkio import plinkfile
import torch
import time
from wasabi import msg
from pathlib import Path

plink_path = os.path.join("..", "..", "dsmwpred", "data", "ukbb", "geno")
save_path = os.path.join("data", "processed", "tensors", "dsmwpred")

plink = plinkfile.open(plink_path)

samples = plink.get_samples()
loci = plink.get_loci()

n_samples = len(samples)
n_loci = len(loci)
batch = 1000


def batch_plink_to_tensor(
    interval: Tuple[int, int], n_loci: int = n_loci
) -> torch.Tensor:
    s = time.time()
    stop = False
    s, e = interval
    batch_size = e - s
    X = torch.zeros(batch_size, n_loci, dtype=torch.int8)
    for r, row in enumerate(plink):
        # and write file
        for c, geno in enumerate(row):
            if c < (interval[0] - 1):
                continue
            if c == interval[1]:
                stop = True
                continue
            X[r][c] = geno
        if stop is True:
            break
    y = torch.tensor([s.phenotype for s in samples[s:e]], dtype=torch.int8)
    return X, y


def interval_gen(n_samples, step=3000):
    for i in range(0, n_samples, step):
        if i == 0:
            ii = 0
        else:
            yield (ii, i)
            ii = i


intervals = list(interval_gen(n_samples, step=10_000))
intervals.append((intervals[-1][1], n_samples))

plink_name = os.path.split(plink_path)[-1]
Path(save_path).mkdir(parents=True, exist_ok=True)

ss = time.time()
for interval in intervals[-1:]:  # change this assuming it works
    s = time.time()
    X, y = batch_plink_to_tensor(interval, n_loci)
    e = time.time() - s
    msg.info(interval, "finished")

    x_path = os.path.join(
        save_path, f"x_{interval[0]}-{interval[1]}_" + plink_name + ".pt"
    )
    y_path = os.path.join(
        save_path, f"y_{interval[0]}-{interval[1]}_" + plink_name + ".pt"
    )

    msg.info(f"Writing SNPs/genotype (X) to {x_path}")
    torch.save(X, x_path)
    msg.info(f"Writing phenotype (Y) to {y_path}")
    torch.save(y, y_path)

    print("\t Time: ", e)

e = time.time() - ss
print("Total Time: ", e)
plink.close()
