import os
from typing import Tuple

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from plinkio import plinkfile
from plinkio.plinkfile import PlinkFile

from wasabi import msg


def plinkfile_to_tensor(file: PlinkFile) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns a two tensors, the first one containing snps with the shapes (n_snps, n_samples) the second containing phenotype with the shape (n_samples)
    """
    x = torch.Tensor([[genotype for genotype in row] for row in file])
    y = torch.Tensor([sample.phenotype for sample in file.get_samples()])
    return x, y


def read_plink_as_tensor(file: str) -> Tuple[torch.Tensor, torch.Tensor]:
    plink_file = plinkfile.open(file)
    x, y = plinkfile_to_tensor(plink_file)
    plink_file.close()
    return x, y


def snps_to_one_hot(x: torch.Tensor) -> torch.Tensor:
    """
    transform tensor of snps to one hot vectors
    """
    one_hot = F.one_hot(x.to(torch.int64), num_classes=4)
    return one_hot


def snps_na_to_0(x: torch.Tensor) -> torch.Tensor:
    """
    transform NA's in tensor of snps to 0
    """
    x[x == 3] = 0
    return x


def reshape_to_cnn(x: torch.Tensor) -> torch.Tensor:
    """
    tranform x into the shape (batch_size, n channels, height, width)
    """
    return torch.reshape(x.T, (x.shape[1], 1, 1, -1))


def write_plink_to_pt(
    plink_path: str = os.path.join("data", "raw", "mhcuvps"),
    save_path: str = os.path.join("data", "processed", "tensors"),
) -> None:
    
    plink_name = os.path.split(plink_path)[-1]
    x, y = read_plink_as_tensor(plink_path)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    x_path = os.path.join(save_path, "x_" + plink_name + ".pt")
    y_path = os.path.join(save_path, "y_" + plink_name + ".pt")

    msg.info(f"Writing SNPs/genotype (X) to {x_path}")
    torch.save(x, x_path)
    msg.info(f"Writing phenotype (Y) to {y_path}")
    torch.save(y, y_path)
    msg.good(f"Finished")


def write_plink_to_pt_batched(
    plink_path: str, save_path: str, batch_size: int = 10_000
) -> None:
    """
    Convert a plink file to pytorch tensor and write them to .pt in chunk of size (number of loci) x (batch size). 
    """
    plink = plinkfile.open(plink_path)

    samples = plink.get_samples()
    loci = plink.get_loci()

    n_samples = len(samples)
    n_loci = len(loci)

    def batch_plink_to_tensor(
        interval: Tuple[int, int], n_loci: int = n_loci
    ) -> torch.Tensor:
        s, e = interval
        batch_size = e - s
        start = time.time()
        X = torch.zeros(batch_size, n_loci, dtype=torch.int8)
        for r, row in enumerate(plink): # looping over SNP
            # and write file
            for c, geno in enumerate(row): # looping over samples
                if c < (interval[0] - 1):
                    continue
                if c >= interval[1]:
                    break
                X[c][r] = geno

            if r % 1000 == 0:
                e = time.time()-start
                print(f"\tCurrently at {r}/{n_loci}. Time {e}")
        y = torch.tensor([s.phenotype for s in samples[s:e]], dtype=torch.int8)
        return X, y

    def interval_gen(n_samples: int, step: int):
        for i in range(0, n_samples, step):
            if i == 0:
                ii = 0
            else:
                yield (ii, i)
                ii = i

    intervals = list(interval_gen(n_samples, step=batch_size))
    intervals.append((intervals[-1][1], n_samples))  # add tail

    plink_name = os.path.split(plink_path)[-1]
    Path(save_path).mkdir(parents=True, exist_ok=True)

    start_overall = time.time()
    for interval in intervals:
        start = time.time()
        X, y = batch_plink_to_tensor(interval, n_loci)

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
        e = time.time() - start
        print("\t Time: ", e)

    e = time.time() - start_overall
    msg.good("Finished. \n\tTotal Time: ", e)
    plink.close()


