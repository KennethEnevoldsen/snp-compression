import os
from typing import Tuple

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


# def make_dataset_for_ae(x: torch.Tensor):
#     from torch.utils.data import TensorDataset, DataLoader
#     return TensorDataset(x, x)
#     # loader = DataLoader(ae, batch_size=3, shuffle=True)
