import os
import sys
from typing import Iterable, Optional, Union

sys.path.append(".")
sys.path.append("../../.")

from pathlib import Path

import dask.array as da

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.data.dataloaders import load_dataset
from src.models.create import create_model
from src.util import create_config_namespace


def filter_key(value):
    epoch, step = [int(v.split("=")[1]) for v in value[:-5].split("-")]
    return epoch, step


def load_model(
    ckpt_file_path: str,
    default_yml_path: Optional[str] = None,
    config: Union[str, dict] = {},
) -> pl.LightningModule:
    """loads model

    Args:
        ckpt_file_path (str): Loads a model ckpt from file path, if the path isn't to a
            specified file, it will assume the model name ans check models/{model_name}/
        config (Union[str, dict]): An additional config or additonal arguments.
        default_yml_path (Optional[str], optional): The default_yml_path, defaults to
        src/configs/default_config.yaml. Defaults to None.

    Returns:
        pl.LightningModule: the model
    """
    fpath = os.path.dirname(__file__)
    if not os.path.isfile(ckpt_file_path):
        path = os.path.join(fpath, "..", "..", "models", ckpt_file_path)
        ckpt_path = os.path.join(path, "None", "version_None", "checkpoints")
        ckpts = sorted(os.listdir(ckpt_path), key=filter_key)

        ckpt_file_path = os.path.join(ckpt_path, ckpts[0])

    if default_yml_path is None:
        default_yml_path = os.path.join(fpath, "..", "configs", "default_config.yaml")

    config = create_config_namespace(default_yml_path, config)
    model = create_model(config)
    model = model.load_from_checkpoint(
        ckpt_file_path, learning_rate=config.learning_rate, model=model.model
    )
    return model


def compress(model: pl.LightningModule, dataloader: DataLoader, save_path: str) -> None:
    def encode(dataloader):
        model.eval()
        with torch.no_grad():
            for sample in dataloader:
                yield model.encode(sample.to(model.device))

    encode_stream = encode(dataloader)
    arr = torch_tensor_stream_to_dask(encode_stream, save_path)
    save_name = os.path.join(save_path, "transposed.zarr")
    arr.T.to_zarr(save_name)


def torch_tensor_stream_to_dask(
    arrays: Iterable[torch.Tensor], save_path: str, overwrite: bool = True
) -> da.Array:
    s_paths = []

    for i, array in enumerate(arrays):
        print("Currently at: ", i)
        dask_arr = da.from_array(array.cpu().numpy())
        save_name = os.path.join(save_path, f"{i}.zarr")
        s_paths.append(save_name)
        dask_arr.to_zarr(save_name, overwrite=overwrite)

    dask_arrays = [da.from_zarr(p) for p in s_paths]
    arr = da.concatenate(dask_arrays, axis=0)
    return arr


if __name__ == "__main__":
    chrom = 6
    mdl = load_model("rich-thunder-72", config={"chromosome": chrom})
    mdl.to(device=torch.device("cuda"))
    mdl.eval()

    train, val, test = load_dataset(chromosome=chrom)
    ds = val
    loader = DataLoader(val, batch_size=32)

    fpath = os.path.dirname(__file__)
    save_path = os.path.join(fpath, "..", "..", "data", "processed", f"chrom_{chrom}")
    print(f"Saving to path:\n\t{save_path}")
    Path(save_path).mkdir(parents=True, exist_ok=True)

    compress(mdl, loader, save_path=save_path)

chrom = 6
read_p = os.path.join("data", "processed", f"chrom_{chrom}", "transposed.zarr")
arr = da.from_zarr(read_p)
arr=arr.squeeze(2).squeeze(0)
test = arr.compute()

import numpy as np
np.savetxt("tmp.txt", test, delimiter=" ")




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



G.shape
fam.shape
bim.shape
