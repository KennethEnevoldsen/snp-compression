from typing import Optional, Union

import os
import sys
from pathlib import Path

sys.path.append(".")
sys.path.append("../../.")

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import xarray as xr

from pandas_plink import write_plink1_bin

from src.data.dataloaders import PLINKIterableDataset, xarray_collate_batch
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
                yield model.xarray_encode(sample)

    encode_stream = encode(dataloader)
    arr = xr.concat(list(encode_stream), dim="sample")
    save_name = os.path.join(save_path, "chrom.bed")
    write_plink1_bin(arr, save_name)


if __name__ == "__main__":
    chrom = 6
    mdl = load_model("rich-thunder-72", config={"chromosome": chrom})
    mdl.to(device=torch.device("cuda"))

    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "interim", "genotype.zarr"
    )
    print("Loading data")
    ds = PLINKIterableDataset(path, chromosome=chrom, to_tensor=False)
    loader = DataLoader(ds, batch_size=32, collate_fn=xarray_collate_batch)

    fpath = os.path.dirname(__file__)
    save_path = os.path.join(fpath, "..", "..", "data", "processed", f"chrom_{chrom}")
    print(f"Saving to path:\n\t{save_path}")
    Path(save_path).mkdir(parents=True, exist_ok=True)

    compress(mdl, loader, save_path=save_path)

# some error with GPU - probably have to move to CPU at the right place (maybe in the dataloader?)
