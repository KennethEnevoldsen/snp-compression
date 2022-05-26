"""
Fine-tune pre-trained networks on new data samples
"""
import os
import sys
from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np
import dask.array as da

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

sys.path.append(".")
sys.path.append("../../.")

from src.apply.validate import load_model
from src.data.dataloaders import load_dataset, DaskIterableDataset
from src.models.classifier import OneHotClassifier
from src.fine_tune.baselines import phenotypes, pheno_path
from src.util import create_argparser, config_yaml_to_dict


def load_encoders():
    models = {
        6: {"name": "rich-thunder-72"},
        2: {"name": "clear-oath-74"},
        1: {"name": "ruby-sea-73"},
    }
    for chrom in models:
        print("Loading Model")
        mdl = load_model(models[chrom]["name"], config={"chromosome": chrom})
        # mdl.to(device=torch.device("cuda"))
        models[chrom]["model"] = mdl.model.encoder
    return models


def load_geno():
    path = "/home/kce/NLPPred/snp-compression/data/interim/genotype.zarr"
    zds = xr.open_zarr(path)
    geno = zds.genotype
    return geno


def load_dask_geno(chrom=[1, 2, 6]):
    geno = []
    for c in chrom:
        ds, val, test = load_dataset(c, p_val=0, p_test=0)  # no test val
        geno.append(ds.X)
    return da.concatenate(geno, axis=1)


def load_pheno(path: str, geno, dask_geno, split="train"):
    """
    dask geno is much more efficient than geno, but geno has all the metadata attached.
    """
    path = Path(path).with_suffix("." + split)

    df = pd.read_csv(path, sep=" ", header=None)
    assert sum(df[0] == df[1]) == len(df[0])  # FID == IID
    df.columns = ["FID", "IID", "PHENO"]
    df["IID"] = df["IID"].astype(int)
    overlapping_ids = geno.iid.astype(int).isin(df["IID"]).compute()

    pheno_mapping = {iid: pheno for iid, pheno in zip(df["IID"], df["PHENO"])}
    out = geno[overlapping_ids]
    X = dask_geno[overlapping_ids]
    y = np.array(
        [pheno_mapping[i] for i in out.coords["iid"].astype(int).compute().data]
    )
    return X, y, out


def create_data_loaders(phenotype, chrom=[1, 2, 6]):
    geno = load_geno()
    dask_geno = load_dask_geno()

    X, y, meta = load_pheno(pheno_path / phenotype, geno, dask_geno, split="train")
    train = DaskIterableDataset(
        X[:-20_000], y[:-20_000]
    )  # TODO: fix this to a random mask
    val = DaskIterableDataset(X[-20_000:], y[-20_000:])
    X_test, y_test, meta_test = load_pheno(
        pheno_path / phenotype, geno, dask_geno, split="test"
    )
    test = DaskIterableDataset(X_test, y_test)

    metadata = {c: (meta.chrom == str(c)).sum().compute() for c in chrom}
    return train, val, test, metadata


def create_model(metadata, train, val, config):
    i = 0
    chrom_to_snp_indexes = {}
    for chrom, value in metadata.items():
        chrom_to_snp_indexes[chrom] = i, i + value
        i += value

    clf = OneHotClassifier(
        encoders=load_encoders(),
        chrom_to_snp_indexes=chrom_to_snp_indexes,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        train_loader=train,
        val_loader=val,
    )
    return clf


def create_trainer(config) -> Trainer:
    wandb_logger = WandbLogger()
    callbacks = [ModelCheckpoint(monitor="val_loss", mode="min")]
    if config.patience:
        early_stopping = EarlyStopping("val_loss", patience=config.patience)
        if callbacks is None:
            callbacks = []
        callbacks.append(early_stopping)

    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=config.log_step,
        val_check_interval=config.val_check_interval,
        callbacks=callbacks,
        gpus=config.gpus,
        profiler=config.profiler,
        max_epochs=config.max_epochs,
        default_root_dir=config.default_root_dir,
        weights_save_path=os.path.join(config.default_root_dir, config.run_name),
        precision=config.precision,
        auto_lr_find=config.auto_lr_find,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
    )
    return trainer


def main():
    # Create config
    yml_path = Path(__file__).parent / ".." / "configs" / "default_clf_config.yaml"
    parser = create_argparser(yml_path)

    arguments = parser.parse_args()
    wandb.init(
        config=arguments,
        project=f"snp-classifiers-{arguments.phenotype}",
        dir=arguments.default_root_dir,
        allow_val_change=True,
    )
    config = wandb.config
    config.run_name = wandb.run.name

    #   if config is specified update arguments according to config.
    if config.config:
        hyperparameter_config = config_yaml_to_dict(config.config)
        config.update(hyperparameter_config, allow_val_change=True)

    # Create model, dataset, trainer
    train_loader, val_loader, test_loader, metadata = create_data_loaders(
        config.phenotype
    )
    model = create_model(metadata, train_loader, val_loader)
    trainer = create_trainer(config)

    # Train
    if config.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model)
        config.update({"learning_rate": lr_finder.suggestion()}, allow_val_change=True)
        fig = lr_finder.plot(suggest=True)
        wandb.log({"lr_finder.plot": fig})
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
