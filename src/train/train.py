"""
Trains the model and logs to wandb.

Example:
# using custom config
python src/train/train.py --config src/configs/test_SNPNet.yaml
"""

import os
import sys
from typing import Tuple

sys.path.append(".")
sys.path.append("../../.")

from torch.utils.data import DataLoader
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.data.dataloaders import load_dataset
from src.models.create import create_model
from src.util import create_argparser, config_yaml_to_dict, log_conf_matrix


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    train, val, _ = load_dataset(
        chromosome=config.chromosome,
        p_val=config.p_val,
        p_test=config.p_val,
        limit_train=config.limit_train,
    )
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    return train_loader, val_loader


def create_trainer(config) -> Trainer:
    wandb_logger = WandbLogger()
    callbacks = None
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
    )
    return trainer


def main():
    # Create config
    yml_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "default_config.yaml"
    )
    parser = create_argparser(yml_path)
    arguments = parser.parse_args()
    wandb.init(
        config=arguments,
        project="snp-compression",
        dir=arguments.default_root_dir,
    )
    config = wandb.config
    config.run_name = wandb.run.name

    #   if config is specified update arguments according to config.
    if config.config:
        hyperparameter_config = config_yaml_to_dict(config.config)
        config.update(hyperparameter_config, allow_val_change=True)

    # Create model, dataset, trainer
    train_loader, val_loader = create_dataloaders(config)
    model = create_model(config, train_loader, val_loader)
    trainer = create_trainer(config)

    # Train
    if config.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model)
        config.learning_rate = lr_finder.suggestion()
        fig = lr_finder.plot(suggest=True)
        wandb.log({"lr_finder.plot": fig})
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Finalize
    log_conf_matrix(model, val_loader)


if __name__ == "__main__":
    main()
