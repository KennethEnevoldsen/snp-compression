"""
Trains the model and logs to wandb.

python src/models/train_SNPNet.py
"""

import sys

sys.path.append(".")
sys.path.append("../../.")

import torch
from torch.utils.data import DataLoader
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.data.dataloaders import load_dataset

from src.models.models.SNPNet import SNPEncoder, SNPDecoder
from src.models.models.DenoisingAutoencoder import DenoisingAutoencoder
from src.models.models.pl_wrappers import PlOnehotWrapper


# hyperparameter_defaults = dict(
#     dropout=0.5,
#     batch_size=100,
#     learning_rate=0.001,
#     )


hyperparameter_defaults = {
    "learning_rate": 1e-3,
    "batch_size": 4,
    "num_workers": 4,
    "log_step": 1000,
    "val_check_interval": 5000,  # TODO: set this higher
    "optimizer": "Adam",
    "architecture": "SNPNet",
    "snp_encoding": "one-hot",
    "snp-location-feature": "None",
    "chromosome": 6,
    "filter_factor": 0.5,
    "width": 32,
    "layers_factor": 0.5,
    "fc_layer_size": None,
    "dropout_p": 0.1,  # has prev. been 0.2
    "p_val": 1_000,  # TODO: change to 10_000
    "p_test": 1_000,  # TODO: change to 10_000
    "limit_train": 10_000,  # TODO: change to None
    "gpus": None,  # TODO: change
}
config_dictionary = dict(
    yaml=my_yaml_file,
    params=hyperparameter_defaults,
)

import args

wandb.init(config=hyperparameter_defaults, project="snp-compression-src_models")
wandb.update
config = wandb.config

print(f"GPUS: {config.gpus}")
raise ValueError("!")

# Build dataset
train, val, test = load_dataset(
    chromosome=config.chromosome,
    p_val=config.p_val,
    p_test=config.p_val,
    limit_train=config.limit_train,
)
train_loader = DataLoader(
    train, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
)
val_loader = DataLoader(
    val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
)


# Build model
enc_filters = [int(f * config.filter_factor) for f in [64, 128, 256]]
dec_filters = [int(f * config.filter_factor) for f in [128, 128, 64, 32]]
enc_layers = [int(n * config.layers_factor) for n in [1, 3, 4, 6]]
dec_layers = [int(n * config.layers_factor) for n in [6, 4, 3, 3]]
enc = SNPEncoder(width=config.width, filters=enc_filters, layers=enc_layers)
dec = SNPDecoder(width=config.width, filters=dec_filters, layers=dec_layers)
dae = DenoisingAutoencoder(
    enc, dec, dropout_p=config.dropout_p, fc_layer_size=config.fc_layer_size
)
model = PlOnehotWrapper(model=dae, learning_rate=config.learning_rate)

config.n_of_parameters = sum(p.numel() for p in model.parameters())
config.n_trainable_parameters = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
wandb.watch(model, log_freq=config.log_step)


# train model
wandb_logger = WandbLogger()

early_stopping = EarlyStopping("val_loss", patience=10)

trainer = Trainer(
    logger=wandb_logger,
    log_every_n_steps=config.log_step,
    val_check_interval=config.val_check_interval,
    callbacks=[early_stopping],
    gpus=config.gpus,
    profiler="simple",  # TODO: remove
    max_epochs=1,  # TODO: remove
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def log_conf_matrix(model, dataloader):
    preds = []
    y = []
    for x, y_ in dataloader:
        x_hat = model(x)
        probs_ = x_hat.softmax(dim=1)
        y.append(y_)
        preds.append(probs_.argmax(dim=1))

    preds = torch.cat(preds, dim=0)
    y = torch.cat(y, dim=0)
    wandb.log(
        {
            "conf": wandb.plot.confusion_matrix(
                preds=preds.view((-1)).cpu().numpy(), y_true=y.view((-1)).cpu().numpy()
            )
        }
    )


log_conf_matrix(model, val_loader)
