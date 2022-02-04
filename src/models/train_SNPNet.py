"""
Trains the model and logs to wandb.

python src/models/train_model.py
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

# wandb + config
args = {
    "learning_rate": 1e-3,
    "batch_size": 4,
    "num_workers": 4,
    "log_step": 1000,
    "val_check_interval": 5000,
    "optimizer": "Adam",
    "architecture": "SNPNet",
    "snp_encoding": "one-hot",
    "snp-location-feature": "None",
    "p_val": 0.01,
    "p_test": 0.01,
    "chromosome": 6,
    "filter_factor": 0.5,
    "width": 32,
    # layers
}

wandb.init(config=args)
config = wandb.config

# Build dataset
train, val, test = load_dataset(
    chromosome=config.chromosome, p_val=config.p_val, p_test=config.p_val
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
enc = SNPEncoder(width=config.width, filters=enc_filters)
dec = SNPDecoder(width=config.width, filters=dec_filters)
dae = DenoisingAutoencoder(enc, dec)
model = PlOnehotWrapper(model=dae, learning_rate=config.learning_rate)
wandb.watch(model, log_freq=config.log_step)


# train model
wandb_logger = WandbLogger()

early_stopping = EarlyStopping("val_loss", patience=10)

trainer = Trainer(
    logger=wandb_logger,
    log_every_n_steps=config.log_step,
    val_check_interval=config.val_check_interval,
    callbacks=[early_stopping],
    gpus=-1,
    profiler="simple",
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
