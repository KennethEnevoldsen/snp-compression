"""
Trains the model and logs to wandb.

python src/models/train_model.py
"""

import sys

sys.path.append(".")
sys.path.append("../../.")
import os

import pathlib


import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.models.cnn import Encoder, Decoder
from src.models.models.DenoisingAutoencoder import DenoisingAutoencoder
from src.data.data_handlers import snps_to_one_hot
from src.models.models.pl_wrappers import PlOnehotWrapper

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

import wandb

# wandb + config
args = {
    "learning_rate": 1e-3,
    "batch_size": 4,
    "num_workers": 4,
    "train_samples": 7000,
    "encode_size": 512,
    "log_step": 1000,
    "check_val_every_n_epoch": 1,
    "optimizer": "Adam",
    "architecture": "CNN",
    "snp_encoding": "one-hot",
    "snp-location-feature": "None",
    "chromosome": 6,
    # filters
    # layers
}

wandb.init(config=args)
config = wandb.config


# Build dataset
p = pathlib.Path(__file__).parent.parent.parent.resolve()
x = torch.load(os.path.join(p, "data", "processed", "tensors", "x_mhcuvps.pt"))
target = x.T.type(torch.LongTensor)
x = snps_to_one_hot(target)

x = torch.unsqueeze(x, 1)
x = x.permute(0, 1, 3, 2)

x = x.type(torch.FloatTensor)  # as it needs to be a float
train = TensorDataset(x[: config.train_samples], target[: config.train_samples])
val = TensorDataset(x[config.train_samples :], target[config.train_samples :])
train_loader = DataLoader(
    train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
)
val_loader = DataLoader(
    val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
)


# Build model
encoder = Encoder(
    input_size=(x.shape[-3], x.shape[-2], x.shape[-1]),
    encode_size=config.encode_size,
    conv_kernels=[(4, 9), (2, 9), (1, 9)],
    n_filters=[32, 16, 1],
    strides=[1, 1, 1],
    maxpool_kernels=[(2, 4), (2, 4), (1, 3)],
)

decoder = Decoder(
    input_size=config.encode_size,
    output=x.shape[-1],
    conv_kernels=[(1, 9), (2, 9), (4, 9)],
    upsampling_kernels=[(1, 3), (2, 4), (2, 4)],
    n_filters=[60, 60, 1],
    strides=[1, 1, 1],
)

config.encoder_conv_layers = len(encoder.convolutions)
config.decoder_conv_layers = len(decoder.convolutions)

dae = DenoisingAutoencoder(encoder, decoder)

model = PlOnehotWrapper(model=dae, learning_rate=config.learning_rate)

wandb.watch(model, log_freq=config.log_step)

# train model
wandb_logger = WandbLogger()

early_stopping = EarlyStopping("val_loss", patience=10)

trainer = Trainer(
    logger=wandb_logger,
    log_every_n_steps=config.log_step,
    check_val_every_n_epoch=config.check_val_every_n_epoch,
    callbacks=[early_stopping],
    gpus=-1,
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
