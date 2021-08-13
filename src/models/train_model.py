"""

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
from src.data.dataloader import snps_to_one_hot

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from src.models.models.pl_wrappers import PlOnehotWrapper


# Build dataset
p = pathlib.Path(__file__).parent.parent.parent.resolve()
x = torch.load(os.path.join(p, "data", "processed", "tensors", "x_mhcuvps.pt"))
target = x.T.type(torch.LongTensor)
x = snps_to_one_hot(target)

x = torch.unsqueeze(x, 1)
x = x.permute(0,1,3,2)

x = x.type(torch.FloatTensor)  # as it needs to be a float
train = TensorDataset(x[:7000], target)
val = TensorDataset(x[7000:], target)
train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val, batch_size=32, shuffle=True, num_workers=4)

# Build model
encoder = Encoder(
    input_size=(x.shape[-3], x.shape[-2], x.shape[-1]),
    encode_size=128,
    conv_kernels=[(4, 9), (2, 9), (1, 9)],
    n_filters=[32, 16, 1],
    strides=[1, 1, 1],
    maxpool_kernels=[(2, 4), (2, 4), (1, 3)],
)

decoder = Decoder(
    input_size=128,
    output=x.shape[-1],
    conv_kernels=[(1, 9), (2, 9), (4, 9)],
    upsampling_kernels=[(1, 3), (2, 4), (2, 4)],
    n_filters=[60, 60, 1],
    strides=[1, 1, 1],
)

dae = DenoisingAutoencoder(encoder, decoder)

model = PlOnehotWrapper(model=dae)


# train model
wandb_logger = WandbLogger()
trainer = Trainer(logger=wandb_logger, log_every_n_steps=100)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)