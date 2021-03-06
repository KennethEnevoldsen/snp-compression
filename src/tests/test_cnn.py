import sys

sys.path.append(".")
sys.path.append("../../.")
import os

import pathlib


import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.models.cnn import Encoder, Decoder
from src.models.models.DenoisingAutoencoder import DenoisingAutoencoder
from src.data.dataloader import snps_to_one_hot


def test_forward():
    p = pathlib.Path(__file__).parent.parent.parent.resolve()
    x = torch.load(os.path.join(p, "data", "processed", "tensors", "x_mhcuvps.pt"))
    target = x.T[:32].type(torch.LongTensor)
    x = snps_to_one_hot(target)

    x = torch.unsqueeze(x, 1)
    x = x.permute(0, 1, 3, 2)

    x = x.type(torch.FloatTensor)  # as it needs to be a float
    ds_ae = TensorDataset(x, target)
    loader = DataLoader(ds_ae, batch_size=4, shuffle=True)

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
    x_ = dae(x)

    x = torch.squeeze(x, 1)

    loss = nn.CrossEntropyLoss()

    assert loss(x_, target) > loss(x, target)
