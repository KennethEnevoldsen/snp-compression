from functools import partial
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    A convolutional encoder a fully connected layer at the end
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (1, 35088),
        conv_kernels: List[Tuple[int, int]] = [
            (1, 9),
            (1, 9),
            (1, 9),
        ],
        n_filters: List[int] = [60, 60, 1],
        strides: List[int] = [1, 1, 1],
        maxpool_kernels: List[Tuple[int, int]] = [(1, 4), (1, 4), (1, 3)],
        encode_size: int = 128,
        activation_function: Callable = F.relu,
    ):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.maxpools = nn.ModuleList()
        self.activation = activation_function
        self.padding = []

        self.fc_input = list(input_size)
        for kernel, filters, stride in zip(conv_kernels, n_filters, strides):
            self.padding.append(calc_same_padding(kernel, stride=stride))
            self.convolutions.append(
                nn.Conv2d(self.fc_input[0], filters, kernel_size=kernel, stride=stride)
            )
            self.fc_input[0] = filters
            self.fc_input[1:] = [self.fc_input[1] // stride, self.fc_input[2] // stride]

        for kernel in maxpool_kernels:
            self.maxpools.append(nn.MaxPool2d(kernel))
            self.fc_input[1:] = [
                self.fc_input[1] // kernel[0],
                self.fc_input[2] // kernel[1],
            ]

        input_size = self.fc_input[0] * self.fc_input[1] * self.fc_input[2]
        self.fc = nn.Linear(input_size, encode_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, pad, maxpool in zip(self.convolutions, self.padding, self.maxpools):
            x = self.activation(maxpool(conv(pad(x))))
        x = torch.reshape(x, (x.shape[0], -1))  # keep batch size, flatten the rest
        x = self.activation(self.fc(x))
        return x


class Decoder(nn.Module):
    """
    A convolutional decoder
    """

    def __init__(
        self,
        input_size: int = 128,
        output: int = 35088,
        conv_kernels: List[Tuple[int, int]] = [(9, 1), (9, 1), (9, 1), (100, 1)],
        upsampling_kernels: List[Tuple[int, int]] = [(3, 1), (4, 1), (4, 1)],
        activation_function: Callable = F.relu,
        n_filters: List[int] = [60, 60, 60, 1],
        strides: List[int] = [1, 1, 1],
    ):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.activation = activation_function
        self.input_size = input_size
        self.padding = []

        in_channels = 1
        for kernel, filters, stride in zip(conv_kernels, n_filters, strides):
            self.padding.append(calc_same_padding(kernel, stride=stride))
            self.convolutions.append(
                nn.Conv2d(in_channels, filters, kernel_size=kernel, stride=stride)
            )
            in_channels = filters

        upsampling_factor = 1
        for kernel in upsampling_kernels:
            self.upsampling.append(nn.Upsample(scale_factor=kernel))
            upsampling_factor = upsampling_factor * kernel[1]

        # using a NN to rescale input to match decoder
        dim = output // upsampling_factor
        if dim < input_size:
            raise ValueError(
                "The proposed decoder leads to a output matrix which is bigger than the output"
            )

        self.fc = nn.Linear(input_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        x = self.fc(x)
        x = x.view(x.shape[0], 1, 1, -1)  # batch, channels, h, w
        for conv, pad, upsample in zip(
            self.convolutions, self.padding, self.upsampling
        ):
            x = self.activation(conv(pad(upsample(x))))

        return torch.squeeze(x, 1)


def calc_same_padding(kernel: Tuple[int, int], stride: int = 1):
    """
    calculate padding function equivalent to the tensorflows "same"
    """
    padding = []

    for k in kernel[::-1]:
        if (k - stride) != 0 and (k - stride) % 2 != 0:
            padding += [(k - stride) // 2, (k - stride) // 2 + 1]
        else:
            padding += [(k - stride) // 2] * 2
    return partial(F.pad, pad=padding)


if __name__ == "__main__":
    import os

    from src.data.dataloader import read_plink_as_tensor, reshape_to_cnn

    file = os.path.join("data", "raw", "mhcuvps")
    x, y = read_plink_as_tensor(file)
    x = reshape_to_cnn(x)[0:2]
    x.shape  # batch, n channels, height, width

    encoder = Encoder(
        input_size=(x.shape[-2], x.shape[-1]),
        encode_size=128,
        conv_kernels=[(1, 9), (1, 9), (1, 9)],
        n_filters=[60, 60, 1],
        strides=[1, 1, 1],
        maxpool_kernels=[(1, 4), (1, 4), (1, 3)],
    )
    x_ = encoder(x)
    encoder.padding

    decoder = Decoder(
        input_size=(1, 128),
        output=x.shape[-1],
        conv_kernels=[(1, 9), (1, 9), (1, 9)],
        upsampling_kernels=[(1, 3), (1, 4), (1, 4)],
        n_filters=[60, 60, 1],
        strides=[1, 1, 1],
    )
    y = decoder(x_)
    y.unique()
