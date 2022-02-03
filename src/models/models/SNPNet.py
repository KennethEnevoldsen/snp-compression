"""
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

https://github.com/arnaghosh/Auto-Encoder/blob/master/resnet.py
"""

from typing import Callable, List, Optional
from torch import nn
import torch
from torch.nn import functional as F
from functools import partial

from src.data.data_handlers import snps_to_one_hot


class OneHotInput(nn.Module):
    def __init__(
        self,
        filters=64,
        kernel: int = 7,
        stride=2,
        categories: int = 4,
        activation: nn.Module = nn.ReLU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.kernel = (kernel, categories)
        self.activation = activation(inplace=True)
        self.conv = nn.Conv2d(
            1,
            filters,
            kernel_size=self.kernel,
            stride=stride,
            bias=False,
            padding="valid",
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=stride)
        self.norm1 = norm_layer(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = snps_to_one_hot(x)
        x = torch.unsqueeze(x, 1)
        x = x.type(torch.float)

        x = self.conv(x)
        if x.shape[2] % 2 != 0:
            x = F.pad(x, pad=(0, 0, 1, 0))  # to ensure not invalid neurons
        x = self.pool(x)
        x = self.norm1(x)
        return self.activation(x)


class OneHotOutput(nn.Module):
    def __init__(
        self,
        filters=64,
        kernel: int = 7,
        stride=2,
        categories: int = 4,
        activation: nn.Module = nn.ReLU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.kernel = (kernel, categories)
        self.activation = activation(inplace=True)
        self.tconv = nn.ConvTranspose2d(
            filters,
            filters,
            kernel_size=self.kernel,
            stride=stride,
            bias=False,
        )
        self.norm1 = norm_layer(filters)
        self.conv1 = nn.Conv2d(
            filters,
            1,
            kernel_size=self.kernel,
            bias=False,
            padding="same",
        )

    def forward(self, x: torch.Tensor, out_size: int) -> torch.Tensor:
        x = self.tconv(x)

        # cut shape to output
        if x.shape[-2] > out_size:
            diff = x.shape[-2] - out_size - x.shape[-2]
            diff_start = diff // 2
            diff_end = diff - diff_start
            x = x[:, :, diff_start:-diff_end, :]

        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = torch.squeeze(x, 1)
        return x


def conv3xN(
    in_planes: int, out_planes: int, emb_dim: int = 1, stride: int = 1
) -> nn.Conv2d:
    """3xN convolution with padding"""
    padding = (1, 0) if stride == 1 else 0
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(3, 1),
        stride=stride,
        padding=padding,
        bias=False,
    )


def tconv3xN(
    in_planes: int, out_planes: int, emb_dim: int = 1, stride: int = 1
) -> nn.Conv2d:
    """3xN convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=(3, 1),
        stride=stride,
        padding=0,
        bias=False,
    )


def conv1x1(
    in_filters: int, out_filters: int, stride: int = 1, padding: int = 0
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_filters,
        out_filters,
        kernel_size=1,
        stride=stride,
        bias=False,
        padding=padding,
    )


def tconv1x1(in_filters: int, out_filters: int, stride: int = 1) -> nn.Conv2d:
    """1x1 transposed convolution"""
    return nn.ConvTranspose2d(
        in_filters, out_filters, kernel_size=1, stride=stride, bias=False
    )


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        infilters: int,
        filters: int,
        downsample: Optional[nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        width: int = 64,
        embedding_dimension: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.activation = activation(inplace=True)
        self.stride = stride
        self.width = width
        self.infilters = infilters
        self.filters = filters

        self.conv1x1_1 = conv1x1(infilters, width)
        self.norm1 = norm_layer(width)
        self.conv_strided = conv3xN(
            width, width, emb_dim=embedding_dimension, stride=stride
        )
        self.norm2 = norm_layer(width)
        self.conv1x1_2 = conv1x1(width, filters * self.expansion)
        self.norm3 = norm_layer(filters * self.expansion)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sav = x  # TODO remove
        if self.downsample:
            identity = self.downsample(x)
        else:
            identity = x

        x = self.conv1x1_1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # 2P  =2 + 2P

        if self.downsample:
            # ((W - K + 2P) / S) + 1
            # where W is width, K kernel, p padding, s stride
            x = F.pad(x, pad=(0, 0, 2, 2))

        x = self.conv_strided(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv1x1_2(x)
        x = self.norm3(x)

        x += identity
        x = self.activation(x)
        return x


class SNPEncoder(nn.Module):
    def __init__(
        self,
        block: nn.Module = Bottleneck,
        layers: List[int] = [1, 3, 4, 6],
        input_module: nn.Module = OneHotInput,
        filters: List[int] = [64, 128, 256],
        activation: str = "relu",
        norm_layer: nn.Module = nn.BatchNorm2d,
        kaimin_init: bool = True,
        width: int = 64,
    ):
        super().__init__()
        self.forward_shapes = []
        self.filters = filters
        self.infilters = filters[0]
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
            self._activation = nn.ReLU
        else:
            raise NotImplementedError("only relu implemented as activation func.")
        self._norm_layer = norm_layer

        self.conv1 = input_module(
            filters=filters[0],
            kernel=7,
            stride=2,
            categories=4,
            activation=self._activation,
        )
        self.pad = partial(F.pad, pad=(0, 0, 1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=2)

        self.layers = nn.ModuleList()
        for n_layers, n_filters in zip(layers, filters):
            self.layers.append(
                self._make_layer(block, n_filters, n_layers, stride=2, width=width)
            )

        if kaimin_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=activation
                    )

    def _make_layer(
        self,
        block: Bottleneck,
        filters: int,
        blocks: int,
        stride: int = 1,
        width: int = 64,
    ):
        norm_layer = self._norm_layer
        downsample = None

        if (stride != 1) or (self.infilters != block.expansion * filters):
            downsample = nn.Sequential(
                conv1x1(
                    self.infilters, filters * block.expansion, stride, padding=(1, 0)
                ),
                norm_layer(filters * block.expansion),
            )

        layers = []
        layers.append(
            block(
                infilters=self.infilters,
                filters=filters,
                stride=stride,
                downsample=downsample,
                norm_layer=norm_layer,
                activation=self._activation,
                width=width,
            )
        )
        self.infilters = filters * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    infilters=self.infilters,
                    filters=filters,
                    stride=1,
                    downsample=None,
                    norm_layer=norm_layer,
                    activation=self._activation,
                    width=width,
                ),
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_shapes = []
        self.forward_shapes.append(x.shape[-1])

        print(f"input - x.shape={x.shape}")
        x = self.conv1(x)
        self.forward_shapes.append(x.shape[-2])
        print(f"first conv - x.shape={x.shape}")
        for i, l in enumerate(self.layers):
            x = l(x)
            print(f"\tlayer {i} - x.shape={x.shape}")
            self.forward_shapes.append(x.shape[-2])

        # pool across filters (as apposed to adaptivepool2d)
        x = x.mean(dim=1).unsqueeze(1)
        return x


class ReverseBottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        infilters: int,
        filters: int,
        upsample: Optional[nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        width: int = 64,
        embedding_dimension: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.activation = activation(inplace=True)
        self.stride = stride
        self.width = width
        self.infilters = infilters
        self.filters = filters

        self.conv1x1_1 = tconv1x1(infilters, width)
        self.norm1 = norm_layer(width)
        self.conv_strided = tconv3xN(
            width, width, emb_dim=embedding_dimension, stride=stride
        )
        self.norm2 = norm_layer(width)
        self.conv1x1_2 = tconv1x1(width, filters * self.expansion)
        self.norm3 = norm_layer(filters * self.expansion)
        self.upsample = upsample

    def forward(self, x: torch.Tensor, enc_size: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor
            enc_size (int): Encoding size of the correspond layer in the encoder
        """
        if self.upsample:
            x_ = F.pad(x, pad=(0, 0, 1, 0))
            identity = self.upsample(x_)
        else:
            x_ = F.pad(x, pad=(0, 0, 1, 1))
            identity = x_

        x = self.conv1x1_1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv_strided(x)
        assert x.shape[-2] == identity.shape[-2]

        # cut shape to match encoder
        if enc_size and x.shape[-2] > enc_size:
            diff = x.shape[-2] - enc_size
            diff_start = diff // 2
            diff_end = diff - diff_start
            x = x[:, :, diff_start:-diff_end, :]
            identity = identity[:, :, diff_start:-diff_end, :]

        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv1x1_2(x)
        x = self.norm3(x)

        x += identity
        x = self.activation(x)
        return x


class SequentialWithArgs(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class SNPDecoder(nn.Module):
    def __init__(
        self,
        block: nn.Module = ReverseBottleneck,
        layers: List[int] = [6, 4, 3, 3],
        output_module: nn.Module = OneHotOutput,
        filters: List[int] = [128, 128, 64, 32],
        activation: str = "relu",
        norm_layer: nn.Module = nn.BatchNorm2d,
        width: int = 64,
    ):
        super().__init__()
        self.filters = filters
        self.infilters = 1
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
            self._activation = nn.ReLU
        else:
            raise NotImplementedError("only relu implemented as activation func.")
        self._norm_layer = norm_layer

        self.conv1 = output_module(
            filters=filters[0],
            kernel=7,
            stride=2,
            categories=4,
            activation=self._activation,
        )
        self.pad = partial(F.pad, pad=(0, 0, 1, 1))
        self.pool = nn.MaxUnpool2d(kernel_size=(2, 1), stride=2)

        self.layers = nn.ModuleList()
        for n_layers, n_filters in zip(layers, filters):
            self.layers.append(
                self._make_layer(block, n_filters, n_layers, stride=2, width=width)
            )

    def _make_layer(
        self,
        block: ReverseBottleneck,
        filters: int,
        blocks: int,
        stride: int = 1,
        width: int = 64,
    ):
        norm_layer = self._norm_layer
        upsample = None

        if (stride != 1) or (self.infilters != block.expansion * filters):
            upsample = nn.Sequential(
                tconv1x1(self.infilters, filters * block.expansion, stride),
                norm_layer(filters * block.expansion),
            )

        layers = nn.ModuleList()
        layers.append(
            block(
                infilters=self.infilters,
                filters=filters,
                stride=stride,
                upsample=upsample,
                norm_layer=norm_layer,
                activation=self._activation,
                width=width,
            )
        )
        self.infilters = filters * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    infilters=self.infilters,
                    filters=filters,
                    stride=1,
                    upsample=None,
                    norm_layer=norm_layer,
                    activation=self._activation,
                    width=width,
                ),
            )
        return SequentialWithArgs(layers)

    def forward(self, x: torch.Tensor, encoder_shapes: List[int]) -> torch.Tensor:
        print(f"decoder input - x.shape={x.shape}")
        for i, l in enumerate(self.layers):
            x = l(x=x)
            print(f"\tlayer {i} - x.shape={x.shape}")
            # assert x.shape[-2] == encoder_shapes[-(i+2)]
        x = self.conv1(x, encoder_shapes[0])
        print(f"\Output Layer - x.shape={x.shape}")
        return x


if __name__ == "__main__":
    x = torch.zeros(10, 1, 44638, 4)  # samples, 'channel', snps, onehot
    # 44638 -> 1395 is a ~32 times reduction
    enc = SNPEncoder()
    h = enc(x)

    # forward pass works

    dec = SNPDecoder()
    x_hat = dec(h, enc.forward_shapes)
