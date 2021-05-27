from typing import Callable, List, Tuple, Union
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial

class Encoder(nn.Module):
    """
    A convolutional encoder a fully connected layer at the end
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (1, 18873),
        conv_kernels: List[Tuple[int, int]] = [
            (1, 60),
            (1, 9),
            (1, 9),
        ],
        n_filters: List[int] = [150, 100, 1, ],
        strides: List[int] = [1, 1, 1, 1],
        maxpool_kernels: List[Tuple[int, int]] = [(1, 3), (1, 3), (1, 3)],
        encode_size: int = 128,
        activation_function: Callable = F.relu,
    ):
        super(Encoder, self).__init__()
        self.convolutions = nn.ModuleList()
        self.maxpools = nn.ModuleList()
        self.activation = activation_function
        self.padding = []

        fc_input = input_size
        in_channels = input_size[0]
        for kernel, filters, stride in zip(conv_kernels, n_filters, strides):
            self.padding.append(calc_same_padding(kernel, stride=stride))
            self.convolutions.append(
                nn.Conv2d(in_channels, filters, kernel_size=kernel, stride=stride)
            )
            in_channels = filters
            fc_input = (fc_input[0], fc_input[1]//stride)

        for kernel in maxpool_kernels:
            self.maxpools.append(nn.MaxPool2d(kernel))
            fc_input = (fc_input[0]//kernel[0], fc_input[1]//kernel[1])

        self.fc = nn.Linear(fc_input[1], encode_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        for conv, pad, maxpool in zip(self.convolutions, self.padding, self.maxpools):
            x = maxpool(self.activation(conv(pad(x))))
            print(x.shape)
        x = torch.reshape(x, (x.shape[0], -1)) # keep batch size, flatten the rest
        print(x.shape)
        x = self.activation(self.fc(x))
        return x


class Decoder(nn.Module):
    """
    A convolutional decoder
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (1, 128),
        output: int = 18873,
        conv_kernels: List[Tuple[int, int]] = [
            (9, 1),
            (9, 1),
            (100, 1)
        ],
        upsampling_kernels: List[Tuple[int, int]] = [(3, 1), (3, 1), (3, 1)],
        activation_function: Callable = F.relu,
        n_filters: List[int] = [100, 150, 1],
        strides: List[int] = [1, 1, 1],
    ):
        super(Decoder, self).__init__()
        self.convolutions = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.activation = activation_function
        self.padding = []

        in_channels = input_size[0]
        for kernel, filters, stride in zip(conv_kernels, n_filters, strides):
            self.padding.append(calc_same_padding(kernel, stride=stride))
            self.convolutions.append(
                nn.Conv2d(in_channels, filters, kernel_size=kernel, stride=stride)
            )
            in_channels = filters

        upsampling_factor = 1
        for kernel in upsampling_kernels:
            self.upsampling.append(nn.Upsample(kernel))
            upsampling_factor = upsampling_factor * kernel[0]

        # using a NN to rescale input to match decoder
        dim = output // upsampling_factor
        if dim < input_size[1]:
            raise ValueError("The proposed decoder leads to a output matrix which is bigger than the output")

        self.fc = nn.Linear(input_size[0], dim) if dim != input_size[1] else None
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        if self.fc is not None:
            x = self.fc(x)
        for conv, pad, upsample in zip(self.convolutions, self.padding, self.upsampling):
            x = self.activation(conv(upsample(pad(x))))
        return x




class ConvDenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        decoding_method: str = "upsampling",
        in_channels: int = 1,
        fully_connected: List[int] = [128],
    ):
        super().__init__()
        self.dropout

        # Encoder
        for filter, kernel, stride in zip(convolution_kernel, n_filters, strides):
            self.convolutions.append(
                nn.Conv2d(in_channels, filter, kernel_size=kernel, stride=stride)
            )
            in_channels = filter
        for kernel in max_pools:
            self.maxpools.append(nn.MaxPool2d(kernel))

        # input_ = #CALC
        for dim in fully_connected:
            self.fc.append(nn.Linear(input_, dim))
            input_ = dim

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def make_test_data(hidden_dim = 128, types=10, output_dim=10_000, noise = 0.1, NA: Union[int, None] =0):
    import random

    def add_noise(x):
        for i, x in enumerate(x):
            if random.uniform(0, 1) < noise:
                x[i] = NA

    def make_random_snp(length = output_dim//128):
        start = random.randint(0, 10)
        step = random.randint(1, 10)
        stop = length*step+start
        return [t % 3 for t in range(start, stop, step)]


    hidden_vars = random.choices(range(types), k=hidden_dim)
    patterns = {t: make_random_snp() for t in range(types)}
    input_ = [v for hv in hidden_vars for v in patterns[hv]]
    input_ += (output_dim - len(input_))*[0]
    return torch.Tensor(input_)



def calc_same_padding(kernel: Tuple[int, int], stride: int = 1):
    """
    calculate padding function equivalent to the tensorflows "same"
    """
    padding = []
    
    for k in kernel[::-1]:
        if (k-stride) != 0 and (k-stride) % 2 == 0:
            padding += [(k-stride)//2, (k-stride)//2 + 1]
        else:
            padding += [(k-stride)//2]*2
    return partial(F.pad, pad = padding)


if __name__ == "__main__":
    x = make_test_data()
    x = torch.reshape(x, (1, 1, 1, -1)) # batch, n channels, height, width 
    encoder = Encoder(input_size = (1, 10_000), encode_size=128)
    x_ = encoder(x) # encoder works
    print("Encoder:", x_.shape)

    decoder = Decoder(input_size = (1, 128), output=10_000)
    y = decoder(x_)

    # x.shape

    # conv = nn.Conv2d(1, 30, kernel_size=(1,9), padding=(0, 4))
    # conv(x).shape
    # print(encoder)

    # y = encoder.padding[0](x)
    # y.shape
    # encoder.convolutions[0](encoder.padding[0](x)).shape
    # conv = nn.Conv2d(1, 30, kernel_size=(1,60), padding=(0,29), stride=2)
    # conv(x).shape



    # F.pad(x, (29, 29, 0, 0)).shape
    # padding = []
    # stride = 1; kernel = (1,60)
    # for k in kernel[::-1]:
    #     if (k-stride) != 0 and (k-stride) % 2 == 0:
    #         padding += [(k-stride)//2, (k-stride)//2 + 1]
    #     else:
    #         padding += [(k-stride)//2]*2