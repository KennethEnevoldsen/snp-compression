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


p = pathlib.Path(__file__).parent.parent.parent.resolve()
x = torch.load(os.path.join(p, "data", "processed", "tensors", "x_mhcuvps.pt"))
target = x.T[:32].type(torch.LongTensor)
x = snps_to_one_hot(target)

x = torch.unsqueeze(x, 1)
x = x.permute(0,1,3,2)

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
y_ = encoder(x)

decoder = Decoder(
    input_size=128,
    output=x.shape[-1],
    conv_kernels=[(1, 9), (2, 9), (4, 9)],
    upsampling_kernels=[(1, 3), (2, 4), (2, 4)],
    n_filters=[60, 60, 1],
    strides=[1, 1, 1],
)

x_ = decoder(y_)

x_ = torch.squeeze(x_, 1)
x = torch.squeeze(x, 1)

from torch import nn

x = x[0:1, :, :]
x_ = x_[0:1, :, :]
target = target[0:1, :]

print("x.shape", x.shape)
print("x_.shape", x_.shape)
print("target.shape", target.shape)

loss = nn.NLLLoss()
m = nn.LogSoftmax(dim=1)
out = torch.exp(m(x))

print("x", x[0,:,0])
print("logsoftmax(x)", m(x)[0,:,0])
print("softmax(x)", out[0,:,0])

print(loss(input = m(x), target = target))
print("?", loss(input = x, target = target))


print("zero?:", loss(input = m(x), target = target))
x2 = x
print(x2[0,:,0])
for i in range(1000):
    if target[0, i] == 2:
        x2[0,:,i] = torch.Tensor([0, 0, 40, 0]) # introduce error
    if target[0, i] == 2:
        x2[0,:,i] = torch.Tensor([0, 0, 40, 0]) # introduce error
    x2[0,:,i] = x2[0,:,i]*10

print(x2[0,:,0])
print("zero??:", loss(input = m(x2), target = target))
print("not zero:", loss(input = m(x_), target = target))


print("limited:", loss(input = m(x)[:, :, :10], target = target[:, :10]))
print(m(x)[:, :, :10])
print(x[:, :, :10])
print(target[:, :10])
print(loss(input = m(x)[:, :, :10], target = target[:, :10]).dtype)
print(target.dtype)
print(x.dtype)
print("zero-limited?:", loss(input = m(x)[:, :, :1000], target = target[:, :1000]))

raise Exception("!")

loss = torch.nn.CrossEntropyLoss()



print("x.shape", x.shape)
print("x_.shape", x_.shape)
print("target.shape", target.shape)

x = x[0:1, :, :]
x_ = x_[0:1, :, :]
target = target[0:1, :]

print("x.shape", x.shape)
print("x_.shape", x_.shape)

print("zero?:", loss(x, target))
x2 = x
print(x2[0,:,0])
x2[0,:,0] = torch.Tensor([0, 1, 0, 0])
print(x2[0,:,0])
print("zero??:", loss(x2, target))
print("not zero:", loss(x_, target))


n=0
for i in range(x.shape[-1]):
    if x[0,:,i].argmax() != target[0, i]:
        n+=1 
        print("x[0,:,i]", x[0,:,i])
        print("target[0, i]", target[0, i])

print("n", n)

# x[0,:, 0]
# x_[0,:, 0]

# import torch.nn as nn
# softmax = nn.LogSoftmax(dim=1)
# output = softmax(x_)
# output.shape
# output[0,:, 0]

# loss = nn.BCELoss()
# loss(x, output)

# x.shape ==output.shape

# assert (1 - torch.exp(output[0, :, 0]).sum()) < 0.0001  # is it approximately 1

# loss = nn.BCELoss()
# l = loss(x, x_)
# l

# # TESTING BCELoss
# loss = torch.nn.BCELoss()
# y_hat = torch.Tensor([[0.99, 0.99, 0], [0, 0, 0.99]])
# y = torch.Tensor([[1, 1, 0], [0, 0, 1]])
# loss(y_hat, y)

# y_hat = torch.Tensor([[0.99, 0.99, 0.01], [0.01, 0.01, 0.99]])
# loss(y_hat, y)

# y_hat = torch.Tensor([[0.99, 0.01, 0.01], [0.01, 0.99, 0.99]])
# loss(y_hat, y)

# y_hat = torch.Tensor([[0.99, 0.99, 0.99], [0.01, 0.01, 0.01]])
# loss = torch.nn.BCELoss(reduction="none")
# loss(y_hat, y)

# # TESTING NLLLoss
# loss = nn.NLLLoss()

# y = torch.Tensor([0, 0, 1]).type(torch.LongTensor)
# y_hat = torch.Tensor([[4, 4, 3], [0, 0, 0]])

# # adding batch = 1
# y = torch.unsqueeze(y, 0)
# y_hat = torch.unsqueeze(y_hat, 0)
# y_hat.shape

# m = nn.LogSoftmax(dim=1)
# torch.exp(m(y_hat))
# loss(input = m(y_hat), target = y)


# # which is the same as:
# loss = nn.CrossEntropyLoss()
# loss(input = y_hat, target = y)  # This is the loss to use - but target needs to be changed



# torch.sigmoid(torch.Tensor([1, 0, 0, 0]))
# x = x.type(torch.FloatTensor)
# torch.exp(x)
# loss = nn.NLLLoss()
# loss(x, target)
# loss(x_, target)
# x.shape
# target.shape

# criterion = nn.CrossEntropyLoss()
# criterion(x.permute(0, 2, 1), target)


# criterion(torch.exp(x), target)
# x[0, :, :10]
# target[0, :10]
# loss = nn.BCELoss()
# output = loss(input, target)




# dae = DenoisingAutoencoder(encoder, decoder)


# import torch
# import torch.nn as nn
# criterion = nn.CrossEntropyLoss()
# batch_size = 16
# no_of_classes = 5
# input = torch.randn(batch_size, no_of_classes, 31)
# target = torch.randint(0,4,(batch_size,31))
# input.shape
# target.shape
# target
# import torch.nn.functional as F
# one_hot = F.one_hot(target, num_classes=5)
# one_hot.shape
# one_hot = one_hot.permute((0, 2, 1)) 
# input.shape
# loss = nn.NLLLoss()
# loss(one_hot.type(torch.FloatTensor), target) 
# criterion(input,target)




# m = nn.Sigmoid()
# loss = nn.BCELoss()
# output = loss(input, target)

# dae(x)
# # from pytorch_lightning.loggers import WandbLogger
# # from pytorch_lightning import Trainer

# # wandb_logger = WandbLogger()
# # trainer = Trainer(logger=wandb_logger)

# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)

# output = loss(input, target)
# output


# import numpy as np

# input_torch = torch.randn(1, 3, 2, 5, requires_grad=True)

# one_hot = np.array([[[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]],    
#                     [[0, 0, 0, 0, 0], [1, 1, 1, 0, 0]],
#                     [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]])

# target = np.array([np.argmax(a, axis = 0) for a in target])
# target_torch = torch.tensor(target)

# loss = torch.nn.CrossEntropyLoss()
# output = loss(input_torch, target_torch)
