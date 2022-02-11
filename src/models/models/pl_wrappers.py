import time
from typing import Tuple

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import PearsonCorrCoef


class PlOnehotWrapper(pl.LightningModule):
    def __init__(
        self, model: nn.Module, learning_rate: float = 1e-3, num_classes: int = 4
    ):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.accuracy = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1(num_classes=num_classes, mdmc_average="global")
        self.pearson = PearsonCorrCoef()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=3)
        # x.shape should be (batch, channels=1, genotype/snp=4, sequence length)
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        s = time.time()
        x = train_batch
        # x.shape should be (batch, sequence length)

        x_hat = self.forward(x)
        # x.shape should be (batch, genotype/snp=4, sequence length)

        # remove NAs
        x, x_hat = self.__remove_NAs(x, x_hat)

        # calculate metrics
        x.type(torch.LongTensor).to(self.device)
        loss = self.loss(x_hat, x)
        probs = x_hat.softmax(dim=1)
        preds = probs.argmax(dim=1)

        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(probs, x))
        self.log("train_f1", self.f1(preds, x))
        self.log("val_cor", self.pearson(preds, x))
        self.log("train_step/sec", time.time() - s)

        return loss

    def __remove_NAs(
        self, x: torch.Tensor, x_hat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = ~x.isnan()
        x = x[valid]
        valid_one_hot = valid.unsqueeze(1).expand((-1, x[1], -1))
        x_hat = x_hat[valid_one_hot]

    def validation_step(self, val_batch, batch_idx):
        s = time.time()
        x = val_batch

        x_hat = self.forward(x)
        # x.shape should be (batch, genotype/snp=4, sequence length)

        # remove NAs
        x, x_hat = self.__remove_NAs(x, x_hat)

        # calculate metrics
        x.type(torch.LongTensor).to(self.device)
        loss = self.loss(x_hat, x)
        probs = x_hat.softmax(dim=1)
        preds = probs.argmax(dim=1)

        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(probs, x))
        self.log("val_f1", self.f1(preds, x))
        self.log("val_cor", self.pearson(preds, x))
        self.log("val_step/sec", time.time() - s)
