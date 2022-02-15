import time
from typing import Tuple

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import PearsonCorrCoef

from functools import partial


class PlOnehotWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        num_classes: int = 4,
        ignore_index=3,
    ):
        """
        Args:
            ignore_index (int, optional): Ignores index of NA. Defaults to 3.
        """
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lr = learning_rate
        self.accuracy = torchmetrics.Accuracy(ignore_index=ignore_index)
        self.f1 = torchmetrics.F1(
            num_classes=num_classes, mdmc_average="global", ignore_index=ignore_index
        )
        self.pearson = PearsonCorrCoef

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape should be (batch, channels=1, genotype/snp=4, sequence length)
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        s = time.time()
        x = train_batch
        # x.shape should be (batch, sequence length)

        x = torch.nan_to_num(x, nan=3)
        x_hat = self.forward(x)
        # x.shape should be (batch, genotype/snp=4, sequence length)

        # calculate metrics
        x = x.type(torch.LongTensor).to(self.device)
        loss = self.loss(x_hat, x)
        probs = x_hat.softmax(dim=1)
        preds = probs.argmax(dim=1)

        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(preds, x))
        self.log("train_f1", self.f1(preds, x))
        self.log("train_step/sec", time.time() - s)

        return loss

    def validation_step(self, val_batch, batch_idx):
        s = time.time()
        x = val_batch

        notnan = ~x.isnan()
        x = torch.nan_to_num(x, nan=3)
        x_hat = self.forward(x)
        # x.shape should be (batch, genotype/snp=4, sequence length)

        # calculate metrics
        x = x.type(torch.LongTensor).to(self.device)
        loss = self.loss(x_hat, x)

        probs = x_hat.softmax(dim=1)
        preds = probs.argmax(dim=1)

        # calculate pearson corr. ignoring NAs
        pearson = self.pearson()
        for batch in range(probs.shape[0]):
            target_no_na = x[batch][notnan[batch]].unsqueeze(0)
            preds_no_na = preds[batch][notnan[batch]].unsqueeze(0)
            # needs to be float:
            pearson(
                preds_no_na.squeeze(0).type(torch.float).to(self.device),
                target_no_na.squeeze(0).type(torch.float).to(self.device),
            )

        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(preds, x))
        self.log("val_f1", self.f1(preds, x))
        self.log("val_pearson_cor", pearson.compute())
        self.log("val_step/sec", time.time() - s)
