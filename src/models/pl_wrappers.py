import time
from typing import Optional

import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchmetrics
from torchmetrics import PearsonCorrCoef


class PlOnehotWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        optimizer="adam",
        num_classes: int = 4,
        ignore_index=3,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Args:
            ignore_index (int, optional): Ignores index of NA. Defaults to 3.
        """
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(ignore_index=ignore_index)
        self.f1 = torchmetrics.F1Score(
            num_classes=num_classes, mdmc_average="global", ignore_index=ignore_index
        )
        self.pearson = PearsonCorrCoef
        self.optimizer_name = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape should be (batch, channels=1, genotype/snp=4, sequence length)
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError(f"{self.optimizer_name}")
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
                preds_no_na.squeeze(0).type(torch.float).to("cpu"),
                target_no_na.squeeze(0).type(torch.float).to("cpu"),
            )

        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(preds, x))
        self.log("val_f1", self.f1(preds, x))
        self.log("val_pearson_cor", pearson.compute())
        self.log("val_step/sec", time.time() - s)

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader
