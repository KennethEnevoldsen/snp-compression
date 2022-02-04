import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import time


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        s = time.time()
        x = train_batch
        x = torch.nan_to_num(x, nan=3)

        # y shape should be (batch, sequence length)
        # x should be the one hot encoded version of y

        # in shape should be (batch, channels=1, genotype/snp=4, sequence length)
        x_hat = self.forward(x)
        x = x.type(torch.LongTensor).to(self.device)
        # out shape should be (batch, genotype/snp=4, sequence length)
        loss = self.loss(x_hat, x)
        self.log("train_loss", loss)

        probs = x_hat.softmax(dim=1)
        preds = probs.argmax(dim=1)
        self.log("train_acc", self.accuracy(probs, x))
        self.log("train_f1", self.f1(preds, x))
        self.log("train_step/sec", time.time() - s)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        x = torch.nan_to_num(x, nan=3)

        # y shape should be (batch, sequence length)

        # in shape should be (batch, channels=1, genotype/snp=4, sequence length)
        x_hat = self.forward(x)
        x = x.type(torch.LongTensor).to(self.device)
        loss = self.loss(x_hat, x)

        self.log("val_loss", loss)
        probs = x_hat.softmax(dim=1)
        preds = probs.argmax(dim=1)

        self.log("val_acc", self.accuracy(probs, x))
        self.log("val_f1", self.f1(preds, x))
