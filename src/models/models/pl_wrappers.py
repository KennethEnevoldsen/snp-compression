import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb

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
        x, y = train_batch
        # y shape should be (batch, sequence length)
        # x should be the one hot encoded version of y

        # in shape should be (batch, channels=1, genotype/snp=4, sequence length)
        x_hat = self.model(x)
        # out shape should be (batch, genotype/snp=4, sequence length)
        loss = self.loss(x_hat, y)
        self.log("train_loss", loss)

        probs = x_hat.softmax(dim=1)
        preds = probs.argmax(dim=1)
        self.log("train_acc", self.accuracy(probs, y))
        self.log("train_f1", self.f1(preds, y))
        # self.log("train_conf_mat", self.conf_mat(preds, y))

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        x_hat = self.model(x)
        loss = self.loss(x_hat, y)

        self.log("val_loss", loss)
        probs = x_hat.softmax(dim=1)
        preds = probs.argmax(dim=1)

        self.log("val_acc", self.accuracy(probs, y))
        self.log("val_f1", self.f1(preds, y))
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    preds=preds.view((-1)), y_true=y.view((-1))
                )
            }
        )
