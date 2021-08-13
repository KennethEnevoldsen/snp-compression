import pytorch_lightning as pl
import torch
import torch.nn as nn


class PlOnehotWrapper(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
        return loss


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        x_hat = self.model(x)
        loss = self.loss(x_hat, y)

        self.log('val_loss', loss)


# pl_model = PlOnehotWrapper(model)
# trainer = pl.Trainer(gpus=1, precision=16, limit_train_batches=0.5)
# trainer.fit(model, train_loader, val_loader)
