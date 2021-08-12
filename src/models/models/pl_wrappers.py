import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlOnehotWrapper(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__(self)
        model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # y shape should be (batch, sequence length)
        # x should be the one hot encoded version of y

        x_ = torch.unsqueeze(x, 1) 
        # in shape should be (batch, channels=1, genotype/snp=4, sequence length) 
        x_hat = self.model(x_)
        # out shape should be (batch, genotype/snp=4, sequence length) 
        logits = F.log_softmax(x_hat, 1)
        loss = F.nll_loss(logits, y)

        self.log("train_loss", loss)
        return loss


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        x_ = torch.unsqueeze(x, 1) 
        x_hat = self.model(x_)
        logits = F.log_softmax(x_hat, 1)
        loss = F.nll_loss(logits, y)

        self.log('val_loss', loss)


# pl_model = PlOnehotWrapper(model)
# trainer = pl.Trainer(gpus=1, precision=16, limit_train_batches=0.5)
# trainer.fit(model, train_loader, val_loader)
