import time
from typing import Optional, Dict, Tuple

import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class OneHotClassifier(pl.LightningModule):
    def __init__(
        self,
        encoders: Dict[int, nn.Module],
        chrom_to_snp_indexes: Dict[str, Tuple[int, int]],
        learning_rate: float,
        optimizer="adamw",
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        """_summary_

        Args:
            models (Dict[int, nn.Module]): _description_
            chrom_to_snp_indexes ( Dict[str, Tuple[int, int]]): Chromosome indexed in the matrix.
                E.g. the dict {1: 100, 2: 150}, indicated that the first 100 snps in
                chrom 1 and the next 150 snps is chrom 2.
            learning_rate (float): _description_
            optimizer (str, optional): _description_. Defaults to "adam".
            train_loader (Optional[DataLoader], optional): _description_. Defaults to
                None.
            val_loader (Optional[DataLoader], optional): _description_. Defaults to
                None.
        """
        super().__init__()
        self.encoders = encoders
        self.chrom_to_snp_indexes = chrom_to_snp_indexes
        self.input_snps = sum(chrom_to_snp_indexes.values)
        # self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.clf_layer = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.input_snps
        x = x.to(self.device)

        encoded = []
        for chrom, chrom_encoder in self.encoders.items():
            lower, upper = self.chrom_to_snp_indexes[chrom]
            chrom_snps = x[:, lower:upper]
            encoded_chrom = chrom_encoder(chrom_snps)
            encoded.append(encoded_chrom)
        return torch.concat(encoded, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x)
        if self.clf_layer is None:
            self.clf_layer = nn.Linear(encoded.shape[1], 1)
        return self.clf_layer(encoded)

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

        self.log("train_loss", loss)
        self.log("train_step/sec", time.time() - s)
        return loss

    def validation_step(self, val_batch, batch_idx):
        s = time.time()
        x = val_batch

        x = torch.nan_to_num(x, nan=3)
        x_hat = self.forward(x)
        # x.shape should be (batch, genotype/snp=4, sequence length)

        # calculate metrics
        x = x.type(torch.LongTensor).to(self.device)
        loss = self.loss(x_hat, x)

        self.log("val_loss", loss)
        self.log("val_step/sec", time.time() - s)

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader
