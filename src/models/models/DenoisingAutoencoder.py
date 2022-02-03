import torch.nn as nn
import torch


class DenoisingAutoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, dropout_p=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.encoder(x)
        return self.decoder(x, self.encoder.forward_shapes)
