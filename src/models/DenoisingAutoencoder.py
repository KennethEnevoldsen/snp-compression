from typing import Optional
import torch.nn as nn
import torch


class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        dropout_p: float,
        fc_layer_size: Optional[int],
        activation="relu",
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder = encoder
        self.decoder = decoder

        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError("Only relu is currently implemented")

        self.fc_layer_size = fc_layer_size
        self._fc = None

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        """Dyanimic fully connected middle layer"""

        if self.fc_layer_size and self._fc is None:
            self._fc = nn.Sequential(
                nn.Linear(x.shape[-2], self.fc_layer_size),
                self.activation,
                nn.Linear(self.fc_layer_size, x.shape[-2]),
                self.activation,
            )
        if self._fc:
            x = x.squeeze(-1).squeeze(1)
            x = self._fc(x)
            x = x.unsqueeze(1).unsqueeze(-1)
            return x
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.encoder(x)

        if self.fc_layer_size:
            x = self.fc(x=x)

        return self.decoder(x, self.encoder.forward_shapes)
