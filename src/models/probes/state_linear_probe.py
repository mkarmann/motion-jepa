"""Linear probe: embedding -> state via a single affine map."""

import torch
from torch import nn


class StateLinearProbe(nn.Module):
    """Linear (affine) state probe: emb_dim -> state_dim."""

    def __init__(self, state_dim, emb_dim=192):
        super().__init__()
        self.head = nn.Linear(emb_dim, state_dim)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        orig_shape = emb.shape
        x = emb.reshape(-1, orig_shape[-1])
        x = self.head(x)
        return x.view(*orig_shape[:-1], -1)

    def train_step(self, emb: torch.Tensor, states: torch.Tensor) -> dict:
        pred = self(emb)
        loss = (pred - states.float()).pow(2).mean()
        return {'loss': loss, 'pred': pred}
