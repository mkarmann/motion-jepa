"""Predict state from frozen embeddings via a shallow pre-activation residual MLP."""

import torch
import torch.nn.functional as F
from torch import nn


class PreActResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Pre-activation: BN -> ReLU -> Linear
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(F.relu(self.bn1(x)))
        h = self.fc2(F.relu(self.bn2(h)))
        return x + h  # No activation after the residual addition


class StateResidualMLPProbe(nn.Module):
    """Pre-activation ResNet-style probe: embedding -> state."""

    def __init__(self, state_dim, emb_dim=192, hidden_dim=128, depth=3):
        super().__init__()
        self.stem = nn.Linear(emb_dim, hidden_dim)
        self.blocks = nn.ModuleList([PreActResBlock(hidden_dim) for _ in range(depth)])
        self.post_norm = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Linear(hidden_dim, state_dim)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # Save original shape and flatten to 2D for BatchNorm1d
        orig_shape = emb.shape
        x = emb.view(-1, orig_shape[-1])
        
        # Network pass
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
            
        # Final pre-activation before the head
        x = self.head(F.relu(self.post_norm(x)))
        
        # Restore original leading dimensions (e.g., Batch, Time, Dim)
        return x.view(*orig_shape[:-1], -1)

    def train_step(self, emb: torch.Tensor, states: torch.Tensor) -> dict:
        pred = self(emb)
        loss = (pred - states.float()).pow(2).mean()
        return {"loss": loss, "pred": pred}