"""MotionJEPA: LeWM JEPA with separate encoders for frames and frame differences.

Supports a configurable ``history_size`` via causal teacher-forcing (same window
scheme as ``src/models/lewm/jepa.py``)."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.models.lewm.jepa import DEFAULT_EMBED_DIM, JEPA


class MotionJEPA(JEPA):
    def __init__(
        self,
        encoder,
        diff_encoder,
        predictor,
        action_encoder,
        projector=None,
        diff_projector=None,
        pred_proj=None,
        *,
        embed_dim=DEFAULT_EMBED_DIM,
        history_size=3,
        num_preds=1,
        sigreg_weight=0.25,
        diff_sigreg_weight=2.0,
        diff_pred_weight=0.5,
        sigreg=None,
    ):
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj or nn.Identity(),
            history_size=history_size,
            num_preds=num_preds,
            sigreg_weight=sigreg_weight,
            sigreg=sigreg,
            embed_dim=embed_dim,
        )
        self.diff_encoder = diff_encoder
        self.diff_projector = diff_projector or nn.Identity()
        self.diff_sigreg_weight = diff_sigreg_weight
        self.diff_pred_weight = diff_pred_weight

        hidden_dim = 512
        depth = 3
        layers = []
        dim = embed_dim * 2
        for _ in range(max(depth - 1, 0)):
            layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, embed_dim))
        self.inv_pred = nn.Sequential(*layers)

    def encode(self, info):
        output = super().encode(info)
        pixels = info['pixels'].float()
        b, t = pixels.shape[:2]
        diff_pixels = pixels[:, 1:] - pixels[:, :-1]
        diff_pixels = rearrange(diff_pixels, 'b t ... -> (b t) ...')
        diff_out = self.diff_encoder(diff_pixels, interpolate_pos_encoding=True)
        emb_diff = self.diff_projector(diff_out.last_hidden_state[:, 0])
        output['emb_diff'] = rearrange(emb_diff, '(b t) d -> b t d', b=b)
        return output

    def train_step(self, batch):
        info = self._batch_to_info(batch)
        info['action'] = torch.nan_to_num(info['action'], nan=0.0)
        output = self.encode(info)
        emb = output['emb']
        act_emb = output['act_emb']
        emb_diff = output['emb_diff']

        ctx_emb = emb[:, :self.history_size]
        ctx_act = act_emb[:, :self.history_size]
        tgt_emb = emb[:, self.num_preds:]
        pred_emb = self.predict(ctx_emb, ctx_act)
        pred_loss = F.mse_loss(pred_emb, tgt_emb)

        pred_emb_diff = self.inv_pred(torch.cat([emb[:, :-1], emb[:, 1:]], dim=-1))
        inv_pred_loss = F.mse_loss(pred_emb_diff, emb_diff)

        static_sigreg_loss = self.sigreg(emb.transpose(0, 1))
        diff_sigreg_loss = self.sigreg(emb_diff.transpose(0, 1))
        loss = (
            pred_loss
            + self.diff_pred_weight * inv_pred_loss
            + self.sigreg_weight * static_sigreg_loss
            + self.diff_sigreg_weight * diff_sigreg_loss
        )
        return emb, {
            'loss': loss,
            'pred': pred_loss,
            'inv': inv_pred_loss,
            'static_sigreg': static_sigreg_loss,
            'diff_sigreg': diff_sigreg_loss,
        }
