"""SMWM: sigreg-less world model.

Same single-encoder + AR-predictor setup as LeWM, but instead of a SIGReg
variance/isotropy regularizer it prevents representation collapse with an
inverse-dynamics head: an action decoder must recover the action taken between
two consecutive frame embeddings. If the encoder collapsed, actions could not be
recovered, so this term keeps the representation informative.
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.models.lewm.jepa import DEFAULT_EMBED_DIM, JEPA


class SMWM(JEPA):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        *,
        embed_dim=DEFAULT_EMBED_DIM,
        history_size=3,
        num_preds=1,
        action_dim=2,
        action_pred_weight=1.0,
    ):
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj or nn.Identity(),
            history_size=history_size,
            num_preds=num_preds,
            sigreg_weight=0.0,
            embed_dim=embed_dim,
        )
        self.action_pred_weight = action_pred_weight

        # Architecture from: https://arxiv.org/pdf/2606.20104 (Section A.2)
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def train_step(self, batch):
        info = self._batch_to_info(batch)
        info['action'] = torch.nan_to_num(info['action'], nan=0.0)
        output = self.encode(info)
        emb = output['emb']
        act_emb = output['act_emb']

        ctx_emb = emb[:, :self.history_size]
        ctx_act = act_emb[:, :self.history_size]
        tgt_emb = emb[:, self.num_preds:]
        pred_emb = self.predict(ctx_emb, ctx_act)
        pred_loss = F.mse_loss(pred_emb, tgt_emb)

        # Inverse dynamics: recover the action between consecutive embeddings.
        pair = torch.cat([emb[:, :-1], emb[:, 1:]], dim=-1)
        pred_act = self.action_decoder(pair)
        act_loss = F.mse_loss(pred_act, info['action'][:, :-1].float())

        loss = pred_loss + self.action_pred_weight * act_loss
        return emb, {
            'loss': loss,
            'pred': pred_loss,
            'act': act_loss,
        }
