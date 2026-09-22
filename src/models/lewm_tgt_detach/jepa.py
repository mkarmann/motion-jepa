"""LeWM JEPA with stop-grad on prediction targets (``tgt_emb.detach()``)."""

import torch

from src.models.lewm.jepa import JEPA


class LeWMTgtDetach(JEPA):
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
        pred_loss = (pred_emb - tgt_emb.detach()).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        return emb, {'loss': loss, 'pred_loss': pred_loss, 'sigreg_loss': sigreg_loss}
