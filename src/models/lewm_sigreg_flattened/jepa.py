"""LeWM JEPA with SIGReg over all frame embeddings flattened into one batch."""

import torch

from src.models.lewm.jepa import JEPA


class LeWMSigRegFlattened(JEPA):
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
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        b, t, d = emb.shape
        sigreg_loss = self.sigreg(emb.reshape(1, b * t, d))
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        return emb, {'loss': loss, 'pred_loss': pred_loss, 'sigreg_loss': sigreg_loss}
