"""JEPA Implementation"""

import torch
from einops import rearrange
from torch import nn

from src.models.lewm.module import SIGReg

DEFAULT_EMBED_DIM = 192


class JEPA(nn.Module):

    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None, *,
                 embed_dim=DEFAULT_EMBED_DIM, history_size=3, num_preds=1, sigreg_weight=0.09, sigreg=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.history_size = history_size
        self.num_preds = num_preds
        self.sigreg_weight = sigreg_weight
        self.sigreg = sigreg if sigreg is not None else SIGReg()

    def encode_frames(self, images):
        """Encode each frame independently (no action conditioning).

        Args:
            images: (B, T, C, H, W) RGB float in [0, 1]

        Returns:
            (B, T, D) latent embeddings
        """
        return self.encode({'pixels': images.float()})['emb']

    def encode(self, info):
        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def _batch_to_info(self, batch):
        if "pixels" in batch:
            pixels = batch["pixels"]
        elif "images" in batch:
            pixels = batch["images"]
        else:
            raise KeyError("batch must contain 'pixels' or 'images'")
        if "action" in batch:
            action = batch["action"]
        elif "actions" in batch:
            action = batch["actions"]
        else:
            raise KeyError("batch must contain 'action' or 'actions'")
        return {"pixels": pixels, "action": action}

    def train_step(self, batch):
        info = self._batch_to_info(batch)
        info["action"] = torch.nan_to_num(info["action"], nan=0.0)
        output = self.encode(info)
        emb = output["emb"]
        act_emb = output["act_emb"]
        ctx_emb = emb[:, :self.history_size]
        ctx_act = act_emb[:, :self.history_size]
        tgt_emb = emb[:, self.num_preds:]
        pred_emb = self.predict(ctx_emb, ctx_act)
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        return emb, {"loss": loss, "pred_loss": pred_loss, "sigreg_loss": sigreg_loss}
