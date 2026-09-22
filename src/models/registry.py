import inspect
from torch import nn

from src.models.probes.state_linear_probe import StateLinearProbe
from src.models.probes.state_residual_mlp_probe import StateResidualMLPProbe
from src.models.lewm.jepa import DEFAULT_EMBED_DIM, JEPA
from src.models.lewm.module import ARPredictor, Embedder, MLP
from src.models.lewm.vit import vit_hf
from src.models.lewm_sigreg_flattened.jepa import LeWMSigRegFlattened
from src.models.lewm_sigreg_time.jepa import LeWMSigRegTime
from src.models.lewm_tgt_detach.jepa import LeWMTgtDetach
from src.models.motionjepa.jepa import MotionJEPA
from src.models.smwm.jepa import SMWM

LEWM_AR_PREDICTOR = dict(depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0)


def _vit_encoder(img_size):
    return vit_hf('tiny', patch_size=14, image_size=img_size, pretrained=False, use_mask_token=False)


def _lewm_projector(hidden_dim, embed_dim):
    return MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d)


def _filter_kwargs(model_cls, kwargs):
    """Pass only constructor args the model accepts; drop Nones so class defaults apply."""
    params = inspect.signature(model_cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        allowed = None
    else:
        allowed = set(params) - {'self'}
    out = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if allowed is not None and key not in allowed:
            continue
        out[key] = value
    return out


def _create_model(model_cls, embed_dim, img_size, action_dim, history_size, **kwargs):
    encoder = _vit_encoder(img_size)
    hidden_dim = encoder.config.hidden_size
    pred_kw = {**LEWM_AR_PREDICTOR, **(kwargs.pop('predictor', None) or {})}
    model_kw = dict(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **pred_kw,
        ),
        action_encoder=Embedder(input_dim=action_dim, emb_dim=embed_dim),
        projector=_lewm_projector(hidden_dim, embed_dim),
        pred_proj=_lewm_projector(hidden_dim, embed_dim),
        embed_dim=embed_dim,
        history_size=history_size,
        num_preds=1,
        action_dim=action_dim,
    )
    if model_cls is MotionJEPA:
        diff_encoder = _vit_encoder(img_size)
        model_kw['diff_encoder'] = diff_encoder
        model_kw['diff_projector'] = _lewm_projector(diff_encoder.config.hidden_size, embed_dim)
    model_kw.update(kwargs)
    return model_cls(**_filter_kwargs(model_cls, model_kw))


def registry_create_model(model_type='lewm', embed_dim=None, img_size=224, action_dim=2, history_size=3, state_dim=2, **kwargs):
    d = embed_dim if embed_dim is not None else DEFAULT_EMBED_DIM
    models = {
        'lewm': JEPA,
        'lewm-sigreg-time': LeWMSigRegTime,
        'lewm-time-sigreg': LeWMSigRegTime,
        'lewm-sigreg-flattened': LeWMSigRegFlattened,
        'lewm-tgt-detach': LeWMTgtDetach,
        'motionjepa': MotionJEPA,
        'smwm': SMWM,
    }
    model_cls = models.get(model_type)
    if model_cls is None:
        raise ValueError(
            f'Unknown model type: {model_type!r}. '
            f'Options: lewm, lewm-sigreg-time, lewm-sigreg-flattened, lewm-tgt-detach, motionjepa, smwm.'
        )
    return _create_model(model_cls, d, img_size, action_dim, history_size, **kwargs)


def registry_create_state_residual_mlp_probe(state_dim=2, emb_dim=DEFAULT_EMBED_DIM):
    return StateResidualMLPProbe(state_dim=state_dim, emb_dim=emb_dim)


def registry_create_state_linear_probe(state_dim=2, emb_dim=DEFAULT_EMBED_DIM):
    return StateLinearProbe(state_dim=state_dim, emb_dim=emb_dim)
