import math
from pathlib import Path

import yaml
import torch
from torch.optim.lr_scheduler import LRScheduler

from src.models.registry import registry_create_model

PRE_TRAIN_DIR = 'pre_train'
RUN_CONFIG = 'config.yaml'
OFFLINE_PROBES_DIR = 'offline_probes'
EVALUATION_DIR = 'evaluation'


def run_config_dict(**kwargs) -> dict:
    """Build a YAML-serializable config dict from keyword arguments."""
    config = {}
    for key, value in kwargs.items():
        if isinstance(value, Path):
            value = str(value)
        config[key] = value
    return config


def save_run_config(path, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=True, default_flow_style=False)


def load_run_config(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Run config not found: {path}')
    with open(path) as f:
        return yaml.safe_load(f) or {}


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """Linear warmup from ``warmup_start_lr`` to base LR, then cosine decay to ``eta_min``."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def _compute_lr(self, base_lr: float, step: int) -> float:
        if step < self.warmup_steps:
            return self.warmup_start_lr + (base_lr - self.warmup_start_lr) * step / self.warmup_steps
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2

    def get_lr(self):
        return [self._compute_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]


def load_pretrained_model(checkpoint_dir, weights_name='model.pt', config_name=RUN_CONFIG):
    """Build model from ``config.yaml`` in the pre_train run dir and load weights (CPU)."""
    checkpoint_dir = Path(checkpoint_dir)
    config = load_run_config(checkpoint_dir / config_name)
    model_kw = dict(
        model_type=config['model_type'],
        embed_dim=config.get('embed_dim'),
        img_size=config['img_size'],
        action_dim=config.get('action_dim', 2),
        history_size=config.get('history_size', 3),
        state_dim=config.get('state_dim', 2),
    )
    for key in ('sigreg_weight', 'diff_sigreg_weight', 'diff_pred_weight', 'action_pred_weight'):
        if config.get(key) is not None:
            model_kw[key] = config[key]
    model = registry_create_model(**model_kw)
    weights_path = checkpoint_dir / weights_name
    if not weights_path.is_file():
        raise FileNotFoundError(f'Weights not found: {weights_path}')
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    return model, config


@torch.no_grad()
def encoder_and_rollout_embeddings(model, images, actions):
    """History encoder latents and open-loop predicted latents.

    Encodes only the first ``history_size`` frames. Returns
    ``(enc_emb, roll_emb)`` with shapes ``(B, hs, D)`` and ``(B, T-hs, D)``.
    """
    actions = actions.float()
    hs = getattr(model, 'history_size', 1)
    t_total = images.size(1)
    enc = model.encode_frames(images[:, :hs].float())
    act_emb = model.action_encoder(actions)
    steps = [enc[:, i] for i in range(min(hs, t_total))]
    for t in range(hs, t_total):
        start = len(steps) - hs
        ctx = torch.stack(steps[start:], dim=1)
        ctx_act = act_emb[:, start : start + hs]
        pred = model.predict(ctx, ctx_act)[:, -1]
        steps.append(pred)
    if t_total <= hs:
        roll = enc[:, :0]
    else:
        roll = torch.stack(steps[hs:], dim=1)
    return enc, roll


def print_results(result, title='Offline probe NMSE'):
    rows = result['rows']
    columns = list(rows[0].keys())

    def fmt(val):
        if isinstance(val, (tuple, list)):
            return f'{val[0]:.3f}±{val[1]:.3f}'
        return f'{val:.3f}'

    n = result.get('n_runs', 1)
    extra = f'  (n={n}, mean±std)' if n > 1 else ''
    print(f'{result["dataset_name"]}  history={result["history_size"]}  rollout={result["rollout_steps"]}  NMSE↓{extra}')
    print()
    widths = []
    for c in columns:
        vals = [r[c] if c == 'probe' else fmt(r[c]) for r in rows]
        widths.append(max(len(c), max(len(v) for v in vals)))
    print(title)
    print('  '.join(c.ljust(w) for c, w in zip(columns, widths)))
    print('  '.join('-' * w for w in widths))
    for row in rows:
        vals = [row[c] if c == 'probe' else fmt(row[c]) for c in columns]
        print('  '.join(v.ljust(w) for v, w in zip(vals, widths)))
    print()
