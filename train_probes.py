import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.registry import registry_create_dataset, registry_create_dataloader
from src.models.registry import (
    registry_create_state_linear_probe,
    registry_create_state_residual_mlp_probe,
)
from src.utils import (
    OFFLINE_PROBES_DIR,
    PRE_TRAIN_DIR,
    RUN_CONFIG,
    encoder_and_rollout_embeddings,
    load_pretrained_model,
    run_config_dict,
    save_run_config,
)


def train_probes(run_dir,
                 rollout_steps=5,
                 device=None,
                 use_amp=True,
                 compile=True,
                 dataset_size=1_000_000):
    run_dir = Path(run_dir)
    pre_train = run_dir / PRE_TRAIN_DIR if (run_dir / PRE_TRAIN_DIR / 'model.pt').is_file() else run_dir
    run_root = pre_train.parent if pre_train.name == PRE_TRAIN_DIR else pre_train
    out = run_root / OFFLINE_PROBES_DIR
    if out.resolve() == pre_train.resolve():
        raise RuntimeError(f'Refusing to write probes into pre_train dir: {out}')
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading frozen model from {pre_train}...')
    model, cfg = load_pretrained_model(pre_train)
    device = torch.device(device or cfg.get('device', 'cuda'))
    seed = int(cfg.get('seed', 42))
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    dataset_type = cfg['dataset_type']
    img_size = int(cfg['img_size'])
    history_size = int(cfg.get('history_size', 3))
    episode_length = history_size + int(rollout_steps)
    batch_size = int(cfg.get('batch_size', 128))
    dataset_size = int(dataset_size)
    state_lr = float(cfg.get('state_lr', 3e-3))
    state_dim = int(cfg['state_dim'])
    roll = f'rollout_{int(rollout_steps)}'
    probe_names = (
        'state_mlp_encoder',
        f'state_mlp_{roll}',
        'state_linear_encoder',
        f'state_linear_{roll}',
    )
    print(f'history_size={history_size}, rollout_steps={rollout_steps} -> episode_length={episode_length}')
    print(f'Probe weights -> {out}')
    print(f'Train seed={seed + 1}, probe data train={dataset_size:,}')

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print('Initializing probes...')
    probes = {}
    for name in probe_names:
        if '_mlp_' in name:
            mod = registry_create_state_residual_mlp_probe(state_dim=state_dim, emb_dim=model.embed_dim)
        else:
            mod = registry_create_state_linear_probe(state_dim=state_dim, emb_dim=model.embed_dim)
        probes[name] = mod.to(device)

    if compile:
        print('Compiling probes...')
        for mod in probes.values():
            mod.train_step = torch.compile(mod.train_step)

    print('Configuring optimizer...')
    state_optimizer = torch.optim.AdamW([p for m in probes.values() for p in m.parameters()], lr=state_lr, weight_decay=1e-3)
    print(f'Constant probe LR (no schedule): state_lr={state_lr:g}, {max(1, dataset_size // batch_size)} steps')

    save_run_config(out / RUN_CONFIG, run_config_dict(
        **{k: cfg.get(k) for k in (
            'model_type', 'dataset_type', 'img_size', 'history_size', 'action_dim', 'state_dim',
            'batch_size', 'state_lr', 'seed',
        )},
        source_pre_train=str(pre_train),
        dataset_size=dataset_size,
        rollout_steps=rollout_steps,
        episode_length=episode_length,
        device=str(device),
        use_amp=use_amp,
        compile=compile,
        frozen_model=True,
        probes=list(probe_names),
    ))

    print('Starting training...')
    train_dataset = registry_create_dataset(dataset_type=dataset_type, img_size=img_size, episode_length=episode_length, size=dataset_size, seed=seed + 1)
    train_loader = registry_create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, device=device)
    p_bar = tqdm(train_loader, desc='Train')
    for batch in p_bar:
        state_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                enc_emb, roll_emb = encoder_and_rollout_embeddings(model, batch['images'], batch['actions'])
            losses = {}
            for name, mod in probes.items():
                emb = enc_emb if name.endswith('_encoder') else roll_emb
                tgt = batch['states'][:, :history_size] if name.endswith('_encoder') else batch['states'][:, history_size:]
                losses[name] = mod.train_step(emb, tgt)['loss']
        sum(losses.values()).backward()
        state_optimizer.step()
        p_bar.set_postfix_str(' '.join(f'{k.split("_", 1)[1]}={float(v.item() if hasattr(v, "item") else v):.3f}' for k, v in losses.items()))

    print('Saving probes...')
    for name, mod in probes.items():
        torch.save(mod.state_dict(), out / f'{name}_probe.pt')
    print('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--rollout-steps', type=int, default=5)
    parser.add_argument('--dataset-size', type=int, default=1_000_000)
    parser.add_argument('--device', default=None)
    parser.add_argument('--use-amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    train_probes(**vars(args))


if __name__ == '__main__':
    main()
