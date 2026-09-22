import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.registry import registry_create_dataset, registry_create_dataloader
from src.models.registry import registry_create_model
from src.utils import (
    PRE_TRAIN_DIR,
    RUN_CONFIG,
    LinearWarmupCosineAnnealingLR,
    load_run_config,
    run_config_dict,
    save_run_config,
)


def train(model_type='lewm',
          dataset_type='pong',
          embed_dim=None,
          history_size=3,
          sigreg_weight=None,
          diff_sigreg_weight=None,
          diff_pred_weight=None,
          action_pred_weight=None,
          img_size=112,
          episode_length=None,
          lr=5e-5,
          batch_size=128,
          epochs=20,
          dataset_size=500_000,
          output_dir='.',
          use_amp=True,
          compile=True,
          seed=42,
          device='cuda'):
    cfg = run_config_dict(**locals())
    device = torch.device(device)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    if episode_length is None:
        episode_length = history_size + 1
    print(f'Using history_size={history_size} -> training episode_length={episode_length}')

    out = Path(output_dir) / PRE_TRAIN_DIR
    out.mkdir(parents=True, exist_ok=True)
    meta = registry_create_dataset(dataset_type=dataset_type, img_size=img_size, episode_length=episode_length, size=1, seed=seed)
    cfg.update(episode_length=episode_length, action_dim=meta.action_dim, state_dim=meta.state_dim)
    save_run_config(out / RUN_CONFIG, cfg)

    print('Initializing model...')
    model_kw = dict(
        model_type=model_type,
        embed_dim=embed_dim,
        img_size=img_size,
        action_dim=meta.action_dim,
        state_dim=meta.state_dim,
        history_size=history_size,
    )
    if sigreg_weight is not None:
        model_kw['sigreg_weight'] = sigreg_weight
    if diff_sigreg_weight is not None:
        model_kw['diff_sigreg_weight'] = diff_sigreg_weight
    if diff_pred_weight is not None:
        model_kw['diff_pred_weight'] = diff_pred_weight
    if action_pred_weight is not None:
        model_kw['action_pred_weight'] = action_pred_weight
    model = registry_create_model(**model_kw).to(device)

    if compile:
        print('Compiling model...')
        model.train_step = torch.compile(model.train_step)

    print('Configuring optimizer...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(0.01 * total_steps))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=warmup_steps, max_steps=total_steps)
    print(f'Training schedule: {warmup_steps} warmup steps, {total_steps} total steps ({epochs} epochs)')

    print('Starting training...')
    for epoch in range(epochs):
        print('Initializing this epochs training dataset...')
        train_dataset = registry_create_dataset(dataset_type=dataset_type, img_size=img_size, episode_length=episode_length, size=dataset_size, seed=seed + epoch + 1)
        train_dataloader = registry_create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, device=device)

        p_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in p_bar:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                _, losses = model.train_step(batch)
                log = dict(losses)
            losses['loss'].backward()
            optimizer.step()
            scheduler.step()
            log['lr'] = torch.tensor(scheduler.get_last_lr()[0])
            p_bar.set_postfix_str(' - '.join(f'{k}: {v.item():.3f}' for k, v in log.items() if hasattr(v, 'item')))

        torch.save(model.state_dict(), out / 'model.pt')

    print('Saving model...')
    torch.save(model.state_dict(), out / 'model.pt')
    print('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='YAML config; CLI flags override it')
    parser.add_argument('--model-type', default='lewm')
    parser.add_argument('--dataset-type', default='pong')
    parser.add_argument('--embed-dim', type=int, default=None)
    parser.add_argument('--history-size', type=int, default=3)
    parser.add_argument('--sigreg-weight', type=float, default=None)
    parser.add_argument('--diff-sigreg-weight', type=float, default=None)
    parser.add_argument('--diff-pred-weight', type=float, default=None)
    parser.add_argument('--action-pred-weight', type=float, default=None)
    parser.add_argument('--img-size', type=int, default=112)
    parser.add_argument('--episode-length', type=int, default=None, help='Training episode length (default: history_size + 1)')
    parser.add_argument('--output-dir', default='./runs/default')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dataset-size', type=int, default=500_000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use-amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    kwargs = vars(args)
    config_path = kwargs.pop('config')
    if config_path:
        kwargs = {**kwargs, **load_run_config(config_path)}
        argv = sys.argv[1:]
        for action in parser._actions:
            if action.dest in ('help', 'config'):
                continue
            if any(a == opt or a.startswith(opt + '=') for a in argv for opt in action.option_strings):
                kwargs[action.dest] = getattr(args, action.dest)
    train(**kwargs)


if __name__ == '__main__':
    main()
