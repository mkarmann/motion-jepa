import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.dino import DinoDataset
from src.data.golf import GolfDataset
from src.data.pong import PongDataset
from src.data.registry import registry_create_dataset, registry_create_dataloader
from src.models.registry import (
    registry_create_state_linear_probe,
    registry_create_state_residual_mlp_probe,
)
from src.utils import (
    EVALUATION_DIR,
    OFFLINE_PROBES_DIR,
    PRE_TRAIN_DIR,
    RUN_CONFIG,
    encoder_and_rollout_embeddings,
    load_pretrained_model,
    load_run_config,
    print_results,
)

STATE_GROUPS = {
    'pong': PongDataset.state_groups,
    'dino': DinoDataset.state_groups,
    'golf': GolfDataset.state_groups,
}


@torch.no_grad()
def evaluate(run_dir, device=None, use_amp=True, eval_dataset_size=10_000):
    run_dir = Path(run_dir)
    probes_dir = run_dir / OFFLINE_PROBES_DIR
    out = run_dir / EVALUATION_DIR
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_run_config(probes_dir / RUN_CONFIG)
    device = torch.device(device or cfg.get('device', 'cuda'))
    seed = int(cfg['seed'])
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    dataset_type = cfg['dataset_type']
    groups = STATE_GROUPS[dataset_type]
    history_size = int(cfg['history_size'])
    rollout_steps = int(cfg['rollout_steps'])
    episode_length = int(cfg['episode_length'])
    state_dim = int(cfg['state_dim'])
    probe_names = list(cfg['probes'])

    print('Loading model and probes...')
    model, _ = load_pretrained_model(run_dir / PRE_TRAIN_DIR)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    probes = {}
    for name in probe_names:
        if '_mlp_' in name:
            mod = registry_create_state_residual_mlp_probe(state_dim=state_dim, emb_dim=model.embed_dim)
        else:
            mod = registry_create_state_linear_probe(state_dim=state_dim, emb_dim=model.embed_dim)
        mod.load_state_dict(torch.load(probes_dir / f'{name}_probe.pt', map_location='cpu', weights_only=True))
        probes[name] = mod.to(device).eval()

    print('Initializing evaluation dataset...')
    eval_loader = registry_create_dataloader(
        registry_create_dataset(
            dataset_type=dataset_type, img_size=int(cfg['img_size']), episode_length=episode_length,
            size=eval_dataset_size, seed=seed,
        ),
        batch_size=int(cfg['batch_size']), shuffle=False, device=device,
    )

    print('Running evaluation...')
    actions, gt, preds = [], [], {name: [] for name in probes}
    for batch in tqdm(eval_loader, desc='Eval'):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            enc_emb, roll_emb = encoder_and_rollout_embeddings(model, batch['images'], batch['actions'])
            for name, mod in probes.items():
                emb = enc_emb if name.endswith('_encoder') else roll_emb
                preds[name].append(mod(emb).float().cpu().numpy())
        actions.append(batch['actions'].float().cpu().numpy())
        gt.append(batch['states'].float().cpu().numpy())

    actions = np.concatenate(actions)
    gt = np.concatenate(gt)
    preds = {name: np.concatenate(chunks) for name, chunks in preds.items()}
    np.save(out / 'actions.npy', actions)
    np.save(out / 'state_gt.npy', gt)
    for name, arr in preds.items():
        np.save(out / f'{name}.npy', arr)

    rows = []
    for name, pred in preds.items():
        gt_slice = gt[:, :history_size] if name.endswith('_encoder') else gt[:, history_size:history_size + rollout_steps]
        mean_pred = np.broadcast_to(gt_slice.mean(axis=(0, 1), keepdims=True), gt_slice.shape)
        row = {'probe': name}
        for key, dims in (('avg', np.concatenate([list(d) for _, d in groups])), *groups):
            idx = np.asarray(dims, dtype=int)
            mse = float(((pred[..., idx] - gt_slice[..., idx]) ** 2).mean())
            baseline = float(((mean_pred[..., idx] - gt_slice[..., idx]) ** 2).mean())
            row[key] = mse / baseline if baseline > 0 else float('nan')
        rows.append(row)

    result = {
        'dataset_name': dataset_type,
        'history_size': history_size,
        'rollout_steps': rollout_steps,
        'rows': rows,
    }
    with open(out / 'metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved {out}')
    return result


def evaluate_multiple(run_dirs, device=None, use_amp=True, eval_dataset_size=10_000):
    results = [evaluate(p, device=device, use_amp=use_amp, eval_dataset_size=eval_dataset_size) for p in run_dirs]
    keys = [k for k in results[0]['rows'][0] if k != 'probe']
    rows = []
    for i in range(len(results[0]['rows'])):
        row = {'probe': results[0]['rows'][i]['probe']}
        for key in keys:
            vals = [r['rows'][i][key] for r in results]
            row[key] = (float(np.mean(vals)), float(np.std(vals)))
        rows.append(row)
    return {
        'dataset_name': results[0]['dataset_name'],
        'history_size': results[0]['history_size'],
        'rollout_steps': results[0]['rollout_steps'],
        'n_runs': len(results),
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--eval-dataset-size', type=int, default=10_000)
    parser.add_argument('--device', default=None)
    parser.add_argument('--use-amp', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if len(args.run_dir) == 1:
        print_results(evaluate(args.run_dir[0], device=args.device, use_amp=args.use_amp, eval_dataset_size=args.eval_dataset_size))
    else:
        print_results(evaluate_multiple(args.run_dir, device=args.device, use_amp=args.use_amp, eval_dataset_size=args.eval_dataset_size))


if __name__ == '__main__':
    main()
