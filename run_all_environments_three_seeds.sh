#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE="${1:-motionjepa}"
DEVICE="${2:-cuda}"

# 3 seeds × 3 envs, then multi-seed NMSE per env.
for dataset in golf dino pong; do
  for seed in 42 43 44; do
    echo "=== Training ${MODEL_TYPE}  dataset=${dataset}  seed=${seed}  device=${DEVICE} ==="
    bash run.sh "$dataset" "$MODEL_TYPE" "./runs/${dataset}_${MODEL_TYPE}_seed${seed}" "$seed" "$DEVICE"
  done
done

# Run evaluation again to obtain multi-seed NMSE per env.
for dataset in golf dino pong; do
  echo "=== Multi-seed eval  model=${MODEL_TYPE}  dataset=${dataset} ==="
  uv run python evaluate.py --device "$DEVICE" --run-dir \
    "./runs/${dataset}_${MODEL_TYPE}_seed42" \
    "./runs/${dataset}_${MODEL_TYPE}_seed43" \
    "./runs/${dataset}_${MODEL_TYPE}_seed44"
done
