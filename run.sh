#!/usr/bin/env bash
set -euo pipefail

DATASET_TYPE="${1:-pong}"
MODEL_TYPE="${2:-motionjepa}"
OUTPUT_DIR="${3:-./runs/${DATASET_TYPE}_${MODEL_TYPE}}"
SEED="${4:-0}"
DEVICE="${5:-cuda}"

uv run python train.py --model-type "$MODEL_TYPE" --dataset-type "$DATASET_TYPE" --output-dir "$OUTPUT_DIR" --seed "$SEED" --device "$DEVICE"
uv run python train_probes.py --run-dir "$OUTPUT_DIR" --device "$DEVICE"
uv run python evaluate.py --run-dir "$OUTPUT_DIR" --device "$DEVICE"
