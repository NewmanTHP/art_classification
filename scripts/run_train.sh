#!/usr/bin/env bash
# Example end-to-end run: train the ViT, then evaluate it against the
# ResNet-50 baseline on the held-out test split.
set -euo pipefail

CONFIG="${1:-configs/config.yaml}"

python -m src.train --config "$CONFIG"
python -m src.evaluate --config "$CONFIG" --checkpoint checkpoints/vit_best.pt --baseline
