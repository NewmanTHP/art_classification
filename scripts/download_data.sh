#!/usr/bin/env bash
# Pre-download and cache the WikiArt dataset from the Hugging Face Hub so that
# `src/train.py` / `src/evaluate.py` can load it without a network round-trip
# every run. Safe to skip -- `datasets.load_dataset` will cache on first use
# regardless.
set -euo pipefail

CACHE_DIR="${1:-data/cache}"

python - <<PY
from datasets import load_dataset

print("Downloading huggan/wikiart to ${CACHE_DIR} (this can take a while, ~25GB)...")
load_dataset("huggan/wikiart", split="train", cache_dir="${CACHE_DIR}")
print("Done.")
PY
