#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
torch_index=https://download.pytorch.org/whl/cu124
if [[ "${1:-}" == --cpu ]]; then
  torch_index=https://download.pytorch.org/whl/cpu
elif [[ -n "${1:-}" ]]; then
  echo 'Usage: bash scripts/setup.sh [--cpu]' >&2
  exit 2
fi
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch==2.6.0' --index-url "$torch_index"
.venv/bin/python -m pip install -r requirements.txt
echo 'Environment ready. Full editing experiments require a CUDA GPU.'
