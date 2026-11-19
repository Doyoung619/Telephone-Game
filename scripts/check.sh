#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
telephone_python="${TELEPHONE_PYTHON:-$PWD/.venv/bin/python}"
"$telephone_python" -m unittest discover -s tests -v
"$telephone_python" reference/verify_schedulers.py
"$telephone_python" pipeline/run_all.py --mock-check
"$telephone_python" -m compileall -q pipeline reference tests
bash -n start.sh status.sh scripts/setup.sh scripts/check.sh
