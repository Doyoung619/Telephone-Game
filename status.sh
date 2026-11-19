#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
telephone_python="${TELEPHONE_PYTHON:-$PWD/.venv/bin/python}"
"$telephone_python" - <<'PY'
import json
from pathlib import Path
p=Path('runs/main/status.json')
if not p.exists():print('No run status yet');raise SystemExit(0)
s=json.loads(p.read_text())
print('Status:',s.get('status'),'| Phase:',s.get('phase'))
print('Progress:',s.get('phase_done',0),'/',s.get('phase_total',0))
eta=s.get('phase_remaining_seconds')
if eta is not None:print(f'Current-phase measured ETA: {eta/60:.1f} min')
print('Updated UTC:',s.get('updated_utc'))
p=Path('runs/main/selection/selected_method.json')
if p.exists():
 f=json.loads(p.read_text());print('Frozen winner:',f.get('algorithm'),f.get('model',{}).get('order'))
print('Measured stages:')
for k,v in s.get('timings',{}).items():print(f'  {k}: {v.get("wall_seconds",0)/60:.2f} min')
PY
tail -n 4 runs/main/runner.log 2>/dev/null || true
