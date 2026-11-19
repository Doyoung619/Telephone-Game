#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
telephone_python="${TELEPHONE_PYTHON:-$PWD/.venv/bin/python}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --skip-external-refit ) ]]; then
  echo 'Usage: bash start.sh [--skip-external-refit]' >&2
  exit 2
fi
if [[ ! -x "$telephone_python" ]]; then
  echo 'Create the environment with: bash scripts/setup.sh' >&2
  exit 2
fi
"$telephone_python" pipeline/prepare.py check
"$telephone_python" - "$@" <<'PY'
import json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();run=root/'runs/main';run.mkdir(parents=True,exist_ok=True)
if (run/'completion_manifest.json').exists():
    print('Completed run already exists:',run/'completion_manifest.json');raise SystemExit(0)
def spawn(script,logname,extra=()):
    log=(run/logname).open('a',buffering=1)
    return subprocess.Popen([sys.executable,'-u',str(root/'pipeline'/script),*extra],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
if not (run/'external/main_status_before_refit.json').exists():
    p=spawn('supervise.py','runner.log');print('Detached supervisor PID:',p.pid)
pidfile=run/'finalizer.pid'
alive=False
if pidfile.exists():
    proc=Path('/proc')/pidfile.read_text().strip()/'cmdline'
    alive=proc.exists() and b'pipeline/finalize.py' in proc.read_bytes()
if not alive:
    p=spawn('finalize.py','finalizer.log',['--wait',*sys.argv[1:]]);pidfile.write_text(str(p.pid))
    print('Detached final reporting/refit PID:',p.pid)
print('Status: ./status.sh; log:',run/'runner.log')
PY
