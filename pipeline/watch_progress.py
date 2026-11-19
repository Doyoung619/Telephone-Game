"""Refresh a live review figure without occupying or interrupting the editor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
run=root/'runs/main'
for _ in range(1440):
    subprocess.run([sys.executable,str(root/'pipeline/progress.py'),'--run',str(run),
                    '--review',str(root/'artifacts/Review')],cwd=root,timeout=45,check=False)
    status=run/'status.json'
    if status.exists() and json.loads(status.read_text()).get('status') in ('complete','failed'):
        break
    time.sleep(30)
