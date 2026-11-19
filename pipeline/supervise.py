"""Detached supervisor with exclusive lock and bounded crash retries."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
run=root/'runs/main';run.mkdir(parents=True,exist_ok=True)
lock=(run/'supervisor.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    print('Another supervisor already owns this study',flush=True);raise SystemExit(0)
(run/'supervisor.pid').write_text(str(os.getpid()))
env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
for attempt in range(1,4):
    print(json.dumps({'event':'supervisor_attempt','attempt':attempt,'time':time.time()}),flush=True)
    child=subprocess.Popen([sys.executable,'-u',str(root/'pipeline/run_all.py'),'--run',str(run)],cwd=root,env=env)
    (run/'runner.pid').write_text(str(child.pid))
    code=child.wait()
    if code==0:
        print('Supervisor completed successfully',flush=True);raise SystemExit(0)
    print(json.dumps({'event':'runner_failed','exit_code':code,'attempt':attempt}),flush=True)
    if attempt<3:time.sleep(15)
status=run/'status.json'
obj=json.loads(status.read_text()) if status.exists() else {}
obj.update(status='failed',detail='Three attempts failed; see runner.log. start.sh resumes saved batches.')
tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(status)
raise SystemExit(1)
