"""Exclusive GPU worker lease shared with the manuscript pipeline."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import psutil
from .io import timestamp

@contextmanager
def gpu_lease(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for _ in range(2):
        try:
            with path.open('x') as handle:json.dump({'pid':os.getpid(),'created':timestamp(),'owner':'independent-evaluation'},handle)
            break
        except FileExistsError:
            value=json.loads(path.read_text());pid=value.get('pid')
            if not isinstance(pid,int) or psutil.pid_exists(pid):raise RuntimeError('GPU lease is held: '+str(path))
            path.unlink()
    else:raise RuntimeError('Cannot acquire GPU lease')
    try:yield
    finally:
        if path.exists() and json.loads(path.read_text()).get('pid')==os.getpid():path.unlink()
