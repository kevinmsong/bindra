from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import torch


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):value.update(block)
    return value.hexdigest()


def timestamp():return datetime.now(timezone.utc).isoformat()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)


def freeze(path,value):
    path=Path(path)
    value=json.loads(json.dumps(value,allow_nan=False))
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('Frozen identity changed: '+str(path))
    else:write_json(path,value)
    return value


def atomic_save(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);os.replace(temp,path)
