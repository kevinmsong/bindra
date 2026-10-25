"""Resumable validation-only fitting over frozen tokens or pinned graph inputs."""
from dataclasses import asdict
from pathlib import Path
import json
import math
import random
import time

import numpy as np
import torch
from scipy.stats import kendalltau,pearsonr,spearmanr

from .io import sha256,freeze,write_json,atomic_save,timestamp
from .models import FollowupModel,FollowupModelConfig


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
    'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda']:torch.cuda.set_rng_state_all(value['cuda'])


def ci(y,p):
    y,p=np.asarray(y),np.asarray(p)
    if y.shape!=p.shape or y.ndim!=1 or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Invalid metric arrays')
    total=len(y)*(len(y)-1)/2
    def untied(x):
        n=np.unique(x,return_counts=True)[1];return total-np.sum(n*(n-1)/2)
    a,b=untied(y),untied(p)
    if a==0:return None
    return .5 if b==0 else float(.5+.5*kendalltau(y,p).statistic*np.sqrt(b/a))


def metrics(y,p,scale):
    y,p=np.asarray(y,float),np.asarray(p,float)
    if scale not in ('pKd','KIBA_score'):raise ValueError('Explicit regression scale required')
    if y.shape!=p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Invalid predictions')
    return {'n':len(y),'scale':scale,'mse':float(np.mean((y-p)**2)),'rmse':float(np.sqrt(np.mean((y-p)**2))),
        'mae':float(np.mean(np.abs(y-p))),'concordance_index':ci(y,p),
        'pearson':float(pearsonr(y,p).statistic) if np.std(y) and np.std(p) else None,
        'spearman':float(spearmanr(y,p).statistic) if np.std(y) and np.std(p) else None}


def batches(frame,size,seed=None):
    ids=np.arange(len(frame));generator=np.random.default_rng(seed)
    lengths=frame.sequence.str.len().to_numpy();ligand_lengths=frame.smiles.str.len().to_numpy()
    if seed is not None:generator.shuffle(ids)
    groups=[]
    for start in range(0,len(ids),size*32):
        block=ids[start:start+size*32];block=block[np.lexsort((ligand_lengths[block],lengths[block]))]
        groups.extend(np.array(block[k:k+size]) for k in range(0,len(block),size))
    # Upstream BatchNorm cannot train on a singleton, so merge, never drop it.
    if len(groups)>1 and len(groups[-1])==1:
        last=groups.pop();groups[-1]=np.r_[groups[-1],last]
    if seed is not None:generator.shuffle(groups)
    return groups


def make_model(name,provider,dropout,scale):
    if name.startswith('bindra_'):
        return FollowupModel(FollowupModelConfig(arm='C-matched' if name=='bindra_pooled' else 'F',dropout=dropout,label_scale=scale))
    return provider.model(dropout)


def forward(model,batch):
    value=model(batch,validate_inputs=False) if isinstance(model,FollowupModel) else model(batch)
    return (value.values if hasattr(value,'values') and not isinstance(value,torch.Tensor) else value).reshape(-1)


def predict(model,provider,frame,device,size=32):
    model.eval();result=np.empty(len(frame))
    with torch.inference_mode():
        for ids in batches(frame,size):
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                value=forward(model,provider.batch(frame.iloc[ids],device))
            result[ids]=value.float().cpu().numpy()
    if not np.isfinite(result).all():raise ValueError('Nonfinite predictions')
    return result


def fit(name,provider,train,validation,out,*,parameters,seed,policy,device,fixed_epochs=None):
    """No test argument exists. Fixed refits cannot receive selection labels."""
    if not len(train) or train.pair_id.duplicated().any():raise ValueError('Invalid training frame')
    if fixed_epochs is not None and validation is not None:raise ValueError('Fixed refit cannot select on held-out labels')
    if fixed_epochs is None and (validation is None or not len(validation)):raise ValueError('Validation required')
    if validation is not None and set(train.pair_id)&set(validation.pair_id):raise ValueError('Supervised pair leakage')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);train=train.reset_index(drop=True)
    if validation is not None:validation=validation.reset_index(drop=True)
    seed_all(seed);model=make_model(name,provider,parameters['dropout'],policy['label_scale']).to(device)
    framehash=lambda f:None if f is None else hashlib_frame(f)
    identity={'model':name,'parameters':parameters,'seed':seed,'policy':policy,'fixed_epochs':fixed_epochs,
        'train_sha256':framehash(train),'validation_sha256':framehash(validation),'source_sha256':sha256(__file__),
        'model_config':asdict(model.config) if isinstance(model,FollowupModel) else type(model).__name__,
        'fitted_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    identity=json.loads(json.dumps(identity))
    freeze(out/'manifest.json',identity)
    marker=out/'complete.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if sha256(out/'best.pt')!=saved['checkpoint_sha256']:raise ValueError('Completed weights changed')
        return out/'best.pt'
    optimizer=torch.optim.AdamW(model.parameters(),lr=parameters['learning_rate'],weight_decay=policy['weight_decay'],fused=device.type=='cuda')
    mean=float(train.label.mean());sd=max(float(train.label.std(ddof=0)),1e-6)
    # Same train-only raw-score bias initialization and normalized loss for all methods.
    linears=[m for m in model.modules() if isinstance(m,torch.nn.Linear) and m.out_features==1]
    if len(linears)!=1:raise ValueError('Ambiguous regression head')
    torch.nn.init.constant_(linears[0].bias,mean)
    start=0;best=float('inf');best_epoch=0;patience=0;history=[];last=out/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Resume identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore(saved['rng'])
        start=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch'];patience=saved['patience'];history=saved['history']
    duration=fixed_epochs or policy['max_epochs'];physical=policy['physical_batch']
    for epoch in range(start,duration):
        if fixed_epochs is None and patience>=policy['patience']:break
        model.train();optimizer.zero_grad(set_to_none=True);accumulated=0;losses=[];seen=0;begin=time.perf_counter()
        groups=batches(train,physical,seed+epoch)
        for batch,ids in enumerate(groups,1):
            part=train.iloc[ids];target=torch.tensor(part.label.to_numpy(),device=device,dtype=torch.float32)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                value=forward(model,provider.batch(part,device)).float();loss=(((value-target)/sd)**2).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            (loss*len(part)/policy['effective_batch']).backward();accumulated+=len(part)
            if accumulated>=policy['effective_batch'] or batch==len(groups):
                for parameter in model.parameters():
                    if parameter.grad is not None:parameter.grad.mul_(policy['effective_batch']/accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();optimizer.zero_grad(set_to_none=True);accumulated=0
            losses.append(loss.detach()*len(part));seen+=len(part)
            if batch%20==0 or batch==len(groups):
                write_json(out/'progress.json',{'state':'training','epoch':epoch+1,'maximum_epochs':duration,
                    'batch':batch,'batches':len(groups),'patience':patience,'best_validation_mse':best if math.isfinite(best) and validation is not None else None,
                    'pairs_per_second':seen/max(time.perf_counter()-begin,1e-9),'at':timestamp()})
        val_mse=float(np.mean((predict(model,provider,validation,device,physical)-validation.label.to_numpy())**2)) if validation is not None else None
        improved=validation is None or val_mse<best
        trainloss=float(torch.stack(losses).sum())/seen
        if improved:best=val_mse if val_mse is not None else trainloss;best_epoch=epoch+1;patience=0
        else:patience+=1
        history.append({'epoch':epoch+1,'training_normalized_loss':trainloss,'validation_mse':val_mse,'seconds':time.perf_counter()-begin})
        payload={'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng(),'epoch':epoch+1,
            'best':best,'best_epoch':best_epoch,'patience':patience,'history':history,'train_normalization':{'mean':mean,'sd':sd}}
        atomic_save(payload,last)
        if improved:atomic_save(payload,out/'best.pt')
        write_json(out/'history.json',history)
    write_json(marker,{'checkpoint_sha256':sha256(out/'best.pt'),'best_epoch':best_epoch,'epochs':len(history),
        'best_validation_mse':best if validation is not None else None,'training_seconds':sum(x['seconds'] for x in history),'at':timestamp()})
    return out/'best.pt'


def hashlib_frame(frame):
    import hashlib
    return hashlib.sha256(frame[['pair_id','smiles','sequence','label']].to_csv(index=False,float_format='%.17g').encode()).hexdigest()
