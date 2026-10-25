import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from reference.data import TokenStore,digest
from reference.io import atomic_save,freeze
from reference.models import CachedFeatures,FollowupModel,FollowupModelConfig,trainable_count
from reference.study import test_gate as require_test_gate
from reference.train import batches,ci,fit,metrics


def test_counts_and_scale_are_the_actual_downstream_architectures():
    for arm,expected in [('C-matched',2455221),('F',2228293)]:
        model=FollowupModel(FollowupModelConfig(arm=arm,label_scale='KIBA_score'))
        assert trainable_count(model)==expected
        assert model.config.label_scale=='KIBA_score'
    with pytest.raises(ValueError):FollowupModelConfig(label_scale='pKi')


def test_ci_handles_truth_and_prediction_ties():
    y=np.array([1,1,2,3,4]);p=np.array([2,1,2,3,3])
    values=[float(p[j]>p[i])+.5*float(p[j]==p[i]) for i in range(len(y)) for j in range(i+1,len(y)) if y[j]>y[i]]
    assert ci(y,p)==pytest.approx(np.mean(values))
    assert ci([1,2,3],[2,2,2])==.5
    assert ci([1,1],[0,2]) is None
    assert metrics([10,12],[11,13],'KIBA_score')['mse']==1
    with pytest.raises(ValueError):metrics([10],[10],'pX')


def test_batches_keep_every_row_and_merge_a_singleton():
    frame=pd.DataFrame({'sequence':['AC']*65,'smiles':['CC']*65})
    groups=batches(frame,32,seed=7)
    assert sorted(np.concatenate(groups))==list(range(65))
    assert min(map(len,groups))>1


def test_test_gate_rejects_incomplete_or_test_selected_runs(tmp_path):
    policy={'neural_models':['bindra_pooled'],'classical_models':[],'final_seeds':[1,2,3]}
    with pytest.raises(ValueError,match='Exposure'):require_test_gate(tmp_path,policy)
    from reference.io import sha256
    (tmp_path/'exposure').mkdir()
    for name in ['pairs','molecules']:(tmp_path/f'exposure/{name}.csv').write_text('id\n1\n')
    (tmp_path/'exposure/complete.json').write_text(json.dumps({'complete':True,**{name+'_sha256':sha256(tmp_path/f'exposure/{name}.csv') for name in ['pairs','molecules']}}))
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'bindra_pooled').mkdir();selection=tmp_path/'bindra_pooled/selection.json'
    selection.write_text(json.dumps({'selection_scope':'test_MSE'}))
    with pytest.raises(ValueError,match='selection scope'):require_test_gate(tmp_path,policy)
    selection.write_text(json.dumps({'selection_scope':'validation_only','protocol_sha256':sha256(tmp_path/'protocol.json')}))
    with pytest.raises(ValueError,match='incomplete'):require_test_gate(tmp_path,policy)


def token_store(tmp_path):
    contract={'weights_checksum':'original'}
    (tmp_path/'manifest.json').write_text(json.dumps({'schema':'bindra-followup-frozen-encoders-v1',
        'protein_contract':contract,'ligand_checkpoint_sha256':'original-light'}))
    for seq in ['AC','ACD']:
        atomic_save({'input_sha256':digest(seq),'encoder_identity':contract,'features':torch.ones(len(seq),1152,dtype=torch.bfloat16)},
                    tmp_path/'protein'/(digest(seq)+'.pt'))
    for smiles in ['CC','CCC']:
        atomic_save({'input_sha256':digest(smiles),'encoder_identity':'original-light','tokens':torch.ones(len(smiles),96,dtype=torch.bfloat16),
            'physical_mask':torch.tensor([True]+[False]*(len(smiles)-1)),'graph_states':torch.ones(288,dtype=torch.bfloat16)},
                    tmp_path/'ligand'/(digest(smiles)+'.pt'))
    return TokenStore(tmp_path)


def test_cache_keeps_axes_masks_and_frozen_identity(tmp_path):
    store=token_store(tmp_path);frame=pd.DataFrame({'sequence':['AC','ACD'],'smiles':['CC','CCC']})
    batch=store.batch(frame,torch.device('cpu'))
    assert batch.protein_features.shape==(2,3,1152) and batch.ligand_tokens.shape==(2,3,96)
    assert batch.protein_mask.tolist()==[[True,True,False],[True,True,True]]
    assert batch.physical_mask.sum()==2 and batch.graph_states.shape==(2,288)
    model=FollowupModel(FollowupModelConfig(arm='F',label_scale='KIBA_score'))
    assert model(batch).values.shape==(2,)
    record=store.get('protein','AC');record['encoder_identity']={'weights_checksum':'adapted'}
    atomic_save(record,tmp_path/'protein'/(digest('AC')+'.pt'))
    with pytest.raises(ValueError,match='identity'):TokenStore(tmp_path).get('protein','AC')


def test_training_has_no_test_argument_and_fixed_refit_rejects_selection(tmp_path):
    import inspect
    assert 'test' not in inspect.signature(fit).parameters
    train=pd.DataFrame({'pair_id':['a'],'sequence':['AC'],'smiles':['CC'],'label':[1.]})
    with pytest.raises(ValueError,match='Fixed refit'):
        fit('unused',None,train,train,tmp_path,parameters={},seed=1,policy={},device=torch.device('cpu'),fixed_epochs=1)


def test_freeze_rejects_changed_protocol(tmp_path):
    freeze(tmp_path/'lock.json',{'selection':'validation_only'})
    with pytest.raises(ValueError,match='Frozen identity'):freeze(tmp_path/'lock.json',{'selection':'test'})


def test_real_fit_resume_keeps_checkpoint_and_training_only_normalization(tmp_path):
    features=tmp_path/'features';features.mkdir();store=token_store(features)
    training=pd.DataFrame({'pair_id':['a','b','c','d'],'sequence':['AC','ACD']*2,'smiles':['CC','CCC']*2,'label':[8.,9.,10.,11.]})
    validation=pd.DataFrame({'pair_id':['e','f'],'sequence':['AC','ACD'],'smiles':['CC','CCC'],'label':[7.,12.]})
    policy={'label_scale':'KIBA_score','weight_decay':.01,'max_epochs':1,'patience':1,'physical_batch':2,'effective_batch':2}
    params={'dropout':.1,'learning_rate':.0003}
    checkpoint=fit('bindra_pooled',store,training,validation,tmp_path/'fit',parameters=params,seed=19,policy=policy,device=torch.device('cpu'))
    from reference.io import sha256
    before=sha256(checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['train_normalization']['mean']==pytest.approx(training.label.mean())
    assert saved['optimizer']['state'] and saved['rng']['torch'] is not None
    fit('bindra_pooled',store,training,validation,tmp_path/'fit',parameters=params,seed=19,policy=policy,device=torch.device('cpu'))
    assert sha256(checkpoint)==before
    with pytest.raises(ValueError,match='Frozen identity'):
        fit('bindra_pooled',store,training,validation,tmp_path/'fit',parameters={**params,'learning_rate':.001},seed=19,policy=policy,device=torch.device('cpu'))


def test_official_fold_indexing_and_kiba_scale_when_assets_available():
    from reference.data import official_dataset
    root=Path(__file__).resolve().parents[2]/'data/external'
    if not (root/'kiba/Y').exists():pytest.skip('Licensed benchmark assets are separate from reference code')
    import pickle
    for name,n in [('davis',5010),('kiba',19709)]:
        frame,folds,test=official_dataset(name,root,1000)
        with (root/name/'Y').open('rb') as handle:y=np.asarray(pickle.load(handle,encoding='latin1'))
        r,c=np.where(np.isfinite(y));raw=y[r,c]
        expected=-np.log10(raw/1e9) if name=='davis' else raw
        assert np.allclose(frame.label,expected) and len(test)==n
        assert np.array_equal(frame.drug_index,r) and np.array_equal(frame.target_index,c)
        assert not set(np.concatenate(folds))&set(test)
        assert frame.sequence.str.len().max()<=1000


def test_prediction_batching_restores_original_order():
    from reference.train import predict
    class Provider:
        def batch(self,frame,device):return torch.tensor(frame.value.to_numpy(),device=device).float()
    class Model(torch.nn.Module):
        def forward(self,x):return x*2
    frame=pd.DataFrame({'sequence':['A'*n for n in [4,1,8,2,3]],'smiles':['CC']*5,'value':[9,4,1,8,3]})
    assert np.array_equal(predict(Model(),Provider(),frame,torch.device('cpu'),2),frame.value.to_numpy()*2)


def test_transient_windows_file_lock_does_not_abort_checkpointing(tmp_path,monkeypatch):
    import reference.io as io
    original=io.os.replace;calls=[]
    def transient(source,destination):
        calls.append(str(destination))
        if len(calls)<=2:raise PermissionError('Temporary file-sharing lock')
        return original(source,destination)
    monkeypatch.setattr(io.os,'replace',transient);monkeypatch.setattr(io.time,'sleep',lambda seconds:None)
    io.write_json(tmp_path/'progress.json',{'epoch':4})
    assert len(calls)==3 and json.loads((tmp_path/'progress.json').read_text())=={'epoch':4}
    io.atomic_save({'epoch':4},tmp_path/'last.pt')
    assert torch.load(tmp_path/'last.pt',weights_only=False)['epoch']==4
