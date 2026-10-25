"""Locked independent KIBA evaluation and additional Davis validation folds.

Test scoring is a separate gated stage. Nothing selects on test labels.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import os
import statistics

import numpy as np
import pandas as pd
import torch

from .data import official_dataset,TokenStore,GraphInputs,classical_features
from .io import sha256,freeze,write_json,timestamp,atomic_save
from .train import fit,make_model,predict,metrics

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'runs/manuscript/20261024/independent-evaluation'


def policy_digest(policy):
    return hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def compatible_fitting_policy(folder,policy,out):
    """Keep immutable fit identities when only the outer execution schedule changes.

    The trainer still checks model, data, seed, parameters, and training-source
    identities. Existing checkpoint bytes and optimization settings are untouched.
    """
    path=Path(folder)/'manifest.json'
    if not path.exists():return policy
    original=json.loads(path.read_text())['policy']
    if original==policy:return policy
    allowed={'schema','primary_selection','validation_folds','neural_grid','davis_fold_stability'}
    differences={key for key in original.keys()|policy.keys() if original.get(key)!=policy.get(key)}
    if differences-allowed:raise ValueError('Amendment changes fit optimization: '+str(sorted(differences-allowed)))
    amendment=Path(out)/'runtime_amendment.json'
    if not amendment.exists():raise ValueError('Schedule amendment must be recorded before fit reuse')
    record=json.loads(amendment.read_text())
    if record['old_policy_sha256']!=policy_digest(original) or record['new_policy_sha256']!=policy_digest(policy):
        raise ValueError('Schedule amendment identity mismatch')
    freeze(Path(folder)/'policy_reuse.json',{'old_policy_sha256':policy_digest(original),
        'current_policy_sha256':policy_digest(policy),'changed_schedule_fields':sorted(differences),
        'optimization_unchanged':True,'original_checkpoint_metadata_preserved':True})
    return original


def lock_protocol(out,config,data_root):
    out=Path(out);policy=json.loads(Path(config).read_text())
    sources=[Path(data_root)/'kiba'/p for p in ['Y','ligands_can.txt','proteins.txt','folds/train_fold_setting1.txt','folds/test_fold_setting1.txt']]
    value={'policy':policy,'dataset_files':{str(p):sha256(p) for p in sources},
        'implementation':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'upstream_manifest_sha256':sha256(Path(__file__).with_name('upstream_sources.json')),
        'historical_test_exposure':'KIBA was prepared/discussed previously. No completed local KIBA predictions found in the recorded runs audit. This is an independent dataset, not a claim of historically secret test labels.'}
    freeze(out/'protocol.json',value)
    if not (out/'protocol_frozen_at.json').exists():write_json(out/'protocol_frozen_at.json',{'at':timestamp(),'test_predictions_created':False})
    return policy


def test_gate(out,policy):
    out=Path(out)
    if not (out/'exposure/complete.json').is_file():raise ValueError('Exposure audit must finish before test scoring')
    audit=json.loads((out/'exposure/complete.json').read_text())
    if audit.get('complete') is not True:raise ValueError('Exposure audit is not complete')
    for filename,key in [('pairs.csv','pairs_sha256'),('molecules.csv','molecules_sha256')]:
        if not audit.get(key) or sha256(out/'exposure'/filename)!=audit[key]:raise ValueError('Exposure export identity changed')
    expected=policy['neural_models']+policy['classical_models']
    for name in expected:
        path=out/name/'selection.json'
        if not path.exists():raise ValueError('All selections must freeze before test scoring: '+name)
        selection=json.loads(path.read_text())
        if selection.get('selection_scope')!='validation_only':raise ValueError('Invalid model selection scope')
        if selection.get('protocol_sha256')!=sha256(out/'protocol.json'):raise ValueError('Selection protocol changed')
        seeds=[None] if name=='ridge' else policy['final_seeds']
        for seed in seeds:
            folder=out/name/f'seed-{seed}'
            if not (folder/'complete.json').exists():raise ValueError('Final training incomplete: '+name+'/'+str(seed))
            marker=json.loads((folder/'complete.json').read_text())
            asset='best.pt' if name in policy['neural_models'] else 'model.joblib'
            expected_hash=marker.get('checkpoint_sha256' if asset=='best.pt' else 'model_sha256')
            if not expected_hash or sha256(folder/asset)!=expected_hash:raise ValueError('Final model changed')
    return True


def neural(out,policy,frame,folds,device,feature_store,sources):
    train_ids=np.concatenate(folds);test_ids=frame.index[frame.partition=='test']
    if set(train_ids)&set(test_ids):raise ValueError('Official test leakage')
    for name in policy['neural_models']:
        provider=TokenStore(feature_store) if name.startswith('bindra_') else GraphInputs(name,frame,sources)
        if name.startswith('bindra_'):
            if provider.manifest['ligand_checkpoint_sha256']!=policy['ligand_checkpoint_sha256']:raise ValueError('Wrong ligand pretraining')
            if provider.manifest['protein_contract']['weights_checksum']!=policy['protein_weights_sha256']:raise ValueError('Wrong protein encoder')
            write_json(out/name/'cache_profile.json',provider.preload(frame,device))
        rows=[]
        for configuration,parameters in enumerate(policy['neural_grid']):
            for fold in policy['validation_folds']:
                validation=frame.iloc[folds[fold]];training=frame.iloc[np.concatenate([v for i,v in enumerate(folds) if i!=fold])]
                folder=out/name/'tuning'/f'configuration-{configuration:02d}'/f'fold-{fold}'
                write_json(out/'status.json',{'state':'training','stage':'independent_KIBA_tuning','model':name,
                    'configuration':configuration,'fold':fold,'run_directory':str(folder),'pid':os.getpid(),'at':timestamp()})
                fit_policy=compatible_fitting_policy(folder,policy,out)
                fit(name,provider,training,validation,folder,parameters=parameters,seed=policy['tuning_seed'],policy=fit_policy,device=device)
                result=json.loads((folder/'complete.json').read_text())
                rows.append({'configuration':configuration,'fold':fold,'validation_mse':result['best_validation_mse'],'best_epoch':result['best_epoch']})
        table=pd.DataFrame(rows);chosen=int(table.groupby('configuration').validation_mse.mean().idxmin())
        duration=math.ceil(statistics.median(table[table.configuration==chosen].best_epoch))
        selection={'selection_scope':'validation_only','configuration':chosen,'parameters':policy['neural_grid'][chosen],
            'fixed_epochs':duration,'fold_results':rows,'protocol_sha256':sha256(out/'protocol.json')}
        freeze(out/name/'selection.json',selection)
        table.to_csv(out/name/'validation_results.csv',index=False)
        for seed in policy['final_seeds']:
            folder=out/name/f'seed-{seed}'
            write_json(out/'status.json',{'state':'training','stage':'independent_KIBA_refit','model':name,'seed':seed,'run_directory':str(folder),'pid':os.getpid(),'at':timestamp()})
            fit(name,provider,frame.iloc[train_ids],None,folder,parameters=selection['parameters'],seed=seed,
                policy=policy,device=device,fixed_epochs=duration)
        del provider;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def classical(out,policy,frame,folds):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    x=classical_features(frame,out/'features/classical.npy');y=frame.label.to_numpy()
    def estimator(name,parameters,seed,train):
        if name=='ridge':
            scaler=StandardScaler(with_mean=False).fit(x[train]);model=Ridge(alpha=parameters['alpha'],solver='lsqr',tol=1e-5)
            model.fit(scaler.transform(x[train]),y[train]);return scaler,model
        model=RandomForestRegressor(**parameters,random_state=seed,n_jobs=4);model.fit(x[train],y[train]);return None,model
    for name in policy['classical_models']:
        grid=[{'alpha':a} for a in policy['ridge_alphas']] if name=='ridge' else policy['forest_grid']
        rows=[]
        for configuration,parameters in enumerate(grid):
            for fold in policy['validation_folds']:
                folder=out/name/'tuning'/f'configuration-{configuration:02d}'/f'fold-{fold}';folder.mkdir(parents=True,exist_ok=True)
                marker=folder/'complete.json'
                if marker.exists():row=json.loads(marker.read_text())
                else:
                    train=np.concatenate([v for i,v in enumerate(folds) if i!=fold]);validation=folds[fold]
                    write_json(out/'status.json',{'state':'training','stage':'classical_tuning','model':name,'configuration':configuration,'fold':fold,'pid':os.getpid(),'at':timestamp()})
                    scaler,model=estimator(name,parameters,policy['tuning_seed'],train)
                    values=model.predict(scaler.transform(x[validation]) if scaler else x[validation])
                    row={'configuration':configuration,'fold':fold,'validation_mse':float(np.mean((values-y[validation])**2))};write_json(marker,row)
                    del model,scaler;gc.collect()
                rows.append(row)
        table=pd.DataFrame(rows);chosen=int(table.groupby('configuration').validation_mse.mean().idxmin())
        selection={'selection_scope':'validation_only','parameters':grid[chosen],'fold_results':rows,
                   'deterministic_once':name=='ridge','protocol_sha256':sha256(out/'protocol.json')}
        freeze(out/name/'selection.json',selection)
        for seed in [None] if name=='ridge' else policy['final_seeds']:
            folder=out/name/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():continue
            scaler,model=estimator(name,selection['parameters'],seed,np.concatenate(folds))
            joblib.dump((scaler,model),folder/'model.joblib')
            write_json(folder/'complete.json',{'model_sha256':sha256(folder/'model.joblib'),'selection_sha256':sha256(out/name/'selection.json')})
            del model,scaler;gc.collect()


def evaluate(out,policy,frame,folds,test,device,feature_store,sources):
    import joblib
    test_gate(out,policy);rows=[]
    evaluation=frame.iloc[np.sort(test)].reset_index(drop=True)
    exposure=pd.read_csv(out/'exposure/pairs.csv').set_index('pair_id')
    for name in policy['neural_models']+policy['classical_models']:
        selection=json.loads((out/name/'selection.json').read_text());members=[]
        provider=TokenStore(feature_store) if name.startswith('bindra_') else GraphInputs(name,frame,sources) if name in policy['neural_models'] else None
        x=classical_features(frame,out/'features/classical.npy') if provider is None else None
        for seed in [None] if name=='ridge' else policy['final_seeds']:
            folder=out/name/f'seed-{seed}'
            if provider is not None:
                model=make_model(name,provider,selection['parameters']['dropout'],policy['label_scale']).to(device)
                model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
                values=predict(model,provider,evaluation,device,policy['physical_batch']);del model
            else:
                scaler,model=joblib.load(folder/'model.joblib');data=x[np.sort(test)];values=model.predict(scaler.transform(data) if scaler else data)
            result=evaluation[['pair_id','label']].copy();result['predicted']=values;result.to_csv(folder/'test_predictions.csv',index=False)
            rows.append({'model':name,'seed':seed,'aggregation':'single','stratum':'all',**metrics(evaluation.label,values,policy['label_scale'])});members.append(values)
        values=np.stack(members).mean(0)
        ensemble=evaluation[['pair_id','label']].copy();ensemble['predicted']=values;ensemble.to_csv(out/name/'ensemble_predictions.csv',index=False)
        for stratum,mask in [('all',np.ones(len(ensemble),bool))]+[(column+'='+str(v),
            (exposure.loc[ensemble.pair_id,column].isna() if v=='unknown' else exposure.loc[ensemble.pair_id,column].eq(v)).to_numpy())
            for column in ['ligand_pretraining_identity','davis_supervised_pair'] for v in [True,False,'unknown']]:
            if mask.any():rows.append({'model':name,'seed':None,'aggregation':'deterministic_once' if name=='ridge' else 'equal_mean_three','stratum':stratum,
                **metrics(ensemble.label.to_numpy()[mask],values[mask],policy['label_scale'])})
        del provider;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    write_json(out/'evaluation_complete.json',{'results_sha256':sha256(out/'results.csv'),'test_n':len(test),'at':timestamp()})


def davis_stability(out,policy,data_root,feature_store,device):
    frame,folds,_=official_dataset('davis',data_root)
    provider=TokenStore(feature_store);subpolicy={**policy,'label_scale':'pKd'};rows=[]
    write_json(out/'davis_stability/cache_profile.json',provider.preload(frame,device))
    confirmation=policy['davis_fold_stability'].get('mode')=='fixed_configuration_confirmation'
    for fold in policy['davis_fold_stability']['additional_folds']:
        for arm in policy['davis_fold_stability']['arms']:
            from .models import FollowupModel,FollowupModelConfig
            class ArmProvider:
                def batch(self,frame,device):return provider.batch(frame,device)
                def model(self,dropout):return FollowupModel(FollowupModelConfig(arm=arm,dropout=dropout,label_scale='pKd'))
            training=frame.iloc[np.concatenate([v for i,v in enumerate(folds) if i!=fold])];validation=frame.iloc[folds[fold]]
            tuning=[]
            grid=[policy['davis_fold_stability']['fixed_parameters']] if confirmation else policy['neural_grid']
            for configuration,parameters in enumerate(grid):
                folder=out/'davis_stability'/f'fold-{fold}'/arm/'tuning'/f'configuration-{configuration:02d}'
                write_json(out/'status.json',{'state':'training','stage':'Davis_foldwise_retuning','fold':fold,'arm':arm,
                    'configuration':configuration,'run_directory':str(folder),'pid':os.getpid(),'at':timestamp()})
                fit('davis_validation_arm',ArmProvider(),training,validation,folder,parameters=parameters,
                    seed=policy['tuning_seed'],policy=subpolicy,device=device)
                record=json.loads((folder/'complete.json').read_text());tuning.append({'configuration':configuration,**record})
            chosen=min(tuning,key=lambda row:(row['best_validation_mse'],row['configuration']))['configuration']
            parameters=grid[chosen]
            freeze(out/'davis_stability'/f'fold-{fold}'/arm/'selection.json',{'selection_scope':'validation_only',
                'configuration':chosen,'parameters':parameters,'tuning_results':tuning,
                'hyperparameters_selected_on_this_fold':not confirmation,
                'comparison_scope':'Fixed fold-0-selected settings; two leading architectures only' if confirmation else 'All declared tuning configurations'})
            for seed in policy['davis_fold_stability']['seeds']:
                folder=out/'davis_stability'/f'fold-{fold}'/arm/('tuning/configuration-'+f'{chosen:02d}' if seed==policy['tuning_seed'] else f'seed-{seed}')
                write_json(out/'status.json',{'state':'training','stage':'Davis_additional_validation','fold':fold,'arm':arm,'seed':seed,'run_directory':str(folder),'pid':os.getpid(),'at':timestamp()})
                fit('davis_validation_arm',ArmProvider(),training,validation,folder,parameters=parameters,seed=seed,policy=subpolicy,device=device)
                record=json.loads((folder/'complete.json').read_text());rows.append({'fold':fold,'arm':arm,'seed':seed,'validation_mse':record['best_validation_mse']})
                pd.DataFrame(rows).to_csv(out/'davis_stability/results.csv',index=False)
    means=pd.DataFrame(rows).groupby(['fold','arm']).validation_mse.mean().reset_index()
    means['rank']=means.groupby('fold').validation_mse.rank(method='min');means.to_csv(out/'davis_stability/ranks.csv',index=False)
    write_json(out/'davis_stability/complete.json',{'selected_configuration_seed_scores':len(rows),
        'unique_validation_fits':policy['davis_fold_stability']['planned_unique_validation_fits'],
        'foldwise_hyperparameters_retuned':not confirmation,'test_predictions_created':False,'at':timestamp()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',default=str(Path(__file__).with_name('protocol.json')))
    parser.add_argument('--data',default='data/external');parser.add_argument('--out',default=str(DEFAULT_OUT))
    parser.add_argument('--features');parser.add_argument('--davis-features',default='runs/manuscript/20261018/architecture-generalization/features/davis')
    parser.add_argument('--sources',default='data/external/reviewer_baselines');parser.add_argument('--device',default='auto')
    parser.add_argument('--stage',choices=['freeze','neural','classical','evaluate','davis-stability','all'],default='freeze')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    policy=lock_protocol(out,args.protocol,args.data)
    if args.stage=='freeze':print('Protocol frozen before new KIBA scoring:',out);return
    if args.device=='auto':
        device=torch.device(f'cuda:{max(range(torch.cuda.device_count()),key=lambda i:torch.cuda.get_device_properties(i).total_memory)}' if torch.cuda.is_available() else 'cpu')
    else:device=torch.device(args.device)
    torch.set_num_threads(8)
    frame,folds,test=official_dataset('kiba',args.data,policy['protein_input_cap'])
    from .lease import gpu_lease
    from contextlib import nullcontext
    with gpu_lease(ROOT/'runs/manuscript/gpu-worker.lock') if args.stage!='classical' else nullcontext():
        if args.stage in ['neural','all']:neural(out,policy,frame,folds,device,args.features,args.sources)
        if args.stage in ['classical','all']:classical(out,policy,frame,folds)
        if args.stage in ['evaluate','all']:evaluate(out,policy,frame,folds,test,device,args.features,args.sources)
        if args.stage in ['davis-stability','all']:davis_stability(out,policy,args.data,args.davis_features,device)
    write_json(out/'status.json',{'state':'complete' if args.stage=='all' else 'stage_complete','stage':args.stage,'at':timestamp()})


if __name__=='__main__':main()
