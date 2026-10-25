"""Official folds, pinned baseline inputs, and frozen representation cache ABI."""
from collections import OrderedDict
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import pickle
import urllib.request

import numpy as np
import pandas as pd
import torch

from .io import sha256,freeze,write_json
from .models import CachedFeatures


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def official_dataset(name,root,protein_cap=None):
    if name not in ('davis','kiba'):raise ValueError('Unknown benchmark')
    folder=Path(root)/name
    drugs=json.loads((folder/'ligands_can.txt').read_text(),object_pairs_hook=OrderedDict)
    proteins=json.loads((folder/'proteins.txt').read_text(),object_pairs_hook=OrderedDict)
    with (folder/'Y').open('rb') as handle:y=np.asarray(pickle.load(handle,encoding='latin1'),float)
    rows,cols=np.where(np.isfinite(y));labels=y[rows,cols]
    if name=='davis':labels=-np.log10(labels/1e9)
    folds=json.loads((folder/'folds/train_fold_setting1.txt').read_text())
    test=json.loads((folder/'folds/test_fold_setting1.txt').read_text())
    all_ids=np.concatenate([*folds,test])
    if len(folds)!=5 or len(all_ids)!=len(labels) or not np.array_equal(np.sort(all_ids),np.arange(len(labels))):
        raise ValueError('Official folds are not a disjoint exhaustive partition')
    if len(test)!={'davis':5010,'kiba':19709}[name]:raise ValueError('Unexpected official test size')
    sequence=np.asarray(list(proteins.values()))[cols]
    frame=pd.DataFrame({'pair_id':[f'{name}:{i}' for i in range(len(labels))],
        'official_pair_index':np.arange(len(labels)),'drug_index':rows,'target_index':cols,
        'smiles':np.asarray(list(drugs.values()))[rows],'sequence_raw':sequence,'sequence':sequence,
        'label':labels,'at_detection_limit':y[rows,cols]==10000 if name=='davis' else False})
    if protein_cap is not None:frame['sequence']=frame.sequence.str[:protein_cap]
    frame['partition']='train'
    frame.loc[test,'partition']='test'
    return frame,[np.array(f,dtype=int) for f in folds],np.asarray(test,dtype=int)


class TokenStore:
    """Cache only frozen upstream values; masks and variable token axes persist."""
    def __init__(self,directory):
        self.directory=Path(directory);self.manifest=json.loads((self.directory/'manifest.json').read_text())
        if self.manifest.get('schema')!='bindra-followup-frozen-encoders-v1':raise ValueError('Unknown frozen feature schema')
        self.pool=OrderedDict()

    def get(self,kind,text):
        key=(kind,text)
        if key not in self.pool:
            value=torch.load(self.directory/kind/(digest(text)+'.pt'),map_location='cpu',weights_only=False)
            expected=self.manifest['protein_contract'] if kind=='protein' else self.manifest['ligand_checkpoint_sha256']
            if value['input_sha256']!=digest(text) or value['encoder_identity']!=expected:raise ValueError('Frozen cache identity mismatch')
            self.pool[key]=value
            while len(self.pool)>4096:self.pool.popitem(last=False)
        self.pool.move_to_end(key);return self.pool[key]

    def batch(self,frame,device):
        if hasattr(self,'gpu'):
            ps=[self.gpu['pi'][s] for s in frame.sequence];ls=[self.gpu['li'][s] for s in frame.smiles]
            length=max(self.gpu['pl'][i] for i in ps);tokens=max(self.gpu['ll'][i] for i in ls)
            pids=torch.tensor(ps,device=device);lids=torch.tensor(ls,device=device)
            return CachedFeatures(protein_features=self.gpu['p'][:,:length].index_select(0,pids).float(),
                protein_mask=self.gpu['pm'][:,:length].index_select(0,pids),
                ligand_tokens=self.gpu['l'][:,:tokens].index_select(0,lids).float(),
                ligand_mask=self.gpu['lm'][:,:tokens].index_select(0,lids),
                physical_mask=self.gpu['physical'][:,:tokens].index_select(0,lids),
                graph_states=self.gpu['graph'].index_select(0,lids).float())
        p=[self.get('protein',s)['features'] for s in frame.sequence]
        l=[self.get('ligand',s) for s in frame.smiles]
        n=len(frame);length=max(map(len,p));tokens=max(len(x['tokens']) for x in l)
        protein=torch.zeros(n,length,p[0].shape[-1]);pm=torch.zeros(n,length,dtype=torch.bool)
        ligand=torch.zeros(n,tokens,l[0]['tokens'].shape[-1]);lm=torch.zeros(n,tokens,dtype=torch.bool);physical=lm.clone()
        for i,(pv,lv) in enumerate(zip(p,l)):
            protein[i,:len(pv)]=pv.float();pm[i,:len(pv)]=True
            ligand[i,:len(lv['tokens'])]=lv['tokens'].float();lm[i,:len(lv['tokens'])]=True
            physical[i,:len(lv['tokens'])]=lv['physical_mask']
        graph=torch.stack([x['graph_states'].float() for x in l])
        return CachedFeatures(protein_features=protein.to(device),protein_mask=pm.to(device),
            ligand_tokens=ligand.to(device),ligand_mask=lm.to(device),physical_mask=physical.to(device),graph_states=graph.to(device))

    def preload(self,frame,device,budget_gib=1.5):
        if device.type!='cuda':return {'enabled':False,'reason':'CPU'}
        proteins=sorted(frame.sequence.unique());ligands=sorted(frame.smiles.unique())
        pv=[self.get('protein',s)['features'] for s in proteins];lv=[self.get('ligand',s) for s in ligands]
        pl=[len(x) for x in pv];ll=[len(x['tokens']) for x in lv];lp=max(pl);lt=max(ll)
        byte_count=2*(len(pv)*lp*pv[0].shape[-1]+len(lv)*lt*lv[0]['tokens'].shape[-1])
        if byte_count>budget_gib*1024**3:return {'enabled':False,'reason':'bounded GPU cache budget','bytes':byte_count}
        before=self.batch(frame.head(3),device)
        p=torch.zeros(len(pv),lp,pv[0].shape[-1],device=device,dtype=torch.bfloat16);pm=torch.zeros(len(pv),lp,device=device,dtype=torch.bool)
        l=torch.zeros(len(lv),lt,lv[0]['tokens'].shape[-1],device=device,dtype=torch.bfloat16);lm=torch.zeros(len(lv),lt,device=device,dtype=torch.bool);physical=lm.clone()
        for i,v in enumerate(pv):p[i,:len(v)]=v.to(device);pm[i,:len(v)]=True
        for i,v in enumerate(lv):l[i,:len(v['tokens'])]=v['tokens'].to(device);lm[i,:len(v['tokens'])]=True;physical[i,:len(v['tokens'])]=v['physical_mask'].to(device)
        self.gpu={'p':p,'pm':pm,'l':l,'lm':lm,'physical':physical,'graph':torch.stack([x['graph_states'] for x in lv]).to(device),
            'pi':{s:i for i,s in enumerate(proteins)},'li':{s:i for i,s in enumerate(ligands)},'pl':pl,'ll':ll}
        after=self.batch(frame.head(3),device)
        for key in vars(before):torch.testing.assert_close(getattr(before,key),getattr(after,key),atol=0,rtol=0)
        return {'enabled':True,'bytes':byte_count,'equivalence':'exact padded inputs and masks; no cached trainable computation'}


def module_at(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def baseline_files(name,directory):
    manifest=json.loads(Path(__file__).with_name('upstream_sources.json').read_text())[name]
    folder=Path(directory)/name
    for filename,expected in manifest['files'].items():
        destination=folder/filename
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            url=f'https://raw.githubusercontent.com/{manifest["repository"]}/{manifest["revision"]}/{filename}'
            with urllib.request.urlopen(url,timeout=60) as response:destination.write_bytes(response.read())
        if sha256(destination)!=expected:raise ValueError('Pinned upstream source changed: '+str(destination))
    return folder


class GraphInputs:
    def __init__(self,name,frame,source_directory):
        from rdkit import Chem,RDLogger
        from torch_geometric.data import Data
        RDLogger.DisableLog('rdApp.*')
        self.name=name;folder=baseline_files(name,source_directory)
        if name=='mgraphdta':
            preprocessing=module_at(folder/'regression/preprocessing.py','pinned_mgraph_inputs')
            self.module=module_at(folder/'regression/model.py','pinned_mgraph_model')
            tokenizer=preprocessing.VOCAB_PROTEIN
            featurizer=object.__new__(preprocessing.GNNDataset)
        elif name=='graphdta_gin':
            self.module=module_at(folder/'models/ginconv.py','pinned_graphdta_model')
            tokenizer={c:i+1 for i,c in enumerate('ABCDEFGHIKLMNOPQRSTUVWXYZ')}
        else:raise ValueError(name)
        self.proteins={}
        for sequence in frame.sequence.unique():
            value=torch.zeros(1000,dtype=torch.long)
            value[:min(len(sequence),1000)]=torch.tensor([tokenizer[c] for c in sequence[:1000]])
            self.proteins[sequence]=value
        self.graphs={}
        for smiles in frame.smiles.unique():
            mol=Chem.MolFromSmiles(smiles)
            if mol is None:raise ValueError('Invalid common candidate; no per-model deletion allowed')
            if name=='mgraphdta':
                x,edges,attrs=featurizer.mol2graph(mol);span=x.max()-x.min()
                if span<=0:raise ValueError('Degenerate official molecule normalization')
                x=(x-x.min())/span
            else:
                symbols=['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
                def one(value,choices,strict=False):
                    if value not in choices:
                        if strict:raise ValueError('Unsupported official atom degree')
                        value=choices[-1]
                    return [value==x for x in choices]
                values=[one(a.GetSymbol(),symbols)+one(a.GetDegree(),list(range(11)),True)+
                    one(a.GetTotalNumHs(),list(range(11)))+one(a.GetValence(Chem.rdchem.ValenceType.IMPLICIT),list(range(11)))+[a.GetIsAromatic()] for a in mol.GetAtoms()]
                x=torch.tensor(values,dtype=torch.float32);x=x/x.sum(1,keepdim=True)
                links=[p for b in mol.GetBonds() for p in [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),(b.GetEndAtomIdx(),b.GetBeginAtomIdx())]]
                edges=torch.tensor(links,dtype=torch.long).reshape(-1,2).T.contiguous();attrs=None
            self.graphs[smiles]=Data(x=x,edge_index=edges,edge_attr=attrs)

    def model(self,dropout):
        if self.name=='graphdta_gin':return self.module.GINConvNet(dropout=dropout)
        result=self.module.MGraphDTA(block_num=3,vocab_protein_size=26)
        for layer in result.modules():
            if isinstance(layer,torch.nn.Dropout):layer.p=dropout
        return result

    def batch(self,frame,device):
        from torch_geometric.data import Batch
        values=[]
        for row in frame.itertuples():
            graph=self.graphs[row.smiles].clone();graph.target=self.proteins[row.sequence].unsqueeze(0);values.append(graph)
        return Batch.from_data_list(values).to(device)


def classical_features(frame,path):
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from .io import sha256
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(frame[['pair_id','smiles','sequence']].to_csv(index=False).encode()).hexdigest()
    metadata={'input_sha256':identity,'shape':[len(frame),2468],'schema':'Morgan2048_radius2_AAC20_DPC400_v1'}
    marker=path.with_suffix('.json')
    if path.exists() and marker.exists():
        saved=json.loads(marker.read_text())
        if any(saved[k]!=v for k,v in metadata.items()) or sha256(path)!=saved['sha256']:raise ValueError('Classical cache identity mismatch')
        return np.load(path,mmap_mode='r')
    features=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(frame),2468))
    amino='ACDEFGHIKLMNPQRSTVWY';pc={};mc={};generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    for i,row in enumerate(frame.itertuples()):
        if row.smiles not in mc:
            mol=Chem.MolFromSmiles(row.smiles)
            if mol is None:raise ValueError('Invalid ligand')
            vector=np.zeros(2048,np.float32);DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol),vector);mc[row.smiles]=vector
        if row.sequence not in pc:
            vector=np.zeros(420,np.float32);codes=[amino.find(c) for c in row.sequence]
            for c in codes:
                if c>=0:vector[c]+=1/len(codes)
            for a,b in zip(codes[:-1],codes[1:]):
                if a>=0 and b>=0:vector[20+20*a+b]+=1/(len(codes)-1)
            pc[row.sequence]=vector
        features[i]=np.r_[mc[row.smiles],pc[row.sequence]]
    features.flush();write_json(marker,{**metadata,'sha256':sha256(path)});return features
