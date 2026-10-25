"""Keep pretraining chemistry and prior supervised-pair exposure separate."""
from pathlib import Path
import argparse
import gzip
import json
import pandas as pd

from bindra.data.bindingdb.standardize import standardize_smiles
from .data import official_dataset
from .io import sha256,write_json


def audit(frame,data_root,corpus,reference_table,reference_manifest,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(Path(reference_manifest).read_text())
    if sha256(corpus)!=metadata['pretraining_corpus_sha256'] or sha256(reference_table)!=metadata['reference_sha256']:
        raise ValueError('Exposure reference is not bound to the actual ligand-pretraining corpus')
    with gzip.open(corpus,'rt') as handle:smiles={line.strip() for line in handle if line.strip()}
    ref=pd.read_parquet(reference_table)
    training=ref[ref.standardized_smiles.isin(smiles)]
    missing=smiles-set(training.standardized_smiles)
    if missing:raise ValueError(f'{len(missing)} pretraining molecules missing from reference; no clean classification permitted')
    identities=set(training.identity_key.dropna());parents=set(training.leakage_key.dropna());scaffolds=set(training.scaffold_key.dropna())
    davis,folds,_=official_dataset('davis',data_root)
    supervised=davis.iloc[[i for fold in folds for i in fold]]
    molecule={}
    for text in set(frame.smiles)|set(supervised.smiles):
        item=standardize_smiles(text)
        molecule[text]={'identity_key':item.identity_key,'parent_key':item.leakage_key,'scaffold_key':item.scaffold_key,'normalization_status':item.status}
    supervised_pairs={(molecule[row.smiles]['identity_key'],row.sequence_raw) for row in supervised.itertuples() if molecule[row.smiles]['identity_key']}
    records=[]
    for text in sorted(frame.smiles.unique()):
        m=molecule[text];valid=bool(m['identity_key'])
        records.append({'smiles':text,**m,'ligand_pretraining_identity':m['identity_key'] in identities if valid else None,
            'ligand_pretraining_parent':m['parent_key'] in parents if valid else None,
            'ligand_pretraining_scaffold':m['scaffold_key'] in scaffolds if valid else None,
            'raw_smiles_in_pretraining':text in smiles})
    molecules=pd.DataFrame(records);molecules.to_csv(out/'molecules.csv',index=False)
    pairs=frame[['pair_id','smiles','sequence_raw','partition']].merge(molecules,on='smiles',validate='many_to_one')
    pairs['davis_supervised_pair']=[(key,sequence) in supervised_pairs if pd.notna(key) else None for key,sequence in zip(pairs.identity_key,pairs.sequence_raw)]
    pairs.to_csv(out/'pairs.csv',index=False)
    result={'complete':True,'pairs_sha256':sha256(out/'pairs.csv'),'molecules_sha256':sha256(out/'molecules.csv'),
        'corpus_sha256':sha256(corpus),'reference_sha256':sha256(reference_table),'pretraining_molecules':len(smiles),
        'molecules':len(molecules),'unresolved_chemical_normalizations':int(molecules.identity_key.isna().sum()),
        'counts':{c:molecules[c].value_counts(dropna=False).astype(int).to_dict() for c in ['ligand_pretraining_identity','ligand_pretraining_parent','ligand_pretraining_scaffold']},
        'protein_pretraining':'Historical ESM-C exposure unknown; no chemically clean stratum implies unseen protein pretraining.',
        'supervised_scope':'Davis official full supervised training pool; test observations excluded from supervised-pair reference.',
        'unknowns':'Unresolved normalization is not classified as unexposed.'}
    # JSON object keys must not turn an unknown into a boolean False.
    result['counts']={k:{str(key):value for key,value in values.items()} for k,values in result['counts'].items()}
    write_json(out/'complete.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data/external');parser.add_argument('--corpus',required=True)
    parser.add_argument('--reference-table',required=True);parser.add_argument('--reference-manifest',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();frame,_,_=official_dataset('kiba',args.data)
    print(audit(frame,args.data,args.corpus,args.reference_table,args.reference_manifest,args.out))
