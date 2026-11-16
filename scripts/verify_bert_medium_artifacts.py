"""Verify downloaded evidence hashes, exact budgets, input pairing and CFS cubes."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.bert_medium import SPECS,fingerprint

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    manifest=json.loads((root/'ARTIFACT_MANIFEST.json').read_text())
    for row in manifest:
        path=(root/row['path']).resolve();assert path.is_relative_to(root.resolve())
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
    checkpoints=json.loads((root/'CHECKPOINT_MANIFEST.json').read_text());assert len(checkpoints)==28
    checks=[];targets=[];documents=[];validation_ids=[]
    for model,(_,_,h) in SPECS.items():
        state=json.loads((root/'runs'/model/'COMPLETE.json').read_text())
        assert state['tokens']==2000000000 and state['step']==61036 and state['protocol_hash']==fingerprint()
        targets.append(state['mlm_targets']);documents.append(state['document_visits'])
        with np.load(root/'runs'/model/'validation_2000000000.npz') as z:validation_ids.append(z['document_sha256'].tolist())
        for budget,n in [('1B',64),('2B',256)]:
            directory=root/'cfs'/model/budget;assert (directory/'COMPLETE.json').exists()
            meta=json.loads((directory/'metadata.json').read_text());ids=[];valid=[]
            for path in sorted(directory.glob('shard_*.npz')):
                with np.load(path) as z:
                    batch=len(z['dataset_index']);ids.extend(z['dataset_index'].tolist())
                    drop=z['single_drop'];sub=z['substitution'];cube=z['alpha_error_cube'];alpha=z['best_alpha']
                    assert drop.shape==(batch,1,h) and sub.shape==(batch,1,h,h) and cube.shape==(batch,1,h,h,16)
                    off=~np.eye(h,dtype=bool)
                    assert np.isfinite(drop).all() and np.isfinite(cube[:,:,off,:]).all()
                    np.testing.assert_array_equal(sub[:,:,off],cube[:,:,off,:].min(-1))
                    np.testing.assert_allclose(alpha[:,:,off],np.linspace(0,3,16)[cube[:,:,off,:].argmin(-1)],rtol=0,atol=1e-6)
                    assert np.isnan(sub[:,:,np.eye(h,dtype=bool)]).all()
                    valid.extend((drop>1e-5).reshape(-1).tolist())
            assert ids==meta['sample_indices'] and len(ids)==n and len(set(ids))==n
            check=json.loads((directory/'prefix_equivalence.json').read_text())
            checks.append(dict(model=model,budget=budget,inputs=n,valid_source_ratio=float(np.mean(valid)),**check))
    assert len(set(targets))==len(set(documents))==1
    assert all(x==validation_ids[0] for x in validation_ids) and len(set(validation_ids[0]))==2048
    output=dict(verified_evidence_files=len(manifest),milestone_checkpoints=28,training_tokens_per_model=2000000000,
                identical_target_counts=targets[0],identical_document_visits=documents[0],checks=checks)
    (root/'VERIFICATION.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
