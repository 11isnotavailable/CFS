"""Collect the authorized fresh cohort on GPU 1, then run offline controls.

No training. Run from the isolated experiment directory after preparation.
Requires model_provenance.json and imagenet_provenance.json from preparation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/data/cfs_rebuttal_20261114')
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HOME']=str(ROOT/'cache/huggingface')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')


def run(args,log):
    print('RUN',args,flush=True)
    with (ROOT/log).open('a') as handle:
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)


def check_collection(directory, heads, layers, alpha_steps, samples):
    import numpy as np
    meta=json.loads((directory/'metadata.json').read_text())
    assert meta['heads']==heads
    assert meta.get('selected_layers',meta.get('layers'))==layers
    np.testing.assert_allclose(meta['alpha_grid'],np.linspace(0,3,alpha_steps),atol=3e-7)
    count=0
    for p in sorted(directory.glob('shard_*.npz')):
        with np.load(p) as data:
            drop=data['single_drop']; sub=data['substitution']
            assert drop.shape[1:]==(len(layers),heads)
            assert sub.shape==(*drop.shape,heads)
            assert np.isfinite(drop).all()
            assert np.isfinite(sub[...,~np.eye(heads,dtype=bool)]).all()
            count+=len(drop)
    assert count==samples, (directory,count,samples)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--family',choices=['Qwen','ViT'],required=True); a=p.parse_args()
    import numpy as np
    import fcntl
    lock=(ROOT/'gpu1_pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    entries=[]
    if a.family=='Qwen':
        contexts=ROOT/'data/fineweb_rebuttal_256x129.npz'
        data=np.load(contexts,allow_pickle=False)
        assert data['input_ids'].shape==(256,129)
        assert len(set(data['document_id'][:32]))==32, 'Independent document IDs required'
        input_hash=hashlib.sha256(data['input_ids'][:32].tobytes()).hexdigest()
        tokenizers=[ROOT/'models'/('Qwen2.5-'+size)/'tokenizer.json' for size in ['1.5B','3B','7B','14B']]
        assert len({hashlib.sha256(p.read_bytes()).hexdigest() for p in tokenizers})==1, 'Tokenizer definitions differ'
        (ROOT/'qwen_input_identity.json').write_text(json.dumps(dict(input_identity_sha256=input_hash,document_ids=data['document_id'][:32].tolist(),context_offsets=list(range(32))),indent=2))
        specs=[('1.5B',12,[6,13,20],1543714304),('3B',16,[8,17,26],3085938688),('7B',28,[6,13,20],7615616512),('14B',40,[11,23,35],14770033664)]
        for size,heads,layers,params in specs:
            name='qwen'+size.lower().replace('.','')
            model=ROOT/'models'/('Qwen2.5-'+size)
            out=ROOT/'outputs'/name
            common=['scripts/collect_qwen_cfs.py','--model',model,'--contexts',contexts,'--layers',','.join(map(str,layers)),'--alpha-steps','16','--variant-batch-size','32','--shard-size','1','--prediction-positions','random:8','--position-seed','20260802','--minimum-position','7']
            run(common+['--output-dir',ROOT/'smoke'/name,'--num-samples','1'],f'{name}_smoke.log')
            check_collection(ROOT/'smoke'/name,heads,layers,16,1)
            run(common+['--output-dir',out,'--num-samples','32'],f'{name}_collect.log')
            check_collection(out,heads,layers,16,32)
            entries.append(dict(name=name,family='Qwen',model_id=str(model),parameters=params,heads=heads,samples=32,layers=layers,N=[2,4,8,11],K=[4,8,12],path=str(out),input_identity_sha256=input_hash))
            manifest=ROOT/'qwen_manifest.json'; manifest.write_text(json.dumps(dict(models=entries),indent=2))
            target=ROOT/'analysis'/name
            if not (target/'summary.csv').exists():
                run(['scripts/analyze_matched_scaling.py','--manifest',manifest,'--model',name,'--output',ROOT/'analysis'],f'{name}_analyze.log')
    else:
        provenance=json.loads((ROOT/'imagenet_provenance.json').read_text())
        records=provenance['images']; assert len(records)==50000
        chosen=np.linspace(0,49999,512,dtype=int)
        input_hash=hashlib.sha256(json.dumps([records[i] for i in chosen],sort_keys=True).encode()).hexdigest()
        specs=[('small',6,11,22050664),('base',12,11,86567656),('large',16,23,304326632)]
        import timm
        from timm.data import resolve_model_data_config
        from prepare_rebuttal_remote import API, download_verified
        for size,heads,layer,params in specs:
            name='vit_'+size; model=f'vit_{size}_patch16_224.augreg_in21k_ft_in1k'
            modeldir=ROOT/'models'/model
            info=API.model_info('timm/'+model,files_metadata=True)
            download_verified('timm/'+model,info,modeldir,only=['model.safetensors','config.json'])
            import torch
            from safetensors.torch import load_file
            network=timm.create_model(model,pretrained=False)
            network.load_state_dict(load_file(str(modeldir/'model.safetensors')),strict=True)
            ckpt=modeldir/'collector_checkpoint.pt'
            if not ckpt.exists(): torch.save(dict(model=model,num_classes=1000,state_dict=network.state_dict()),ckpt)
            (modeldir/'provenance.json').write_text(json.dumps(dict(revision=info.sha,preprocessing=resolve_model_data_config(network)),indent=2))
            del network
            out=ROOT/'outputs'/name
            common=['scripts/collect_predictor_dataset.py','--model',model,'--checkpoint',ckpt,'--data-root',ROOT/'data/imagenet1k_val','--layers',str(layer),'--alpha-steps','31','--variant-batch-size','128','--shard-size','8','--selection','spread','--fp16']
            run(common+['--output-dir',ROOT/'smoke'/name,'--num-samples','1'],f'{name}_smoke.log')
            check_collection(ROOT/'smoke'/name,heads,[layer],31,1)
            run(common+['--output-dir',out,'--num-samples','512'],f'{name}_collect.log')
            check_collection(out,heads,[layer],31,512)
            entries.append(dict(name=name,family='ViT',model_id=model,parameters=params,heads=heads,samples=512,layers=[layer],N=[2,4,5],K=[4,6],path=str(out),input_identity_sha256=input_hash))
            manifest=ROOT/'vit_manifest.json'; manifest.write_text(json.dumps(dict(models=entries),indent=2))
            if not (ROOT/'analysis'/name/'summary.csv').exists():
                run(['scripts/analyze_matched_scaling.py','--manifest',manifest,'--model',name,'--output',ROOT/'analysis'],f'{name}_analyze.log')
    print('FAMILY_COMPLETE',a.family,flush=True)

if __name__=='__main__': main()
