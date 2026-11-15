"""Freeze public pretrained GPT-2 and disjoint FineWeb-Edu documents."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--endpoint',default='https://huggingface.co');p.add_argument('--parquet',type=Path)
    p.add_argument('--revision',default='607a30d783dfa663caf39e06633721c8d4cfcd7e')
    args=p.parse_args();root=args.root
    os.environ['HF_HUB_DISABLE_XET']='1'
    import requests
    from transformers import AutoTokenizer
    from datasets import load_dataset
    root.mkdir(parents=True,exist_ok=True);model=root/'model'
    if (root/'data').exists():raise FileExistsError('Frozen data already exists; use a new run root')
    revision=args.revision
    model.mkdir(exist_ok=True)
    for name in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','generation_config.json']:
        path=model/name
        if path.exists():continue
        response=requests.get(f'{args.endpoint}/openai-community/gpt2/resolve/{revision}/{name}',stream=True,timeout=120);response.raise_for_status()
        temporary=path.with_suffix(path.suffix+'.partial')
        with temporary.open('wb') as stream:
            for block in response.iter_content(1024*1024):stream.write(block)
        temporary.replace(path);print('downloaded',name,path.stat().st_size,flush=True)
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    if args.parquet:
        import pyarrow.parquet as pq
        ds_revision=None
        def documents():
            for batch in pq.ParquetFile(args.parquet).iter_batches(batch_size=256):
                yield from batch.to_pylist()
        ds=documents()
    else:
        from huggingface_hub import HfApi
        ds_revision=HfApi(endpoint=args.endpoint).dataset_info('HuggingFaceFW/fineweb-edu').sha
        ds=load_dataset('HuggingFaceFW/fineweb-edu','sample-10BT',split='train',streaming=True,revision=ds_revision)
    records=[];seen=set();seen_tokens=set()
    for index,row in enumerate(ds):
        text=row['text'];digest=hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:continue
        tokens=tokenizer.encode(text,add_special_tokens=False)
        if len(tokens)<129:continue
        prefix=np.asarray(tokens[:129],dtype=np.int64);key=hashlib.sha256(prefix.tobytes()).hexdigest()
        if key in seen_tokens:continue
        seen.add(digest);seen_tokens.add(key)
        records.append(dict(source_index=index,document_id=str(row.get('id',index)),sha256=digest,prefix_sha256=key,text=text,ids=prefix))
        if len(records)>=2432:break
    assert len(records)==2432
    order=np.random.default_rng(20260802).permutation(len(records));data=root/'data';data.mkdir(exist_ok=False)
    start=0;manifest={}
    for split,count in [('train',2048),('tune',128),('test',256)]:
        rows=[records[i] for i in order[start:start+count]];start+=count
        np.savez_compressed(data/f'{split}.npz',input_ids=np.stack([r['ids'] for r in rows]),
            document_id=np.array([r['document_id'] for r in rows]),source_index=np.array([r['source_index'] for r in rows]),
            document_sha256=np.array([r['sha256'] for r in rows]),prefix_sha256=np.array([r['prefix_sha256'] for r in rows]))
        with (data/f'{split}_documents.jsonl').open('w') as stream:
            for r in rows:stream.write(json.dumps({k:v for k,v in r.items() if k!='ids'})+'\n')
        manifest[split]=dict(count=count,file_sha256=hashlib.sha256((data/f'{split}.npz').read_bytes()).hexdigest())
    provenance=dict(model='openai-community/gpt2',revision=revision,dataset='HuggingFaceFW/fineweb-edu',config='sample-10BT',dataset_revision=ds_revision,
        seed=20260802,prefix_tokens=128,prediction_position=127,target_position=128,one_position_per_document=True,
        document_disjoint=True,unique_prefixes=True,splits=manifest,
        download_endpoint=args.endpoint,source_parquet=str(args.parquet),
        source_parquet_sha256=hashlib.sha256(args.parquet.read_bytes()).hexdigest() if args.parquet else None,
        dataset_revision_note='Existing cached public parquet; revision not recorded historically' if args.parquet else None,
        model_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model.glob('*') if p.is_file()})
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
