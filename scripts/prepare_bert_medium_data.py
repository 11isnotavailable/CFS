"""Freeze exact2B WordPiece input positions using original FineWeb document split."""
import argparse,concurrent.futures,gzip,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
import requests
from transformers import BertTokenizerFast
from src.bert_medium import P,fingerprint
from src.bert_throughput import validation_document

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--endpoint',default='https://hf-mirror.com');a=p.parse_args()
    root=a.root/'data';root.mkdir(exist_ok=True);source=root/'source';source.mkdir(exist_ok=True)
    if (root/'READY.json').exists():return
    metadata=json.loads(a.metadata.read_text()); assert metadata['dataset']=='HuggingFaceFW/fineweb'
    tokenizer=BertTokenizerFast.from_pretrained(a.tokenizer,local_files_only=True)
    tokenizer.model_max_length=10000000
    assert tokenizer.vocab_size==P['vocabulary']
    files={r['path']:r for r in metadata['source_files']};order=metadata['seeded_source_order']
    def download(name):
        info=files[name];dest=source/Path(name).name;done=dest.with_suffix('.verified.json')
        if done.exists() and dest.exists() and dest.stat().st_size==info['size']:return dest
        if dest.exists() and dest.stat().st_size==info['size'] and sha(dest)==info['sha256']:
            atomic(done,info);return dest
        url=f"{a.endpoint}/datasets/{metadata['dataset']}/resolve/{metadata['revision']}/{name}?download=true"
        for attempt in range(3):
            try:
                digest=hashlib.sha256();tmp=dest.with_suffix('.partial')
                with requests.get(url,stream=True,timeout=(30,180)) as response:
                    response.raise_for_status()
                    with tmp.open('wb') as f:
                        for b in response.iter_content(4<<20):f.write(b);digest.update(b)
                assert tmp.stat().st_size==info['size'] and digest.hexdigest()==info['sha256'],'Source SHA256 mismatch'
                tmp.replace(dest);atomic(done,info);print('DOWNLOADED_VERIFIED',name,flush=True);return dest
            except Exception:
                if attempt==2:raise
                time.sleep(5)
    total=0;documents=0;train_docs=0;val=[];seen=set();shards=[];began=time.time()
    target=P['target_tokens']//128
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(download,order[0])
        for sid,name in enumerate(order):
            marker=root/f'part_{sid:02d}.json'
            if marker.exists():
                state=json.loads(marker.read_text());total=state['total_sequences'];documents=state['processed_documents'];train_docs=state['train_documents']
                val=json.loads((root/f'validation_after_{sid:02d}.json').read_text());seen={r['sha256'] for r in val};shards.append(state)
                if total==target and len(val)==P['validation_documents']:break
                future=pool.submit(download,order[sid+1]);continue
            path=future.result()
            if sid+1<len(order):future=pool.submit(download,order[sid+1])
            train_path=root/f'train_{sid:02d}.bin';doc_path=root/f'doc_ids_{sid:02d}.bin'
            part_count=0;first=total;row_offset=0
            with train_path.with_suffix('.tmp').open('wb') as tokens,doc_path.with_suffix('.tmp').open('wb') as ids,gzip.open(root/f'documents_{sid:02d}.jsonl.gz','wt') as manifest:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=['text'],use_threads=False):
                    texts=batch.column(0).to_pylist();splits=[validation_document(t) for t in texts]
                    active=[i for i,v in enumerate(splits) if (not v and total<target) or (v and len(val)<P['validation_documents'])]
                    encodings=tokenizer([texts[i] for i in active],add_special_tokens=False)['input_ids'] if active else []
                    mapped=dict(zip(active,encodings));chunks=[];doc_chunks=[]
                    for i,text in enumerate(texts):
                        docid=documents;documents+=1
                        encoded=mapped.get(i)
                        if encoded is None:continue
                        digest=hashlib.sha256(text.encode()).hexdigest()
                        if splits[i]:
                            if len(encoded)>=126 and sum(t not in [0,100,101,102,103] for t in encoded[:126])>=8 and digest not in seen and len(val)<P['validation_documents']:
                                val.append(dict(sha256=digest,source=name,row=row_offset+i,document_id=docid,
                                    input_ids=[101]+encoded[:126]+[102]))
                                seen.add(digest)
                        elif total<target:
                            count=min(len(encoded)//126,target-total)
                            if count:
                                array=np.empty((count,128),dtype='<u2');array[:,0]=101;array[:,-1]=102
                                array[:,1:-1]=np.asarray(encoded[:count*126],dtype='<u2').reshape(count,126)
                                chunks.append(array);doc_chunks.append(np.full(count,docid,dtype='<u4'))
                                total+=count;part_count+=count;train_docs+=1
                                manifest.write(json.dumps(dict(document_id=docid,sha256=digest,source=name,row=row_offset+i,sequences=count))+'\n')
                    if chunks:tokens.write(np.concatenate(chunks).tobytes());ids.write(np.concatenate(doc_chunks).tobytes())
                    row_offset+=len(texts)
                    if row_offset%10240==0:
                        atomic(root/'PREPARATION_STATUS.json',dict(input_tokens=total*128,target=P['target_tokens'],validation_documents=len(val),processed_documents=documents,source=name,seconds=time.time()-began))
                        print('TOKENIZED',total*128,'VALIDATION',len(val),'DOCS',documents,flush=True)
                    if total==target and len(val)==P['validation_documents']:break
            train_path.with_suffix('.tmp').replace(train_path);doc_path.with_suffix('.tmp').replace(doc_path)
            state=dict(source=name,source_sha256=files[name]['sha256'],sequences=part_count,first_sequence=first,
                total_sequences=total,processed_documents=documents,train_documents=train_docs,
                tokens_file=train_path.name,doc_ids_file=doc_path.name,tokens_sha256=sha(train_path),doc_ids_sha256=sha(doc_path))
            atomic(root/f'validation_after_{sid:02d}.json',val);atomic(marker,state);shards.append(state)
            if total==target and len(val)==P['validation_documents']:break
    assert total==target and len(val)==P['validation_documents']
    # Fixed training-style validation masks, independent from model/dropout RNG.
    import torch
    from src.bert_throughput import mask_tokens
    clean=torch.tensor([v['input_ids'] for v in val],dtype=torch.long)
    masked,labels=mask_tokens(clean,generator=torch.Generator().manual_seed(20261116))
    np.savez_compressed(root/'validation.npz',input_ids=masked.numpy(),labels=labels.numpy(),
        clean_input_ids=clean.numpy(),document_sha256=np.asarray([v['sha256'] for v in val]),document_id=np.asarray([v['document_id'] for v in val]))
    atomic(root/'validation_documents.json',val)
    # Original MLM CFS protocol: exactly8 fixed MASK positions, identical models.
    chosen=val[:P['cfs_documents']];inputs=[];positions=[];targets=[]
    for index,v in enumerate(chosen):
        clean=np.asarray(v['input_ids']);eligible=np.flatnonzero(~np.isin(clean,[0,100,101,102,103]))
        at=np.sort(np.random.default_rng(P['cfs_mask_seed']+index).choice(eligible,8,replace=False))
        item=clean.copy();item[at]=103;inputs.append(item);positions.append(at);targets.append(clean[at])
    np.savez_compressed(root/'cfs_inputs.npz',input_ids=np.asarray(inputs),positions=np.asarray(positions),targets=np.asarray(targets),
        document_sha256=np.asarray([v['sha256'] for v in chosen]),sample_id=np.arange(len(chosen)))
    manifest=dict(dataset=metadata['dataset'],revision=metadata['revision'],subset=metadata['subset'],original_split=metadata['split'],
        protocol_sha256=fingerprint(),sequence_count=total,input_tokens=total*128,content_tokens=total*126,
        training_documents=train_docs,processed_documents=documents,validation_documents=len(val),
        validation_unique_hashes=len(seen),shards=shards,tokenizer_vocab_sha256=sha(a.tokenizer/'vocab.txt'),
        validation_sha256=sha(root/'validation.npz'),cfs_inputs_sha256=sha(root/'cfs_inputs.npz'),
        packing='126 WordPiece content tokens within each document plus CLS/SEP; discard document tails; no padding',
        test_split=None,seconds=time.time()-began)
    atomic(root/'READY.json',manifest);print('DATA_READY',json.dumps(manifest),flush=True)

if __name__=='__main__':main()
