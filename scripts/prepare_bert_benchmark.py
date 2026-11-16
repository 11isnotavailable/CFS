"""Bounded WordPiece cache from the original GPT-2 FineWeb revision; no full download."""
import argparse,hashlib,io,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import requests
import pyarrow.parquet as pq
from transformers import BertTokenizerFast
from src.bert_throughput import validation_document

class RangeReader(io.RawIOBase):
    def __init__(self,url,size): self.url=url; self.size=size; self.pos=0; self.bytes=0; self.requests=[]
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        if n<0: n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0: return b''
        if self.bytes+n>200*1024**2: raise RuntimeError('200MiB HTTP range budget reached; never download entire dataset')
        start=self.pos; end=start+n-1
        # Unique query avoids proxies incorrectly reusing a different cached range.
        url=self.url+f'?download=true&range_start={start}&range_end={end}'
        with requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=60) as response:
            response.raise_for_status()
            if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                raise RuntimeError('Server did not honor requested range; refuse full download')
            payload=response.raw.read(n+1)
        if len(payload)!=n: raise RuntimeError('Unexpected range length')
        self.bytes+=n; self.pos+=n
        self.requests.append(dict(start=start,end=end,sha256=hashlib.sha256(payload).hexdigest()))
        return payload

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--gpt2-metadata',type=Path,required=True)
    p.add_argument('--endpoint',default='https://huggingface.co')
    a=p.parse_args(); out=a.root/'data'; out.mkdir(parents=True,exist_ok=True)
    if (out/'READY.json').exists(): return
    metadata=json.loads(a.gpt2_metadata.read_text()); assert metadata['dataset']=='HuggingFaceFW/fineweb'
    token_dir=a.root/'tokenizer'; token_dir.mkdir(exist_ok=True)
    revision='86b5e0934494bd15c9632b12f734a8a67f723594'; vocab=token_dir/'vocab.txt'
    if not vocab.exists():
        response=requests.get(f'{a.endpoint}/google-bert/bert-base-uncased/resolve/{revision}/vocab.txt',timeout=60)
        response.raise_for_status(); assert len(response.content)<2*1024**2; vocab.write_bytes(response.content)
    tokenizer=BertTokenizerFast(vocab_file=str(vocab),do_lower_case=True,model_max_length=1000000)
    assert tokenizer.vocab_size==30522 and tokenizer.cls_token_id==101 and tokenizer.mask_token_id==103
    tokenizer.save_pretrained(token_dir)
    path=metadata['seeded_source_order'][0]
    source=next(row for row in metadata['source_files'] if row['path']==path)
    reader=RangeReader(f"{a.endpoint}/datasets/{metadata['dataset']}/resolve/{metadata['revision']}/{path}",source['size'])
    parquet=pq.ParquetFile(reader); blocks=[]; docids=[]; documents=[]; holdout=0; start=time.time()
    for group in range(min(4,parquet.num_row_groups)):
        texts=parquet.read_row_group(group,columns=['text'],use_threads=False).column(0).to_pylist()
        for text in texts:
            if len(documents)>=2048: break
            digest=hashlib.sha256(text.encode()).hexdigest(); validation=validation_document(text)
            documents.append(dict(document_sha256=digest,row_group=group,split='validation' if validation else 'train',text=text))
            if validation: holdout+=1; continue
            ids=tokenizer.encode(text,add_special_tokens=False)
            for offset in range(0,len(ids)-125,126):
                if len(blocks)>=16384: break
                blocks.append([101]+ids[offset:offset+126]+[102]); docids.append(digest)
        if len(documents)>=2048 or len(blocks)>=16384: break
    assert len(blocks)>=1024, f'Insufficient bounded cache: {len(blocks)}'
    np.save(out/'train_tokens.npy',np.asarray(blocks,dtype=np.uint16))
    np.save(out/'block_document_sha256.npy',np.asarray(docids))
    with (out/'documents.jsonl').open('w') as f:
        for row in documents: f.write(json.dumps(row)+'\n')
    manifest=dict(dataset=metadata['dataset'],revision=metadata['revision'],subset=metadata['subset'],source_file=path,
        full_source_expected_sha256=source['sha256'],full_source_hash_verified=False,
        partial_source_integrity='Pinned revision + hashes of each downloaded HTTP range; full shard not downloaded',
        bytes_downloaded=reader.bytes,ranges=reader.requests,original_split=metadata['split'],
        original_test_split=None,original_test_split_note='Original GPT-2 preparation has train/validation only; no test split invented',
        documents_read=len(documents),validation_documents_excluded=holdout,training_blocks=len(blocks),
        input_tokens=len(blocks)*128,text_tokens=len(blocks)*126,sequence_length=128,
        packing='126 WordPiece tokens within one training document + CLS/SEP; discard short tails for throughput only',
        tokenizer='google-bert/bert-base-uncased',tokenizer_revision=revision,vocab_size=30522,
        vocab_sha256=hashlib.sha256(vocab.read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256((out/'train_tokens.npy').read_bytes()).hexdigest(),
        seconds=time.time()-start,purpose='Small repeatedly sampled throughput cache only; not a formal training split')
    (out/'READY.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest),flush=True)

if __name__=='__main__': main()
