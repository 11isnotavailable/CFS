"""Exact shared2B-token schedule, fixed validation, four milestones, atomic resume."""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.utils.data import DataLoader,IterableDataset,get_worker_info
from src.bert_medium import SPECS,P,make_model,fingerprint,learning_rate,update_sequences
from src.bert_throughput import mask_tokens,sparse_mlm_loss

def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def save(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

class Stream(IterableDataset):
    def __init__(self,root,metadata,start):self.root=root;self.parts=[p for p in metadata['shards'] if p['sequences']];self.start=start
    def __iter__(self):
        worker=get_worker_info();wid=worker.id if worker else 0;workers=worker.num_workers if worker else 1
        ends=np.cumsum([p['sequences'] for p in self.parts]);starts=np.r_[0,ends[:-1]]
        tokens=[np.memmap(self.root/p['tokens_file'],mode='r',dtype='<u2').reshape(-1,128) for p in self.parts]
        docs=[np.memmap(self.root/p['doc_ids_file'],mode='r',dtype='<u4') for p in self.parts]
        cursor=self.start;step=0
        while cursor<P['target_tokens']:
            n=update_sequences(cursor)
            if step%workers==wid:
                begin=cursor//128;finish=begin+n;chunks=[];identities=[]
                while begin<finish:
                    sid=int(np.searchsorted(ends,begin,side='right'));stop=min(finish,int(ends[sid]))
                    chunks.append(np.array(tokens[sid][begin-starts[sid]:stop-starts[sid]],dtype=np.int64))
                    identities.append(np.array(docs[sid][begin-starts[sid]:stop-starts[sid]],dtype=np.int64));begin=stop
                yield torch.from_numpy(np.concatenate(chunks)),torch.from_numpy(np.concatenate(identities))
            cursor+=n*128;step+=1

@torch.no_grad()
def validate(model,validation,output=None):
    model.eval();loss_sum=torch.zeros((),device='cuda');correct=loss_sum.clone();count=0
    per=[]
    for begin in range(0,len(validation['input_ids']),128):
        ids=torch.tensor(validation['input_ids'][begin:begin+128],device='cuda');labels=torch.tensor(validation['labels'][begin:begin+128],device='cuda')
        chosen=labels!=-100
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=model.bert(input_ids=ids,return_dict=True).last_hidden_state
            logits=model.cls(hidden[chosen]).float()
        losses=torch.nn.functional.cross_entropy(logits,labels[chosen],reduction='none')
        accuracy=(logits.argmax(-1)==labels[chosen]).float();which=torch.nonzero(chosen,as_tuple=True)[0]
        row_loss=torch.zeros(len(ids),device='cuda').scatter_add_(0,which,losses)
        row_correct=torch.zeros(len(ids),device='cuda').scatter_add_(0,which,accuracy)
        row_count=chosen.sum(-1);per.append(torch.stack([row_loss,row_correct,row_count]).cpu().numpy())
        loss_sum+=losses.sum();correct+=accuracy.sum();count+=len(losses)
    torch.cuda.synchronize()
    if output is not None:
        values=np.concatenate(per,axis=1)
        np.savez_compressed(output,loss_sum=values[0],correct=values[1],token_count=values[2],document_sha256=validation['document_sha256'])
    return dict(validation_ce=float(loss_sum/count),validation_accuracy=float(correct/count),validation_masked_positions=count)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=SPECS,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-updates',type=int,help='Operational smoke/pause only; schedule remains2B')
    a=p.parse_args();data=a.root/'data';ready=json.loads((data/'READY.json').read_text());assert ready['protocol_sha256']==fingerprint()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=make_model(a.model).cuda();opt=torch.optim.AdamW([
        {'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':P['weight_decay']},
        {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0}],lr=P['lr'],betas=tuple(P['betas']),eps=P['eps'],fused=True)
    masking=torch.Generator(device='cuda').manual_seed(4242)
    tokens=step=targets=visits=0;lastdoc=-1;history=[];gpu_seconds=0.;phase_loss=phase_count=0.;phase_correct=torch.zeros((),device='cuda')
    checkpoint=a.output/'last.pt'
    if checkpoint.exists():
        if not a.resume:raise RuntimeError('Existing run requires explicit resume')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['protocol_hash']==fingerprint() and state['model']==a.model and state['data_hash']==hashlib.sha256((data/'READY.json').read_bytes()).hexdigest()
        model.load_state_dict(state['weights']);opt.load_state_dict(state['optimizer']);tokens=state['tokens'];step=state['step'];targets=state['targets'];visits=state['document_visits'];lastdoc=state['last_document'];history=state['history'];gpu_seconds=state['gpu_seconds']
        torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state(state['rng_cuda']);masking.set_state(state['mask_rng'])
    elif a.resume and (a.output/'run.json').exists():print('REPLAY_FROM_INITIALIZATION',flush=True)
    data_hash=hashlib.sha256((data/'READY.json').read_bytes()).hexdigest()
    write_json(a.output/'run.json',dict(model=a.model,spec=SPECS[a.model],protocol=P,protocol_hash=fingerprint(),data_hash=data_hash,
        parameters=sum(p.numel() for p in model.parameters()),from_scratch=True,benchmark_weights_reused=False))
    validation=np.load(data/'validation.npz');start_tokens=tokens;start_step=step;started=time.perf_counter()
    def persist():
        nonlocal gpu_seconds,started
        torch.cuda.synchronize();gpu_seconds+=time.perf_counter()-started;started=time.perf_counter()
        save(checkpoint,dict(model=a.model,weights=model.state_dict(),optimizer=opt.state_dict(),tokens=tokens,step=step,
            targets=targets,document_visits=visits,last_document=lastdoc,history=history,gpu_seconds=gpu_seconds,
            protocol_hash=fingerprint(),data_hash=data_hash,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),mask_rng=masking.get_state()))
        write_json(a.output/'history.json',history)
    def export():
        dest=a.output/f'model_{tokens}.pt'
        if not dest.exists():save(dest,dict(model=model.state_dict(),tokens=tokens,model_id=a.model,protocol_hash=fingerprint(),data_hash=data_hash))
    if tokens in P['checkpoints']:export()
    if step==0:
        metrics=validate(model,validation);assert math.isfinite(metrics['validation_ce'])
        history.append(dict(tokens=0,step=0,learning_rate=learning_rate(0),**metrics));write_json(a.output/'history.json',history)
    loader=DataLoader(Stream(data,ready,tokens),batch_size=None,num_workers=2,pin_memory=True,prefetch_factor=4)
    model.train();last_log=time.perf_counter();log_tokens=tokens;input_wait=0.;next_started=time.perf_counter()
    for ids,document_ids in loader:
        input_wait+=time.perf_counter()-next_started
        n=len(ids);assert n==update_sequences(tokens)
        doc=document_ids.numpy();visits+=int(np.count_nonzero(doc[1:]!=doc[:-1]))+int(doc[0]!=lastdoc);lastdoc=int(doc[-1])
        ids=ids.cuda(non_blocking=True);corrupted,labels=mask_tokens(ids,generator=masking)
        lr=learning_rate(tokens+n*128)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,count,correct=sparse_mlm_loss(model,corrupted,labels,metrics=True)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),P['clip_grad'],error_if_nonfinite=True);opt.step()
        phase_loss+=float(loss.detach())*count;phase_count+=count;phase_correct+=correct.detach();targets+=count;tokens+=n*128;step+=1
        if step%250==0 or tokens in P['checkpoints']:
            torch.cuda.synchronize();now=time.perf_counter()
            row=dict(tokens=tokens,step=step,train_ce=phase_loss/phase_count,train_accuracy=float(phase_correct/phase_count),
                learning_rate=lr,masked_positions=targets,document_visits=visits,content_tokens=tokens//128*126,
                input_tokens_per_second=(tokens-log_tokens)/(now-last_log),gpu_hours=(gpu_seconds+now-started)/3600,
                host_input_wait_seconds=input_wait,host_input_wait_fraction=input_wait/(now-last_log),
                grad_norm=float(norm),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            if step%1000==0 or tokens in P['checkpoints']:
                metrics=validate(model,validation,a.output/f'validation_{tokens}.npz' if tokens in P['checkpoints'] else None);assert math.isfinite(metrics['validation_ce']);row.update(metrics);model.train()
            assert math.isfinite(row['train_ce']);history.append(row);write_json(a.output/'history.json',history)
            print(json.dumps(row),flush=True);phase_loss=phase_count=0;phase_correct.zero_();last_log=time.perf_counter();log_tokens=tokens;input_wait=0.
        if step%1000==0 or tokens in P['checkpoints']:
            persist()
            if tokens in P['checkpoints']:export()
        if a.stop_after_updates and step-start_step>=a.stop_after_updates:
            persist();write_json(a.output/'PAUSED.json',dict(tokens=tokens,step=step));return
        next_started=time.perf_counter()
    assert tokens==P['target_tokens']
    write_json(a.output/'COMPLETE.json',dict(model=a.model,tokens=tokens,step=step,mlm_targets=targets,document_visits=visits,
        gpu_hours=gpu_seconds/3600,protocol_hash=fingerprint(),data_hash=data_hash,**{k:v for k,v in history[-1].items() if k.startswith('validation_')}))

if __name__=='__main__':main()
