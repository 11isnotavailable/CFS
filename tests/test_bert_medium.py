import numpy as np
import torch
from src.bert_medium import P,update_sequences,learning_rate,make_model
from src.bert_throughput import mask_tokens

def test_exact_budget_and_milestone_alignment():
    count=0;seen=[];steps=0;tails=[]
    while count<P['target_tokens']:
        n=update_sequences(count);assert 0<n<=256
        count+=n*128;steps+=1
        if n<256:tails.append(n)
        if count in P['checkpoints']:seen.append(count)
    assert count==2000000000 and seen==P['checkpoints'] and steps==61036
    assert tails==[202]*4
    assert learning_rate(40000000)==P['lr'] and learning_rate(2000000000)==P['min_lr']

def test_mask_rng_is_independent_of_architecture_rng():
    tokens=torch.full((16,128),104,dtype=torch.long);tokens[:,0]=101;tokens[:,-1]=102
    g1=torch.Generator().manual_seed(4242);g2=torch.Generator().manual_seed(4242)
    a,la=mask_tokens(tokens,generator=g1)
    torch.rand(12345) # Different architectures consume different dropout RNG.
    b,lb=mask_tokens(tokens,generator=g2)
    assert torch.equal(a,b) and torch.equal(la,lb)

def test_compatible_bert_gate_preserves_dense_and_scales_only_selected_head():
    from src.bert_medium_intervention import CompatibleBertGate
    class Attention(torch.nn.Module):
        num_attention_heads=4;attention_head_size=2
        def forward(self,hidden,**kwargs):return (hidden+1,)
    gate=CompatibleBertGate(Attention());x=torch.arange(16,dtype=torch.float32).reshape(1,2,8)
    assert torch.equal(gate(x)[0],x+1)
    scale=torch.ones(1,4);scale[0,1]=0;scale[0,2]=2;gate.set_head_scales(scale)
    out=gate(x)[0];assert torch.equal(out[:,:,:2],x[:,:,:2]+1)
    assert torch.count_nonzero(out[:,:,2:4])==0
    assert torch.equal(out[:,:,4:6],2*(x[:,:,4:6]+1))
    assert torch.equal(out[:,:,6:],x[:,:,6:]+1)

def test_stream_crosses_shards_and_resumes_without_reordering(tmp_path,monkeypatch):
    from scripts.train_bert_medium import Stream
    monkeypatch.setitem(P,'target_tokens',512);monkeypatch.setitem(P,'checkpoints',[256,512])
    parts=[]
    for i in range(2):
        values=np.repeat(np.arange(2*i,2*i+2,dtype='<u2')[:,None],128,axis=1)
        values.tofile(tmp_path/f'train{i}.bin');np.arange(2*i,2*i+2,dtype='<u4').tofile(tmp_path/f'doc{i}.bin')
        parts.append(dict(sequences=2,tokens_file=f'train{i}.bin',doc_ids_file=f'doc{i}.bin'))
    all_rows=list(Stream(tmp_path,dict(shards=parts),0));resumed=list(Stream(tmp_path,dict(shards=parts),256))
    assert len(all_rows)==2 and len(resumed)==1
    assert torch.equal(all_rows[1][0],resumed[0][0])
    assert torch.cat([r[1] for r in all_rows]).tolist()==[0,1,2,3]

def test_bert_oracle_preserves_original_mean_over_inputs(tmp_path):
    from scripts.analyze_bert_medium import summarize
    source=tmp_path/'source';source.mkdir()
    drop=np.array([[1,0,0,0],[1,1,1,1]],dtype=float)
    sub=drop[:,:,None]*np.array([.8,.2])[:,None,None]*np.ones((2,4,4))
    np.savez(source/'shard_00000.npz',dataset_index=[0,1],single_drop=drop[:,None],substitution=sub[:,None])
    summary,_,_=summarize('B6',source,tmp_path/'analysis',full=False)
    oracle=summary.set_index('metric').loc['raw','mean']
    assert np.isclose(oracle,.5) # Not the valid-source-weighted global ratio .68.
