"""Frozen BERT-Medium controlled family and exact input-token budget."""
import hashlib,json,math
from src.bert_throughput import make_from_shape

SPECS={'B1':(384,8,6),'B2':(512,8,8),'B3':(640,8,10),'B4':(512,6,8),
       'B5':(512,12,8),'B6':(512,8,4),'B7':(512,8,16)}
AXES={'width':['B1','B2','B3'],'depth':['B4','B2','B5'],'heads':['B6','B2','B7']}
P=dict(target_tokens=2000000000,checkpoints=[500000000,1000000000,1500000000,2000000000],
       batch=256,seq=128,seed=42,lr=3e-4,min_lr=3e-5,warmup_fraction=.02,optimizer='AdamW',
       betas=[.9,.999],eps=1e-8,weight_decay=.01,clip_grad=1.,dropout=.1,
       precision='BF16 autocast / FP32 parameters and optimizer',mask_probability=.15,
       vocabulary=30522,max_positions=512,validation_documents=2048,cfs_documents=256,probe_documents=64,
       cfs_positions=8,cfs_mask_seed=20260811,alpha_steps=16,alpha_min=0.,alpha_max=3.,
       valid_drop=1e-5,cover_threshold=.5,bootstrap=2000,fixed_k=4,subset_draws=500,
       primary_tokens=2000000000,automatic_extension=False,initialization='random',
       token_count='all128 non-padding positions including CLS/SEP;126 content positions',
       sampling='one pass through shared frozen WordPiece blocks in original seeded shard/document order')

def make_model(name,vocab=30522,dropout=.1): return make_from_shape(*SPECS[name],vocab,dropout,position_embeddings=512)
def fingerprint(): return hashlib.sha256(json.dumps(dict(specs=SPECS,protocol=P),sort_keys=True).encode()).hexdigest()
def learning_rate(tokens):
    warm=P['target_tokens']*P['warmup_fraction']
    if tokens<warm: return P['lr']*max(tokens,1)/warm
    progress=min(1.,(tokens-warm)/(P['target_tokens']-warm))
    return P['min_lr']+.5*(P['lr']-P['min_lr'])*(1+math.cos(math.pi*progress))
def update_sequences(tokens):
    boundary=next(x for x in P['checkpoints'] if x>tokens)
    return min(P['batch'],(boundary-tokens)//P['seq'])
def flops(name):
    w,d,h=SPECS[name]; t=128; masked=.15*126
    attention=d*(8*t*w*w+4*t*t*w); ffn=d*16*t*w*w
    mlm=2*masked*(w*w+w*30522)
    return dict(attention_forward_matmul_flops=attention,ffn_forward_matmul_flops=ffn,
        expected_sparse_mlm_forward_matmul_flops=mlm,expected_total_forward_matmul_flops=attention+ffn+mlm)
