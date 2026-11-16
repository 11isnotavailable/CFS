"""Randomly initialized BERT MLM; project only the supervised positions."""
import hashlib
import torch
from transformers import BertConfig, BertForMaskedLM

SPECS={'B1':(192,12,3),'B2':(256,12,4),'B3':(320,12,5),'B5':(256,18,4)}

def make_model(name, vocab=30522, dropout=.1):
    w,d,h=SPECS[name]
    return make_from_shape(w,d,h,vocab,dropout)

def make_from_shape(w,d,h,vocab=30522,dropout=.1,position_embeddings=128):
    config=BertConfig(vocab_size=vocab,hidden_size=w,num_hidden_layers=d,num_attention_heads=h,
        intermediate_size=4*w,max_position_embeddings=position_embeddings,type_vocab_size=2,
        hidden_dropout_prob=dropout,attention_probs_dropout_prob=dropout,
        hidden_act='gelu',layer_norm_eps=1e-12,pad_token_id=0)
    config._attn_implementation='sdpa'
    return BertForMaskedLM(config)

def validation_document(text):
    digest=hashlib.blake2b(b'42\0'+text.encode('utf-8'),digest_size=8).digest()
    return int.from_bytes(digest,'little')%1000==0

def mask_tokens(ids, vocab=30522,generator=None):
    eligible=(ids!=0)&(ids!=100)&(ids!=101)&(ids!=102)&(ids!=103)
    selected=(torch.rand(ids.shape,device=ids.device,generator=generator)<.15)&eligible
    labels=ids.masked_fill(~selected,-100)
    action=torch.rand(ids.shape,device=ids.device,generator=generator)
    corrupted=torch.where(selected&(action<.8),103,ids)
    random_ids=torch.randint(vocab,ids.shape,device=ids.device,generator=generator)
    corrupted=torch.where(selected&(action>=.8)&(action<.9),random_ids,corrupted)
    return corrupted,labels

def sparse_mlm_loss(model,ids,labels,metrics=False):
    hidden=model.bert(input_ids=ids,return_dict=True).last_hidden_state
    selected=labels!=-100
    # The prediction transform is pointwise, so gather BEFORE the transform,
    # tied vocabulary projection and CE. No dense [batch,sequence,vocab] tensor.
    logits=model.cls(hidden[selected])
    loss=torch.nn.functional.cross_entropy(logits.float(),labels[selected])
    if metrics: return loss,logits.shape[0],(logits.argmax(-1)==labels[selected]).sum()
    return loss,logits.shape[0]
