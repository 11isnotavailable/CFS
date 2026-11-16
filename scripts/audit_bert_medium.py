import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,pandas as pd
from src.bert_medium import SPECS,P,make_model,flops,fingerprint

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);rows=[];hashes={}
    for name,(w,d,h) in SPECS.items():
        torch.manual_seed(42);model=make_model(name);digest=hashlib.sha256()
        for tensor in model.state_dict().values():digest.update(tensor.detach().numpy().tobytes())
        hashes[name]=digest.hexdigest()
        rows.append(dict(model=name,width=w,depth=d,heads=h,head_dim=w//h,
            parameters=sum(p.numel() for p in model.parameters()),initial_sha256=hashes[name],**flops(name)))
        del model
    assert hashes['B2']==hashes['B6']==hashes['B7']
    frame=pd.DataFrame(rows);frame.to_csv(a.output/'architectures.csv',index=False);frame.to_json(a.output/'architectures.json',orient='records',indent=2)
    frame.to_latex(a.output/'architectures.tex',index=False,float_format='%.0f')
    (a.output/'protocol.json').write_text(json.dumps(dict(protocol=P,specs=SPECS,sha256=fingerprint(),
        flop_convention='Forward matmul only, MAC=2, expected15% of126 eligible positions; excludes elementwise/norm/softmax/GELU/backward'),indent=2))
    print(frame[['model','parameters','width','depth','heads']].to_string(index=False))
if __name__=='__main__':main()
