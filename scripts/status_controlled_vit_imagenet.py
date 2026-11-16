"""Read-only compact status; no full image manifests printed."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args(); r=a.root
    shards=list((r/'data/train_manifests').glob('*.complete.json'))
    packed=list((r/'data/packed').glob('train_*.json'))
    overlaps=sum(len(json.loads(f.read_text()).get('validation_content_overlap',[])) for f in shards)
    result=dict(downloaded_shards=len(shards),total_shards=310,packed_shards=len(packed),
                validation_byte_overlap=overlaps,validation_pack_ready=(r/'data/packed/VAL_READY.json').exists())
    result['data_ready']=(r/'data/DATA_READY.json').exists()
    result['preprocessed_shards']=len(list((r/'data/preprocessed').glob('train_[0-9][0-9][0-9][0-9].json')))
    result['preprocessing_ready']=(r/'data/preprocessed/PREPROCESSED_READY.json').exists()
    for name in ['STATUS.json','BENCHMARK_PREPROCESSED_COMPLETE.json']:
        if (r/name).exists():
            value=json.loads((r/name).read_text()); value.pop('measured',None); result[name]=value
    result['models']={}
    for name in ['V1','V2','V3','V4','V5','V6','V7']:
        output=r/'runs'/name; value=dict(complete=(output/'COMPLETE.json').exists())
        if (output/'history.json').exists(): value['last_epoch']=json.loads((output/'history.json').read_text())[-1]
        result['models'][name]=value
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
