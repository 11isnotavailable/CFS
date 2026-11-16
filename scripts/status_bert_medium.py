import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;status={}
    for name in ['STATUS.json','data/PREPARATION_STATUS.json']:
        if (r/name).exists():status[name]=json.loads((r/name).read_text())
    status['data_ready']=(r/'data/READY.json').exists();status['models']={}
    for name in ['B1','B2','B3','B4','B5','B6','B7']:
        entry=dict(training_complete=(r/'runs'/name/'COMPLETE.json').exists(),cfs_1B=(r/'cfs'/name/'1B/COMPLETE.json').exists(),cfs_2B=(r/'cfs'/name/'2B/COMPLETE.json').exists())
        history=r/'runs'/name/'history.json'
        if history.exists():entry['latest']=json.loads(history.read_text())[-1]
        status['models'][name]=entry
    print(json.dumps(status,indent=2))
if __name__=='__main__':main()
