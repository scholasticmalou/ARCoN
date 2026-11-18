#!/usr/bin/env python3
"""Score one complete trajectory, or four corresponding resolutions."""
import argparse,json
from pathlib import Path
import numpy as np
from arcon.quality import from_documents,scores

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,nargs='+',required=True,
                   help='One or four directories containing geometry_quality.json and self_quality.json')
    p.add_argument('--position-consistency',type=Path,help='.npy vector in metres, required for four resolutions')
    p.add_argument('--domain',nargs=2,type=float,default=[.5,2.])
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if len(a.metrics) not in (1,4):p.error('Supply one or four metric directories')
    raw=[from_documents(x/'geometry_quality.json',x/'self_quality.json') for x in a.metrics]
    raw=raw[0] if len(raw)==1 else np.stack(raw)
    result=scores(raw,np.load(a.position_consistency) if a.position_consistency else None,tuple(a.domain))
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text)

if __name__=='__main__':main()
