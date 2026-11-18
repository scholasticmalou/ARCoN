#!/usr/bin/env python3
"""Train ARCoN from initialization through its documented paper course, or resume it."""
import argparse
import json
from pathlib import Path
from arcon.runtime import ROOT


def main():
    course=json.loads((ROOT/'configs/training_course.json').read_text())
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,help='Authorized native data tree described in README.md')
    p.add_argument('--external-root',type=Path,help='Dependency workspace containing external/CCCollisions-torch211-compat and CUDA sample headers')
    p.add_argument('--output',type=Path,help='New output directory; existing run artifacts are never replaced')
    p.add_argument('--resume',type=Path,help='Optional full checkpoint produced by this training course; not the inference-only 25k weights')
    p.add_argument('--stage',choices=['all']+[s['name'] for s in course['stages']],default='all')
    p.add_argument('--until-step',type=int,default=25000,help='Absolute cumulative data iteration at which to stop (default: complete 25k course)')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-seconds',type=int,help='Optional per-stage wall-time limit')
    p.add_argument('--print-plan',action='store_true',help='Print the training stages without importing CUDA dependencies')
    args=p.parse_args()
    if args.print_plan:
        print(json.dumps(course,indent=2));return
    for key in ['data_root','external_root','output']:
        if getattr(args,key) is None:p.error('--'+key.replace('_','-')+' is required for training')
    if not 0<args.until_step<=25000:p.error('--until-step must be in 1..25000')
    if args.max_seconds is not None and args.max_seconds<=0:p.error('--max-seconds must be positive')
    if args.output.exists():p.error('Choose a new --output directory; resume into a new output location')
    from arcon.training import run
    run(args,course)


if __name__=='__main__':main()
