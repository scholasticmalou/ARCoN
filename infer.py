#!/usr/bin/env python3
"""Run the reported ARCoN model on one complete native garment/motion input."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
from arcon.runtime import ROOT, build

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['data-root', 'sequence', 'garment-dict', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--garment', required=True, help='Key in the garment dictionary')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/arcon_25000.pth')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=-1, help='-1: complete trajectory; positive: diagnostic prefix only')
    p.add_argument('--external-root', type=Path)
    p.add_argument('--record-runtime-sources', action='store_true')
    args = p.parse_args()
    if args.output.exists(): p.error('Choose a new output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner, sequence, _ = build(args.data_root, args.checkpoint, args.sequence,
        args.garment_dict, args.garment, args.output.parent/'runtime', args.device,
        args.external_root)
    import torch
    rest = sequence['cloth'].rest_pos.detach().cpu().numpy()
    if rest.ndim == 3: rest = rest[:, 0]
    vertex_type = sequence['cloth'].vertex_type.detach().cpu().numpy()
    with torch.no_grad():
        result = runner.valid_rollout(sequence, n_steps=args.steps, bare=True)
    arrays = {key: result[key] for key in ['pred', 'obstacle', 'cloth_faces', 'obstacle_faces']}
    arrays.update(rest_pos=rest, vertex_type=vertex_type)
    np.savez_compressed(args.output, **arrays)
    meta = {'model':'ARCoN', 'frames':len(result['pred']),
        'complete_trajectory':args.steps < 0, 'units':'metres',
        'mesh_key':args.garment, 'native_resolution_adaptation':True}
    if args.record_runtime_sources:
        sources={}
        for name,module in list(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if path and name.startswith(('arcon.','models.','runners.','utils.','datasets.','criterions.')):
                try:sources[name]=Path(path).resolve().relative_to(ROOT).as_posix()
                except ValueError:sources[name]='OUTSIDE_PACKAGE'
        meta['runtime_sources']=sources
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta))

if __name__ == '__main__': main()
