#!/usr/bin/env python3
"""Measure the six paper quantities on a saved rollout, without rerunning a model."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from arcon.geometry import weights_and_areas,rest_face_inverse,intrinsic_green,persistence
from arcon.runtime import ROOT

def geometry(data,device):
    from arcon.control.surface_measurements import SurfaceDistance
    import warp as wp
    wp.init()
    faces=data['cloth_faces'].astype(np.int64);rest=data['rest_pos'].astype(np.float64)
    weights,area=weights_and_areas(rest,faces);inverse=rest_face_inverse(rest,faces)
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    lengths=np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1)
    surface=SurfaceDistance(data['obstacle'][0],data['obstacle_faces'],device)
    fields={k:[] for k in ['penetration_fraction','mean_penetration_depth','green_strain_area_rms','edge_stretch_rms']}
    for i,q in enumerate(data['pred']):
        surface.update(data['obstacle'][i]);depth=np.maximum(-surface.query(q),0.)
        green=intrinsic_green(q.astype(np.float64)[None],faces,inverse)[0]
        square=np.sum(green*green,axis=(1,2))
        stretch=np.linalg.norm(q[edges[:,1]].astype(np.float64)-q[edges[:,0]],axis=1)/lengths-1
        for k,v in {'penetration_fraction':float((depth>1e-6)@weights),
            'mean_penetration_depth':float(depth@weights),
            'green_strain_area_rms':float(np.sqrt(square@area/area.sum())),
            'edge_stretch_rms':float(np.sqrt(np.mean(stretch**2)))}.items():fields[k].append(v)
    return fields

def self_contact(data,device):
    from arcon.detector import find_close_faces
    import torch
    faces=data['cloth_faces'].astype(np.int64)
    _,area=weights_and_areas(data['rest_pos'].astype(np.float64),faces);area/=area.sum()
    ftorch=torch.as_tensor(faces,device=device);affected=[];fields={'affected_rest_area_fraction':[]}
    for q in data['pred']:
        vertices=torch.as_tensor(q,device=device,dtype=torch.float64)
        pairs=find_close_faces(vertices,ftorch,threshold=0.).detach().cpu().numpy()
        if len(pairs):
            pairs=np.unique(np.sort(pairs,axis=1),axis=0)
            adjacent=(faces[pairs[:,0],:,None]==faces[pairs[:,1],None,:]).any(axis=(1,2))
            pairs=pairs[~adjacent]
        mask=np.zeros(len(faces),dtype=bool)
        if len(pairs):mask[np.unique(pairs)]=True
        affected.append(mask);fields['affected_rest_area_fraction'].append(float(area[mask].sum()))
    _,series=persistence(np.asarray(affected),area)
    fields['persistent10_area_fraction']=series['persistent10_area_fraction']
    return fields

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--part',choices=['all','geometry','self'],default='all',
        help='Separate modes support the original native and collision-detector environments')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    parts=['geometry','self'] if a.part=='all' else [a.part]
    names={'geometry':'geometry_quality.json','self':'self_quality.json'}
    for part in parts:
        if (a.output/names[part]).exists():p.error('A requested metric output already exists')
    import torch
    torch.cuda.set_device(a.device);torch.set_num_threads(4)
    with np.load(a.rollout,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
    for part in parts:
        values=(geometry if part=='geometry' else self_contact)(data,a.device)
        (a.output/names[part]).write_text(json.dumps({'state':'complete','frames':len(data['pred']),
            'values':values},indent=2)+'\n')
    print(json.dumps({'frames':len(data['pred']),'parts':parts}))

if __name__=='__main__':main()
