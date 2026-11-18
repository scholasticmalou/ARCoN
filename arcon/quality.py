"""Frozen JAR/JCV definitions used in the submitted paper; fractions in, points out."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FIELDS=('body_area','body_depth_m','green_global','edge_global','self_area','persistent10')
GROUPS={'B':[0,1],'M':[2,3],'S':[4,5]}

def from_documents(geometry, self_contact):
    if isinstance(geometry,(str,Path)):geometry=json.loads(Path(geometry).read_text())
    if isinstance(self_contact,(str,Path)):self_contact=json.loads(Path(self_contact).read_text())
    g,s=geometry['values'],self_contact['values']
    return np.column_stack([g['penetration_fraction'],g['mean_penetration_depth'],
        g['green_strain_area_rms'],g['edge_stretch_rms'],s['affected_rest_area_fraction'],
        s['persistent10_area_fraction']]).astype(np.float64)

def volume(z,low,high):
    with np.errstate(divide='ignore',invalid='ignore'):
        values=np.clip(np.log(high/np.maximum(z,low))/np.log(high/low),0.,1.)
    return 100.*np.prod(values,axis=1)

def scores(raw,position_consistency=None,domain=(.5,2.),config=None):
    cfg=config or json.loads((ROOT/'configs/evaluation.json').read_text())
    raw=np.asarray(raw,dtype=np.float64)
    if raw.ndim not in (2,3) or raw.shape[-1]!=6 or raw.shape[-2]==0:
        raise ValueError('Expected [T,6] or [4,T,6] complete measurements')
    if np.any(raw[np.isfinite(raw)]<0):raise ValueError('Measurements must be nonnegative')
    raw=np.where(np.isfinite(raw),raw,np.inf)
    z=raw/np.asarray([cfg['scales'][k] for k in FIELDS])
    caps=np.asarray([cfg['thresholds'][k] for k in FIELDS])
    ratio=raw/caps
    severity=np.stack([np.max(ratio[...,indices],axis=-1) for indices in GROUPS.values()],axis=-1)
    if raw.ndim==3:
        if raw.shape[0]!=4 or position_consistency is None:
            raise ValueError('Four-resolution evaluation requires four levels and position consistency')
        rho=np.asarray(position_consistency,dtype=np.float64)
        if rho.shape!=(raw.shape[1],) or np.any(rho[np.isfinite(rho)]<0):
            raise ValueError('Position consistency must have one nonnegative metre value per endpoint')
        rho=np.where(np.isfinite(rho),rho,np.inf)/cfg['position_scale_m']
        z=np.column_stack([z.max(axis=0),rho]);severity=severity.max(axis=0)
    severity=np.log2(np.maximum(1.,severity))
    penalty=100.*float(severity.mean(axis=0)@np.array([cfg['weights'][g] for g in GROUPS]))
    base={'JAR':100.*float((z<=1).all(axis=1).mean()),'JCV':float(volume(z,*domain).mean())}
    suffix='-4R' if raw.ndim==3 else ''
    return {**{k+suffix:float(np.maximum(0.,v-penalty)) for k,v in base.items()},
        'penalty_points':penalty if np.isfinite(penalty) else None,
        'unbounded_severity':bool(not np.isfinite(penalty)),'frames':int(raw.shape[-2])}
