"""Self-contact physical objectives and their differentiable geometry operations."""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import time
import numpy as np
import torch
from torch import nn
from arcon.models.self_contact_targets import GIATopology, original_gia_target
from arcon.models.self_collision import intersections

FAMILIES = ('B', 'G', 'P', 'C', 'I', 'D', 'M')


class SelfObjectiveError(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


@dataclass
class Config:
    family: str = 'G'
    scale: float = 1.
    coarse_scale: float = 1.
    local_scale: float = 1.
    probability: float = .2
    seed: int = 57


def create(mcfg):
    return Criterion(mcfg)


class MaterialRestriction:
    """Oriented manifold edge collapses, fixed boundary/HANDLE and convex P."""
    def __init__(self, rest, faces, pinned, fraction=.25):
        start = time.perf_counter()
        pos = np.asarray(rest, dtype=np.float64).copy()
        original = np.asarray(faces, dtype=np.int64)
        live = {i: tuple(map(int, f)) for i, f in enumerate(original)}
        vf = [set() for _ in pos]
        for i, face in live.items():
            for v in face: vf[v].add(i)
        edges = np.sort(original[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        locked = np.asarray(pinned, dtype=bool).copy()
        locked[np.unique(unique[counts == 1])] = True
        weights = [{i: 1.} for i in range(len(pos))]
        active = np.ones(len(pos), dtype=bool)
        def neighbors(v):
            result = set()
            for i in vf[v]: result.update(live[i])
            result.discard(v)
            return result
        def length(a, b): return float(np.linalg.norm(pos[a]-pos[b]))
        heap = [(length(a,b), int(a), int(b)) for a,b in unique if not (locked[a] or locked[b])]
        heapq.heapify(heap)
        merges = []
        target = math.ceil(fraction*len(original))
        while len(live) > target and heap:
            cost, a, b = heapq.heappop(heap)
            if not active[a] or not active[b] or locked[a] or locked[b]: continue
            edge_faces = vf[a]&vf[b]
            if len(edge_faces) != 2: continue
            opposite = set(v for i in edge_faces for v in live[i] if v not in (a,b))
            if neighbors(a)&neighbors(b) != opposite: continue  # exact manifold link condition
            current = length(a,b)
            if abs(cost-current) > 1e-12:
                heapq.heappush(heap,(current,a,b)); continue
            midpoint = .5*(pos[a]+pos[b])
            changed = vf[a]|vf[b]
            valid = True
            for i in changed-edge_faces:
                old = live[i]
                new = tuple(a if v == b else v for v in old)
                p = pos[list(old)]
                pp = np.stack([midpoint if v == a else pos[v] for v in new])
                no = np.cross(p[1]-p[0],p[2]-p[0])
                nn = np.cross(pp[1]-pp[0],pp[2]-pp[0])
                if np.dot(no,nn) <= 0 or np.linalg.norm(nn) < 1e-15:
                    valid = False; break
            if not valid: continue
            replacements = {}
            for i in changed:
                old = live.pop(i)
                for v in old: vf[v].remove(i)
                if i not in edge_faces: replacements[i] = tuple(a if v == b else v for v in old)
            live.update(replacements)
            for i, face in replacements.items():
                for v in face: vf[v].add(i)
            pos[a] = midpoint
            weights[a] = {k:.5*v for k,v in weights[a].items()} | {k:.5*v for k,v in weights[b].items()}
            weights[b] = {}
            active[b] = False
            merges.append((a,b,.5,.5))
            for v in neighbors(a):
                if not locked[v]:
                    x,y=sorted((a,v)); heapq.heappush(heap,(length(x,y),x,y))
        used = sorted(set(v for face in live.values() for v in face))
        renumber = {v:i for i,v in enumerate(used)}
        self.faces = np.asarray([[renumber[v] for v in face] for face in live.values()], dtype=np.int64)
        rows, columns, values = [], [], []
        for row,v in enumerate(used):
            for k,w in weights[v].items(): rows.append(row); columns.append(k); values.append(w)
        self.indices = np.asarray([rows,columns], dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64)
        self.shape = (len(used),len(pos))
        self.merges = np.asarray(merges, dtype=np.float64).reshape(-1,4)
        self.topology = GIATopology.build(len(used),self.faces)
        self._tensors = {}
        self.details = dict(original_faces=len(original), coarse_faces=len(self.faces),
            coarse_face_fraction=len(self.faces)/len(original), original_vertices=len(pos),
            coarse_vertices=len(used), convex_restriction_nonzeros=len(values), collapses=len(merges),
            fixed_boundary_and_handle_vertices=int(locked.sum()), preprocess_seconds=time.perf_counter()-start,
            restriction='fixed rest-edge manifold-link/orientation-preserving midpoint convex merges')

    def positions(self,q):
        key=(str(q.device),q.dtype)
        if key not in self._tensors:
            self._tensors[key]=torch.sparse_coo_tensor(
                torch.as_tensor(self.indices,device=q.device),
                torch.as_tensor(self.values,device=q.device,dtype=q.dtype),self.shape).coalesce()
        return torch.sparse.mm(self._tensors[key],q)

    def save_fields(self):
        return dict(indices=self.indices,values=self.values,shape=self.shape,faces=self.faces,
                    merges=self.merges,details=self.details)


class PhysicalTopology:
    def __init__(self,rest,faces,pinned):
        self.faces=np.asarray(faces,dtype=np.int64).copy()
        self.rest=np.asarray(rest,dtype=np.float64).copy()
        self.pinned=np.asarray(pinned,dtype=bool).copy()
        self.gia=None
        self.coarse=None
        self.neighbors=[set() for _ in faces]
        edge_faces={}
        for i,(a,b,c) in enumerate(self.faces):
            for x,y in ((a,b),(b,c),(c,a)): edge_faces.setdefault(tuple(sorted((x,y))),[]).append(i)
        for ids in edge_faces.values():
            for a in ids: self.neighbors[a].update(b for b in ids if b!=a)
        triangles=self.rest[self.faces]
        self.areas=.5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=-1)


def local_intersection_target(q,faces,geometry):
    if geometry['details']['coplanar_pairs']:
        raise SelfObjectiveError('I: positive-area coplanar overlap has no unique intersection line',geometry['details'])
    pairs=geometry['pairs']
    if not len(pairs): return q.sum()*0,dict(geometry['details'],partial_gradient='rigid translations')
    a=q[faces[pairs[:,0]]].mean(1)
    b=q[faces[pairs[:,1]]].mean(1)
    da,db=a-a.detach(),b-b.detach()
    endpoints=geometry['endpoints'].to(q.dtype)
    xa=endpoints+torch.einsum('peij,pj->pei',geometry['ja'].to(q.dtype),da)+torch.einsum(
        'peij,pj->pei',geometry['jb'].to(q.dtype),db)
    loss=(xa[:,0]-xa[:,1]).square().sum()
    return loss,dict(geometry['details'],partial_gradient='detached normals/edge vectors;face rigid translations',
                    raw_intersection_length_squared=float(loss.detach()))


def directed_patch_target(q,faces,topology,mass,geometry):
    pairs=geometry['pairs']
    if not len(pairs): return q.sum()*0,dict(geometry['details'],patches=0)
    # Contact-connected material neighborhoods, not Euclidean kNN clusters.
    pair_cpu=pairs.cpu().numpy()
    active=set(map(int,pair_cpu.reshape(-1)))
    parent={v:v for v in active}
    def root(v):
        while parent[v]!=v: parent[v]=parent[parent[v]]; v=parent[v]
        return v
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for a,b in pair_cpu: union(int(a),int(b))
    for a in sorted(active):
        for b in topology.neighbors[a]&active: union(a,b)
    groups={}
    for i,(a,b) in enumerate(pair_cpu): groups.setdefault(root(int(a)),[]).append(i)
    directions=torch.zeros((len(pairs),3),device=q.device,dtype=q.dtype)
    depths=torch.zeros(len(pairs),device=q.device,dtype=q.dtype)
    with torch.no_grad():
        triangles=q.double()[faces]
        normals=torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        area=.5*normals.norm(dim=-1)
        normals=normals/normals.norm(dim=-1,keepdim=True)
        centers=triangles.mean(1)
        for ids in groups.values():
            ids=torch.as_tensor(ids,device=q.device)
            p=pairs[ids]
            face_ids=torch.unique(p)
            n=normals[face_ids]
            matrix=(n.T*area[face_ids])@n
            axes=torch.linalg.eigh(matrix).eigenvectors.T
            choices=[]
            for axis in axes:
                order=((centers[p[:,1]]-centers[p[:,0]])*axis).sum(-1)
                sign=torch.where(order<0,-1.,1.)
                r=axis[None]*sign[:,None]
                projection=(geometry['axes'][ids]*r[:,None]).sum(-1)
                denom=projection.abs().clamp_min(1e-15)
                delta=torch.where(projection>0,
                    (geometry['a_max'][ids]-geometry['b_min'][ids])/denom,
                    (geometry['b_max'][ids]-geometry['a_min'][ids])/denom)
                eligible=geometry['axes_valid'][ids]&(projection.abs()>1e-12)
                delta=delta.masked_fill(~eligible,float('inf')).clamp_min(0).amin(-1)
                depth=delta.amax()
                face_mass=mass[faces[face_ids]].mean()
                cost=(depth+.001).square()*area[face_ids].sum()*face_mass
                choices.append((float(cost),depth,r))
            cost,depth,r=min(choices,key=lambda x:x[0])
            if not math.isfinite(cost): raise SelfObjectiveError('D: no supported rigid separation direction',geometry['details'])
            depths[ids]=depth.to(q.dtype)
            directions[ids]=r.to(q.dtype)
    a=q[faces[pairs[:,0]]].mean(1)
    b=q[faces[pairs[:,1]]].mean(1)
    gap=(directions*((b-a)-(b-a).detach())).sum(-1)-depths-.001
    rest_area=torch.as_tensor(topology.areas,device=q.device,dtype=q.dtype)
    weights=.5*(rest_area[pairs[:,0]]+rest_area[pairs[:,1]])
    multiplier=1000*torch.relu(-gap)
    loss=(weights*500*torch.relu(-gap).square()).sum()
    return loss,dict(geometry['details'],patches=len(groups),separation_clearance_m=.001,
        partial_gradient='detached local SAT direction/depth;pair face translations',
        separation_depth_max_m=float(depths.max()),separation_depth_median_m=float(depths.median()),
        dual_multiplier_max=float(multiplier.detach().max()),quadrature='mean pair rest-face area')


class Criterion(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.name='self_intersection_energy'
        self.family=str(config.family)
        if self.family not in FAMILIES: raise ValueError(self.family)
        self.scale=float(config.scale)
        self.coarse_scale=float(config.coarse_scale)
        self.local_scale=float(config.local_scale)
        self.probability=float(config.probability)
        if self.probability!=.2: raise ValueError('Approved P probability is .2')
        self.rng=np.random.default_rng(int(config.seed))
        self.calls=0
        self.last=None
        self.failure_geometry=None
        self.topologies={}
        self.readout_protocol='wj_warmstart_self_objective_'+self.family+'_v1'

    def topology(self,q,faces,rest,pinned,name):
        key=(str(name),len(q),len(faces))
        if key not in self.topologies:
            self.topologies[key]=PhysicalTopology(rest.detach().cpu().numpy(),faces.detach().cpu().numpy(),pinned.cpu().numpy())
        return self.topologies[key]

    def evaluate(self,q,faces,rest,pinned,mass,name,*,force_full=False,geometry=None):
        begin=time.perf_counter()
        self.calls+=1
        self.last=None
        self.failure_geometry=None
        family=self.family
        selected=family!='P' or force_full or bool(self.rng.random()<self.probability)
        try:
            if family=='B' or not selected:
                energy=q.sum()*0
                details=dict(sampled=selected,gia_called=False,declared_random_zero=family=='P')
            else:
                topology=self.topology(q,faces,rest,pinned,name)
                if family in ('G','P'):
                    if topology.gia is None: topology.gia=GIATopology.build(len(q),topology.faces)
                    result=original_gia_target(q,topology.gia)
                    energy=result.loss/(self.probability if family=='P' and not force_full else 1.)
                    details=dict(result.details,gia_called=True,sampled=True)
                elif family=='C':
                    if topology.coarse is None: topology.coarse=MaterialRestriction(topology.rest,topology.faces,topology.pinned)
                    qc=topology.coarse.positions(q)
                    result=original_gia_target(qc,topology.coarse.topology)
                    energy=result.loss
                    details=dict(result.details,**topology.coarse.details,gia_called=True)
                elif family in ('I','D'):
                    geometry=intersections(q,faces) if geometry is None else geometry
                    if family=='I': energy,details=local_intersection_target(q,faces,geometry)
                    else: energy,details=directed_patch_target(q,faces,topology,mass,geometry)
                    details['gia_called']=False
                else:
                    if topology.coarse is None: topology.coarse=MaterialRestriction(topology.rest,topology.faces,topology.pinned)
                    coarse=original_gia_target(topology.coarse.positions(q),topology.coarse.topology)
                    geometry=intersections(q,faces) if geometry is None else geometry
                    local,details=local_intersection_target(q,faces,geometry)
                    energy=.5*self.coarse_scale*coarse.loss+.5*self.local_scale*local
                    details=dict(details,coarse=coarse.details,restriction=topology.coarse.details,
                                 gia_called=True,mix='fixed half calibrated C + half calibrated I')
            loss=self.scale*energy
            if not torch.isfinite(loss).all(): raise SelfObjectiveError('Nonfinite self target',details)
            self.last=dict(details,family=family,scale=self.scale,criterion_calls=self.calls,
                           readout_protocol=self.readout_protocol,criterion_forward_seconds=time.perf_counter()-begin,
                           target_value=float(loss.detach()))
            return loss
        except Exception as exc:
            self.failure_geometry=dict(q=q.detach().cpu(),physical_faces=faces.detach().cpu(),
                rest_pos=rest.detach().cpu(),pinned=pinned.cpu(),mass=mass.detach().cpu(),garment=str(name),
                family=family,details=getattr(exc,'details',{}),failure=type(exc).__name__+': '+str(exc))
            raise

    def forward(self,sample):
        if sample.num_graphs!=1: raise ValueError('Matrix batch is one')
        example=sample.get_example(0)
        cloth=example['cloth']
        name=example.garment_name
        if isinstance(name,(tuple,list)): name=name[0]
        loss=self.evaluate(cloth.pred_pos,cloth.faces_batch.T,cloth.rest_pos,
            cloth.vertex_type.reshape(-1)==3,cloth.v_mass.reshape(-1),str(name))
        return dict(loss=loss)
