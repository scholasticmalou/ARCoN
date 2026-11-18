"""Counted GPU broad phase and triangle-geometry queries for self-contact supervision."""
from __future__ import annotations
import os
from pathlib import Path
import time
import torch

_EXTENSION = None
GEOMETRY_PROTOCOL = 'counted_cc_bvh_float64_full_sat_ef_v1'


def load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load
        base = Path(os.environ.get('PB4U_MATRIX_BASE', '/workspace/cvpr2027_direction1'))
        source = base/'external/CCCollisions-torch211-compat'
        here = Path(__file__).resolve().parent
        _EXTENSION = load(name='pb4u_self_counted_bvh_v1',
            sources=[str(here/'self_collision_ext.cpp'), str(here/'self_collision_ext.cu')],
            extra_include_paths=[str(source/'src'), str(source/'include'), str(source/'include/cgbn'),
                                 str(base/'external/cuda-samples-12.4/Common')],
            extra_cflags=['-O2'], extra_cuda_cflags=['-O2'], verbose=True)
    return _EXTENSION


def synchronize(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


@torch.no_grad()
def aabb_pairs(q, faces):
    extension = load_extension()
    triangles = q.detach().double()[faces].unsqueeze(0).contiguous()
    boxes, tree = extension.bvh(triangles)
    capacity, attempts = max(8*len(faces), 1), 0
    while True:
        pairs, counter = extension.candidates(boxes, tree, capacity)
        count = int(counter.item())
        attempts += 1
        if count < 0: raise RuntimeError('BVH candidate count overflow')
        if count <= capacity:
            valid = pairs[:, 0] >= 0
            if int(valid.sum()) != count: raise RuntimeError('Counted BVH incomplete output')
            pairs = pairs[valid]
            keys = torch.unique(pairs[:, 0]*len(faces)+pairs[:, 1], sorted=True)
            if len(keys) != count: raise RuntimeError('Counted BVH duplicate output')
            return torch.stack((keys//len(faces), keys%len(faces)), -1), dict(
                aabb_candidates=count, candidate_capacity=capacity, query_attempts=attempts)
        del pairs
        capacity = max(2*capacity, count+1)
        if capacity >= 2**31: raise RuntimeError('BVH candidate capacity exceeds native int range')


def sat_axes(tri_a, tri_b):
    ea = tri_a.roll(-1, 1)-tri_a
    eb = tri_b.roll(-1, 1)-tri_b
    na = torch.linalg.cross(ea[:, 0], -ea[:, 2])
    nb = torch.linalg.cross(eb[:, 0], -eb[:, 2])
    cross = torch.linalg.cross(ea[:, :, None, :], eb[:, None, :, :]).flatten(1, 2)
    # In-plane axes are necessary for coplanar triangles; adding them is also
    # a valid separation test for noncoplanar triangles.
    planar_a = torch.linalg.cross(na[:, None, :], ea)
    planar_b = torch.linalg.cross(nb[:, None, :], eb)
    axes = torch.cat((na[:, None], nb[:, None], cross, planar_a, planar_b), 1)
    lengths = axes.norm(dim=-1)
    good = lengths > 1e-20
    axes = axes/lengths.clamp_min(1e-20)[..., None]
    return axes, good, na, nb


@torch.no_grad()
def intersections(q, faces):
    """Current physical triangle intersections and actual EF/loop endpoints.

    Edge-neighbor faces are excluded as local material adjacency. Single-shared-
    vertex pairs remain, but their ordinary point contact is not a self crossing.
    Positive-area coplanar overlap is reported explicitly (I has no line target).
    """
    begin = time.perf_counter()
    faces = faces.long()
    pairs, details = aabb_pairs(q, faces)
    zero_pairs = pairs.new_empty((0, 2))
    if not len(pairs):
        details.update(geometry_protocol=GEOMETRY_PROTOCOL, intersection_pairs=0, coplanar_pairs=0,
                       geometry_seconds=time.perf_counter()-begin)
        return dict(pairs=zero_pairs, details=details)
    fa, fb = faces[pairs[:, 0]], faces[pairs[:, 1]]
    shared = (fa[:, :, None] == fb[:, None, :]).sum((1, 2))
    details['material_edge_neighbors_excluded'] = int((shared >= 2).sum())
    pairs = pairs[shared < 2]
    fa, fb = faces[pairs[:, 0]], faces[pairs[:, 1]]
    tri_a, tri_b = q.detach().double()[fa], q.detach().double()[fb]
    axes, good, na, nb = sat_axes(tri_a, tri_b)
    if not ((na.norm(dim=-1)>0)&(nb.norm(dim=-1)>0)).all():
        raise RuntimeError('Degenerate physical triangle in self query')
    pa = torch.einsum('pvc,pac->pav', tri_a, axes)
    pb = torch.einsum('pvc,pac->pav', tri_b, axes)
    overlap = torch.minimum(pa.amax(-1), pb.amax(-1))-torch.maximum(pa.amin(-1), pb.amin(-1))
    sat = ((overlap >= -1e-10)|(~good)).all(-1)
    pairs, tri_a, tri_b = pairs[sat], tri_a[sat], tri_b[sat]
    axes, good, na, nb = axes[sat], good[sat], na[sat], nb[sat]
    fa, fb = fa[sat], fb[sat]
    pa, pb, overlap = pa[sat], pb[sat], overlap[sat]
    na = na/na.norm(dim=-1, keepdim=True)
    nb = nb/nb.norm(dim=-1, keepdim=True)
    parallel = torch.linalg.cross(na, nb).norm(dim=-1) < 1e-10
    coplanar = parallel & (((tri_b-tri_a[:, :1])*na[:, None]).sum(-1).abs().amax(-1) < 1e-10)
    inplane = good & ((axes*na[:, None]).sum(-1).abs() < 1-1e-10)
    positive_area = ((overlap > 1e-10)|(~inplane)).all(-1)
    cp = coplanar & positive_area
    # Noncoplanar EF hits: each endpoint belongs to one edge and the other face.
    edge_tri = torch.cat((tri_a, tri_b), 1)
    end_tri = torch.cat((tri_a.roll(-1, 1), tri_b.roll(-1, 1)), 1)
    edge = end_tri-edge_tri
    target_n = torch.cat((nb[:, None].expand(-1, 3, -1), na[:, None].expand(-1, 3, -1)), 1)
    target_c = torch.cat((tri_b.mean(1)[:, None].expand(-1, 3, -1),
                          tri_a.mean(1)[:, None].expand(-1, 3, -1)), 1)
    denom = (edge*target_n).sum(-1)
    usable = denom.abs() > 1e-15
    safe_denom = torch.where(usable, denom, torch.ones_like(denom))
    t = ((target_c-edge_tri)*target_n).sum(-1)/safe_denom
    points = edge_tri+t[..., None]*edge
    target_tri = torch.cat((tri_b[:, None].expand(-1, 3, -1, -1),
                            tri_a[:, None].expand(-1, 3, -1, -1)), 1)
    v0, v1 = target_tri[:, :, 1]-target_tri[:, :, 0], target_tri[:, :, 2]-target_tri[:, :, 0]
    v2 = points-target_tri[:, :, 0]
    d00, d01, d11 = (v0*v0).sum(-1), (v0*v1).sum(-1), (v1*v1).sum(-1)
    determinant = d00*d11-d01.square()
    if (determinant <= 0).any(): raise RuntimeError('Degenerate barycentric denominator in self query')
    d20, d21 = (v2*v0).sum(-1), (v2*v1).sum(-1)
    u = (d11*d20-d01*d21)/determinant
    v = (d00*d21-d01*d20)/determinant
    hit = usable & (t >= -1e-9) & (t <= 1+1e-9) & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)
    # Shared-vertex loop endpoint stays fixed, as in the source component.
    shared_mask = fa[:, :, None] == fb[:, None, :]
    has_loop = shared_mask.any((1, 2))
    shared_corner = shared_mask.any(-1).long().argmax(-1)
    loop_point = tri_a[torch.arange(len(pairs), device=q.device), shared_corner]
    points = torch.cat((points, loop_point[:, None]), 1)
    hit = torch.cat((hit, has_loop[:, None]), 1)
    distances = (points[:, :, None]-points[:, None, :]).square().sum(-1)
    distances = distances.masked_fill(~(hit[:, :, None]&hit[:, None, :]), -1)
    largest, index = distances.flatten(1).max(-1)
    line = (largest > 1e-20)&(~coplanar)
    selected = line|cp
    endpoints_index = torch.stack((index//7, index%7), -1)
    endpoints = points.gather(1, endpoints_index[..., None].expand(-1, -1, 3))
    # Jacobians under rigid translation of A or B; no rotation/stretch gradient.
    outer = edge[..., :, None]*target_n[..., None, :]/safe_denom[..., None, None]
    eye = torch.eye(3, dtype=torch.float64, device=q.device)
    ja = torch.cat((eye-outer[:, :3], outer[:, 3:]), 1)
    jb = torch.cat((outer[:, :3], eye-outer[:, 3:]), 1)
    zeros = torch.zeros_like(ja[:, :1])
    ja, jb = torch.cat((ja, zeros), 1), torch.cat((jb, zeros), 1)
    idx = endpoints_index[..., None, None].expand(-1, -1, 3, 3)
    ja, jb = ja.gather(1, idx), jb.gather(1, idx)
    at_loop = has_loop[:, None] & ((endpoints-loop_point[:, None]).square().sum(-1) < 1e-20)
    ja = ja.masked_fill(at_loop[..., None, None], 0)
    jb = jb.masked_fill(at_loop[..., None, None], 0)
    details.update(geometry_protocol=GEOMETRY_PROTOCOL, intersection_pairs=int(selected.sum()),
        line_pairs=int(line.sum()), coplanar_pairs=int(cp.sum()),
        point_or_material_contact_excluded=int((~selected).sum()),
        shared_vertex_line_pairs=int((line&has_loop).sum()),
        geometry_seconds=time.perf_counter()-begin)
    return dict(pairs=pairs[selected], endpoints=endpoints[selected], ja=ja[selected], jb=jb[selected],
                axes=axes[selected], axes_valid=good[selected], a_min=pa.amin(-1)[selected],
                a_max=pa.amax(-1)[selected], b_min=pb.amin(-1)[selected], b_max=pb.amax(-1)[selected],
                coplanar=cp[selected], details=details)
