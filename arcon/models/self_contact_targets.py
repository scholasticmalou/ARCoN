"""Intersection-region targets adapted from the attributed SENC implementation.

Topology selection is detached; selected geometric coordinates are reconstructed
in Torch for the loss derivative. This module is not a collision-free guarantee."""
from __future__ import annotations
from dataclasses import dataclass
import time
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

SENC_COMMIT = '66e433f9668982aa7d79fc31fd2892c57f19e5ef'
GIA_WEIGHT = 30000.0
GIA_READOUT_PROTOCOL = 'senc_gia_canonical_cgal_actual_region_faces_fp32_l1_30000_v2'
CGAL_ENUMERATION_PROTOCOL = 'original_prefix_class_incidence_oriented_enumeration_v1'


class GIAComputationError(RuntimeError):
    def __init__(self, message, details=None, intersection_pairs=None):
        super().__init__(message)
        self.details = {} if details is None else details
        self.intersection_pairs = intersection_pairs


def _cyclic_faces(faces):
    """Lexicographically first cyclic rotation; never reverse orientation."""
    best = faces.copy()
    for shift in (1, 2):
        candidate = np.roll(faces, -shift, axis=1)
        less = ((candidate[:, 0] < best[:, 0]) |
                ((candidate[:, 0] == best[:, 0]) &
                 ((candidate[:, 1] < best[:, 1]) |
                  ((candidate[:, 1] == best[:, 1]) & (candidate[:, 2] < best[:, 2])))))
        best[less] = candidate[less]
    return best


def canonicalize_cgal_enumeration(output, n_original_vertices, *, return_mapping=False):
    """Relabel a real CGAL tuple without changing geometry or equivalence classes.

    Original loss vertices, including cap centers, retain their indices. Classes
    are anchored by their exact coordinate multiset and fixed original members.
    New aliases are ordered by exact coordinates and oriented incident birth-face
    signatures, not by their incoming CGAL IDs. A class uses its lowest fixed
    original index, or first canonical new alias, as representative. Thus both
    domain and range of `unique` are relabeled, with no class merging/splitting.

    IF refers to ORIGINAL input faces: sort directed rows only, never remap it
    using the remeshed-face permutation. FF permits cyclic rotations, not flips.
    Ambiguous referenced aliases are reported, not resolved by geometry rounding
    or an unstable incoming ID. Unreferenced identical aliases are interchangeable
    in every returned array but remain separate entries.
    """
    vertices, faces, pairs, birth, unique = (np.asarray(x) for x in output)
    n = int(n_original_vertices)
    u = unique.reshape(-1)
    b = birth.reshape(-1)
    _, classes, counts = np.unique(u, return_inverse=True, return_counts=True)
    member_order = np.argsort(classes, kind='stable')
    offsets = np.r_[0, np.cumsum(counts)]
    members = [member_order[offsets[i]:offsets[i+1]] for i in range(len(counts))]
    class_keys = [(tuple(sorted(tuple(row) for row in vertices[group])),
                   tuple(int(i) for i in group if i < n)) for group in members]
    if len(set(class_keys)) != len(class_keys):
        raise ValueError('CGAL has distinct equivalence classes with identical exact anchors')
    class_order = sorted(range(len(members)), key=class_keys.__getitem__)
    class_rank = np.empty(len(members), dtype=np.int64)
    class_rank[class_order] = np.arange(len(members))
    face_classes = class_rank[classes[faces]]
    incident = [[] for _ in range(len(vertices))]
    for face, source, quotient in zip(faces.tolist(), b.tolist(), face_classes.tolist()):
        for corner, vertex in enumerate(face):
            if vertex >= n:
                incident[vertex].append((source, quotient[(corner+1) % 3], quotient[(corner+2) % 3]))
    new_members = []
    for group_id in class_order:
        aliases = [int(i) for i in members[group_id] if i >= n]
        keys = [(tuple(vertices[i]), tuple(sorted(incident[i]))) for i in aliases]
        referenced_keys = [key for key in keys if key[1]]
        if len(set(referenced_keys)) != len(referenced_keys):
            raise ValueError('CGAL has referenced aliases not distinguished by exact oriented birth-face incidence')
        new_members.extend(aliases[i] for i in sorted(range(len(aliases)), key=keys.__getitem__))
    new_to_old = np.r_[np.arange(n, dtype=np.int64), np.asarray(new_members, dtype=np.int64)]
    old_to_new = np.empty(len(vertices), dtype=np.int64)
    old_to_new[new_to_old] = np.arange(len(vertices))
    representatives = np.asarray([old_to_new[group].min() for group in members], dtype=np.int64)
    canonical_unique = representatives[classes[new_to_old]].astype(unique.dtype, copy=False).reshape(unique.shape)
    canonical_faces = _cyclic_faces(old_to_new[faces])
    face_order = np.lexsort((canonical_faces[:, 2], canonical_faces[:, 1], canonical_faces[:, 0], b))
    pair_order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    normalized = (vertices[new_to_old], canonical_faces[face_order].astype(faces.dtype, copy=False),
                  pairs[pair_order], b[face_order].reshape(birth.shape), canonical_unique)
    if return_mapping:
        return normalized, dict(new_to_old=new_to_old, old_to_new=old_to_new,
                                face_new_to_old=face_order, pair_new_to_old=pair_order)
    return normalized


@dataclass
class GIATopology:
    n_vertices: int
    physical_faces: np.ndarray
    closed_faces: np.ndarray
    boundary_paths: tuple[np.ndarray, ...]
    old_face_adjacency: dict
    mesh_pair_keys: np.ndarray
    diagnostics: dict

    @classmethod
    def build(cls, n_vertices, faces):
        from utils_self_inter.mesh import get_vertex_edge_dict_and_face_adj_dict
        f = np.asarray(faces, dtype=np.int64)
        if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
            raise GIAComputationError('GIA needs nonempty triangular physical faces')
        if f.min() < 0 or f.max() >= n_vertices:
            raise GIAComputationError('Physical face index outside the original vertices')
        directed = f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        canonical = np.sort(directed, axis=1)
        keys = canonical[:, 0]*n_vertices+canonical[:, 1]
        mesh_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        diagnostics = dict(n_vertices=int(n_vertices), physical_faces=int(len(f)),
                           physical_mesh_edges=int(len(mesh_keys)),
                           boundary_edges=int(np.count_nonzero(counts == 1)),
                           nonmanifold_edges=int(np.count_nonzero(counts > 2)))
        if diagnostics['nonmanifold_edges']:
            raise GIAComputationError('Physical mesh has an edge shared by more than two faces', diagnostics)
        if np.any(directed[:, 0] == directed[:, 1]):
            raise GIAComputationError('Physical mesh has a degenerate topological edge', diagnostics)
        signs = np.where(directed[:, 0] < directed[:, 1], 1., -1.)
        orientation_sum = np.bincount(inverse, weights=signs, minlength=len(mesh_keys))
        bad_orientation = int(np.count_nonzero((counts == 2) & (orientation_sum != 0)))
        diagnostics['inconsistent_interior_edge_orientations'] = bad_orientation
        if bad_orientation:
            raise GIAComputationError('Physical mesh face orientations do not define an oriented surface', diagnostics)
        boundary = directed[counts[inverse] == 1]
        outgoing, incoming = {}, {}
        for a, b in boundary.tolist():
            outgoing.setdefault(a, []).append(b)
            incoming.setdefault(b, []).append(a)
        vertices = set(outgoing) | set(incoming)
        bad_boundary = [v for v in vertices if len(outgoing.get(v, [])) != 1 or len(incoming.get(v, [])) != 1]
        diagnostics['non_simple_boundary_vertices'] = len(bad_boundary)
        if bad_boundary:
            raise GIAComputationError('Physical boundary is not a union of oriented simple loops', diagnostics)
        remaining = set(vertices)
        loops = []
        while remaining:
            start = min(remaining)
            path, current = [], start
            while True:
                if current not in remaining:
                    raise GIAComputationError('Boundary walk revisited a vertex before closing its own loop', diagnostics)
                remaining.remove(current)
                path.append(current)
                current = outgoing[current][0]
                if current == start:
                    break
            if len(path) < 3:
                raise GIAComputationError('Boundary loop has fewer than three vertices', diagnostics)
            loops.append(np.asarray(path, dtype=np.int64))
        caps = []
        for i, path in enumerate(loops):
            center = n_vertices+i
            # Original boundary a->b is closed with b->a, not a new physical node.
            caps.extend((int(b), int(a), int(center)) for a, b in zip(path, np.roll(path, -1)))
        cap_array = np.asarray(caps, dtype=np.int64).reshape(-1, 3)
        closed = np.concatenate((f, cap_array), axis=0)
        n_closed = n_vertices+len(loops)
        closed_edges = np.sort(closed[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        _, closed_counts = np.unique(closed_edges[:, 0]*n_closed+closed_edges[:, 1], return_counts=True)
        if not np.all(closed_counts == 2):
            raise GIAComputationError('Boundary caps did not produce a closed edge-manifold loss mesh', diagnostics)
        _, old_adjacency = get_vertex_edge_dict_and_face_adj_dict(closed)
        diagnostics.update(boundary_loops=len(loops), boundary_lengths=[len(x) for x in loops],
                           cap_faces=len(cap_array), loss_vertices=n_closed, loss_faces=len(closed),
                           cap_scope='Loss-only differentiable boundary centers; original physical mesh unchanged')
        return cls(n_vertices, f.copy(), closed, tuple(loops), old_adjacency, mesh_keys, diagnostics)

    def positions(self, q):
        if q.shape != (self.n_vertices, 3):
            raise GIAComputationError('Prediction shape differs from cached physical template', self.diagnostics)
        centers = [q[torch.as_tensor(path, device=q.device)].mean(dim=0, keepdim=True)
                   for path in self.boundary_paths]
        return torch.cat([q]+centers, dim=0) if centers else q


@dataclass
class GIAResult:
    loss: torch.Tensor
    details: dict
    intersection_pairs: np.ndarray | None = None


def original_gia_target(q, topology, *, capture_remesh=False):
    from utils_self_inter.find_penetration_triangles_one_batch import compute_self_intersection_one_batch
    import igl.copyleft.cgal as cgal
    if not torch.isfinite(q).all().item():
        raise GIAComputationError('Original GIA received nonfinite natural prediction', topology.diagnostics)
    augmented = topology.positions(q)
    numpy_pos = augmented.detach().cpu().numpy().copy().astype(np.float64)
    captured = {}
    original_remesh = cgal.remesh_self_intersections

    def canonical_remesh(*args, **kwargs):
        output = original_remesh(*args, **kwargs)
        normalization_start = time.perf_counter()
        try:
            output = canonicalize_cgal_enumeration(output, len(args[0]))
        finally:
            captured['cgal_normalization_seconds'] = time.perf_counter()-normalization_start
        if capture_remesh:
            captured['intersection_pairs'] = np.asarray(output[2], dtype=np.int64).copy()
            captured['birth_tri'] = np.asarray(output[3], dtype=np.int64).copy()
        return output

    start = time.perf_counter()
    cgal.remesh_self_intersections = canonical_remesh
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message=r'self-intersection path .*: inside/outside flood fill failed, skipping')
            ff, new_indices, new_bary, regions = compute_self_intersection_one_batch(
                numpy_pos, topology.closed_faces, topology.old_face_adjacency)
    except Exception as exc:
        details = dict(topology.diagnostics, original_gia_seconds=time.perf_counter()-start,
                       failure_type=type(exc).__name__, failure=str(exc),
                       cgal_enumeration_protocol=CGAL_ENUMERATION_PROTOCOL,
                       cgal_normalization_seconds=captured.get('cgal_normalization_seconds', 0.))
        if 'intersection_pairs' in captured:
            pairs = captured['intersection_pairs']
            details['cgal_intersecting_face_pairs'] = len(pairs)
            details['physical_intersecting_face_pairs'] = int(np.count_nonzero(
                (pairs[:, 0] < len(topology.physical_faces)) & (pairs[:, 1] < len(topology.physical_faces))))
        raise GIAComputationError('Pinned original GIA region/reconstruction computation failed', details, captured.get('intersection_pairs')) from exc
    finally:
        cgal.remesh_self_intersections = original_remesh
    gia_seconds = time.perf_counter()-start
    if regions:
        ff_tensor = torch.as_tensor(ff, dtype=torch.long, device=q.device)
        ff_tensor = torch.cat((ff_tensor, torch.zeros((1, 3), dtype=torch.long, device=q.device)), dim=0)
        idx = torch.as_tensor(new_indices, dtype=torch.long, device=q.device)
        bary = torch.as_tensor(new_bary, device=q.device)  # Preserve author float32 coefficients.
        new_points = (augmented[idx]*bary.unsqueeze(-1)).sum(dim=1)
        final_positions = torch.cat((augmented, new_points), dim=0)
        region_tensors = [torch.as_tensor(region, dtype=torch.long, device=q.device) for region in regions]
        padded = pad_sequence(region_tensors, batch_first=True, padding_value=len(ff_tensor)-1)
        ids = ff_tensor[padded]
        p0, p1, p2 = (final_positions[ids[:, :, i]] for i in range(3))
        products = p0*torch.linalg.cross(p1, p2, dim=-1)
        # The [0,0,0] padding face is zero only in exact arithmetic; float32 cross(v0,v0)
        # is not, and ~10^7 padding slots times the weight dominate the sum. Mask them.
        valid = padded != (len(ff_tensor)-1)
        triple = (products*valid.unsqueeze(-1)).sum(dim=(-1, -2))
        loss = GIA_WEIGHT*triple.abs().sum()  # Original L1 expression, no extra 1/6.
        raw = float(triple.detach().abs().sum().cpu())
        signed = triple.detach().cpu().tolist()
        unmasked = products.detach().sum(dim=(-1, -2))
        padding_audit = dict(padding_slots=int((~valid).sum()), actual_region_faces=int(valid.sum()),
                             unmasked_author_loss=float(GIA_WEIGHT*unmasked.abs().sum().cpu()),
                             padding_flipped_region_signs=int((torch.sign(unmasked) != torch.sign(triple.detach())).sum()))
    else:
        loss = q.new_zeros(())  # A real no-region result, not an error fallback.
        raw, signed = 0., []
        padding_audit = dict(padding_slots=0, actual_region_faces=0, unmasked_author_loss=0.,
                             padding_flipped_region_signs=0)
    if not torch.isfinite(loss).item():
        raise GIAComputationError('Original GIA produced a nonfinite target', dict(topology.diagnostics))
    details = dict(readout_protocol=GIA_READOUT_PROTOCOL,
                   cgal_enumeration_protocol=CGAL_ENUMERATION_PROTOCOL,
                   cgal_normalization_seconds=captured['cgal_normalization_seconds'],
                   original_gia_seconds_without_normalization=gia_seconds-captured['cgal_normalization_seconds'],
                   original_gia_seconds=gia_seconds, remeshed_faces=len(ff),
                   reconstructed_new_vertices=len(new_indices), region_count=len(regions),
                   signed_region_triple_products=signed, raw_l1_triple_product=raw,
                   geometric_volume_m3=raw/6., loss_weight=GIA_WEIGHT,
                   geometric_volume_note='Reporting conversion only; the training target is not divided by six.',
                   gradient_path='Detached topology/barycentric selection; Torch reconstruction including cap centers',
                   padding_mask='Region sum over actual faces only; author unmasked value recorded separately',
                   **padding_audit)
    pairs = captured.get('intersection_pairs')
    if pairs is not None:
        physical = (pairs[:, 0] < len(topology.physical_faces)) & (pairs[:, 1] < len(topology.physical_faces))
        details.update(cgal_intersecting_face_pairs=len(pairs),
                       physical_intersecting_face_pairs=int(physical.sum()),
                       cap_involved_intersecting_face_pairs=int((~physical).sum()))
    return GIAResult(loss, details, pairs)


@dataclass
class Config:
    weight: float = GIA_WEIGHT
    norm: str = 'l1'


def create(mcfg):
    if float(mcfg.weight) != GIA_WEIGHT or mcfg.norm != 'l1':
        raise ValueError('This approved screen fixes original GIA L1/30000')
    return Criterion()


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'self_intersection_energy'
        self.weight = GIA_WEIGHT
        self.readout_protocol = GIA_READOUT_PROTOCOL
        self.cgal_enumeration_protocol = CGAL_ENUMERATION_PROTOCOL
        self._topologies = {}
        self.last = None
        self.failure_geometry = None

    def forward(self, sample):
        start = time.perf_counter()
        self.last = None
        self.failure_geometry = None
        if sample.num_graphs != 1:
            raise ValueError('The approved GIA screen has batch size one')
        example = sample.get_example(0)
        cloth = example['cloth']
        q = cloth.pred_pos
        name = example.garment_name
        if isinstance(name, (list, tuple)):
            name = name[0]
        key = (str(name), len(q), int(cloth.faces_batch.shape[1]))
        if key not in self._topologies:
            self._topologies[key] = GIATopology.build(len(q), cloth.faces_batch.T.detach().cpu().numpy())
        try:
            result = original_gia_target(q, self._topologies[key])
        except GIAComputationError as exc:
            self.failure_geometry = dict(garment=str(name), q=q.detach().cpu(),
                physical_faces=cloth.faces_batch.T.detach().cpu(), details=exc.details)
            raise
        self.last = dict(result.details, criterion_forward_seconds=time.perf_counter()-start,
                         garment=str(name), original_vertices=len(q))
        return dict(loss=result.loss)