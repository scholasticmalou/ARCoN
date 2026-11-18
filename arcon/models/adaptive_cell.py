"""ARCoN geometry-conditioned graph computation within one physical time step.

Each of three rounds refreshes geometric conditions, propagates local and
nonlocal information, and applies a displacement proposal with a relation
response. Parameter and normalizer keys remain compatible with the paper weights."""
from __future__ import annotations

import copy
import heapq
import math
import time
import types

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from models.core.pb4u import EncodeProcessDecode as NativeCore, GraphNetBlock
from models.core.base import make_edgesets_dict
from models.core.relation_response import mlp as plain_mlp, hinge_basis, vertex_normals, mean_scatter
from models.core.geometry_feedback import GeometryRelationResponse, unique_hinge_gradient, hinge_mass_gram
from models.core.graph_variants import capture_latents, add_response, recomputed_native_block, _assign_latents, MESH, DIRECT, REVERSE
from models.core.graph_context import ContextStageFeedback, surface_weights
from models.core.self_interaction import MaterialHopTwo, nonlocal_self_pairs
from models.pb4u import Model as NativeModel
from utils.common import NodeType, add_field_to_pyg_batch

ARMS = ('Cell-F', 'Cell-Dir', 'Cell-Scale', 'Cell-Joint', 'Cell-Fixed',
        'Cell-ScaleFixed', 'Cell-NetDir')


class RecomputedMLP(nn.Sequential):
    """Same MLP; optional inner recomputation is an execution-only memory policy."""
    recompute_activations = True

    def forward(self, value):
        if self.recompute_activations and torch.is_grad_enabled():
            return checkpoint(lambda x: nn.Sequential.forward(self, x), value,
                              use_reentrant=False, preserve_rng_state=False)
        return nn.Sequential.forward(self, value)


def configure_mlp_recomputation(module, enabled=True):
    """Keep all outer checkpoints and model/state-dict structure unchanged."""
    count = 0
    for child in module.modules():
        if isinstance(child, RecomputedMLP):
            child.recompute_activations = bool(enabled)
            count += 1
    return count


def mlp(*args, **kwargs):
    original = plain_mlp(*args, **kwargs)
    return RecomputedMLP(*original.children())


class ProposalRelationResponse(GeometryRelationResponse):
    """The existing geometry-conditioned assembly without any message processor."""
    def __init__(self):
        nn.Module.__init__(self)
        self.mode = 'R_geom'
        self.node_encoder = mlp(15)
        self.edge_encoder = mlp(8)
        self.face_encoder = mlp(13)
        self.world_encoder = mlp(13)
        self.hinge_encoder = mlp(3)
        self.node_fusion = mlp(384)
        self.edge_fusion = mlp(384)
        self.world_fusion = mlp(256)
        self.body_fusion = mlp(256)
        self.edge_head = mlp(384, 1, norm=False, zero=True)
        self.hinge_head = mlp(384, 1, norm=False, zero=True)
        self.contact_head = mlp(384, 4, norm=False, zero=True)
        self.record = False
        self.last = None
        self.geometry_conditioned = True
        self.source_before_K = False
        self.contact_readout = 'signed_h'
        self.contact_gap_epsilon = 0.002
        self.contact_query_radius = 0.03
        self.contact_readout_record = None
        self.reset_frame()

    def process_graph(self, graph, repeat):
        return graph

    def assemble_contact(self, sample, q, mass, local_h, direct, normal, contact_a, internal):
        if self.contact_readout == 'signed_h':
            self.contact_readout_record = None
            return super().assemble_contact(sample, q, mass, local_h, direct, normal, contact_a, internal)
        if self.contact_readout == 'surface_face_floor':
            original = super().assemble_contact(sample, q, mass, local_h, direct, normal, contact_a, internal)
            free = sample['cloth'].vertex_type.reshape(-1) != NodeType.HANDLE
            q_raw = q + (internal + original) / mass[:, None] * free[:, None]
            delta, self.contact_readout_record = self.contact_surface_geometry.project(
                q_raw, mass, free, self.contact_target_geometry_cache[0],
                self.contact_gap_epsilon, self.contact_query_radius, self.record)
            return original + mass[:, None] * delta
        if self.contact_readout == 'target_face_floor':
            return self.assemble_target_contact_floor(sample, q, mass, local_h, direct, normal, contact_a, internal)
        if self.contact_readout in ('vertex_floor', 'face_floor'):
            return self.assemble_contact_floor(sample, q, mass, local_h, direct, normal, contact_a, internal)
        if self.contact_readout != 'gap_bounded':
            raise ValueError(self.contact_readout)
        ci, bi = direct
        free = sample['cloth'].vertex_type.reshape(-1) != 3
        # Keep the original tangent impulse and the actual material/hinge action.
        tangent = contact_a[:, 1:]-(contact_a[:, 1:]*normal).sum(-1, keepdim=True)*normal
        tangent_impulse = mass[ci, None]*local_h[ci, None]*tangent
        external_tangent = q.new_zeros(q.shape).index_add(0, ci, tangent_impulse)
        q_mid = q+(internal+external_tangent)/mass[:, None]*free[:, None]
        gap = ((q_mid[ci]-sample['obstacle'].target_pos[bi])*normal).sum(-1)
        # Reinterpret the existing normal channel; do not add a gain network.
        eta = (contact_a[:, 0]+1)*0.5
        count = q.new_zeros(len(q)).index_add(0, ci, q.new_ones(len(ci)))
        weights = count[ci].reciprocal()
        deficit = (self.contact_gap_epsilon-gap).clamp_min(0)
        edge_delta = (weights*eta*deficit)[:, None]*normal
        normal_delta = q.new_zeros(q.shape).index_add(0, ci, edge_delta)
        if self.record:
            with torch.no_grad():
                valid = free[ci]
                self.contact_readout_record = dict(mode=self.contact_readout,
                    epsilon_m=self.contact_gap_epsilon, pairs=len(ci), free_pairs=int(valid.sum()),
                    supported_nodes=int((count>0).sum()),
                    eta_min=float(eta.min().detach()) if len(ci) else None,
                    eta_max=float(eta.max().detach()) if len(ci) else None,
                    gap_mid_min_m=float(gap[valid].min().detach()) if bool(valid.any()) else None,
                    active_fraction=float((deficit[valid]>0).float().mean()) if bool(valid.any()) else None,
                    normal_delta_rms_m=float((normal_delta*free[:, None]).detach().square().mean().sqrt()),
                    support_weight_sum_min=float(weights.new_zeros(len(q)).index_add(0,ci,weights)[count>0].min()) if len(ci) else None,
                    support_weight_sum_max=float(weights.new_zeros(len(q)).index_add(0,ci,weights)[count>0].max()) if len(ci) else None)
        else:
            self.contact_readout_record = None
        return external_tangent+mass[:, None]*normal_delta


    def reset_frame(self):
        super().reset_frame()
        self.contact_face_plane_cache = None
        self.contact_face_queries = 0
        self.contact_target_geometry_cache = None
        self.contact_target_queries = 0
        self.contact_target_precomputations = 0
        self.contact_surface_geometry = None
        self._contact_backward_correspondence = None

    def prepare_contact_face_plane(self, sample):
        """Native collision-plane correspondence, once outside response checkpoints.

        The query is exactly the native current-cloth / current-body-face-centroid
        rule, restricted to nodes already supported by a world edge. Body geometry
        is prescribed input. Only the later assembly differentiates through q_raw.
        """
        if self.contact_face_plane_cache is not None:
            return
        from pytorch3d.ops import knn_points
        from utils.cloth_and_material import FaceNormals
        with torch.no_grad():
            cloth, body = sample['cloth'], sample['obstacle']
            supported = torch.unique(sample[DIRECT].edge_index[0], sorted=True)
            faces = body.faces_batch.T
            if len(supported):
                current_centers = body.pos[faces].mean(dim=1)
                indices = knn_points(cloth.pos[supported].unsqueeze(0),
                    current_centers.unsqueeze(0), K=1).idx[0, :, 0]
                points = body.target_pos[faces].mean(dim=1)[indices]
                normals = FaceNormals()(body.target_pos.unsqueeze(0), faces.unsqueeze(0))[0][indices]
                self.contact_face_queries += 1
            else:
                points = body.target_pos.new_empty((0, 3))
                normals = body.target_pos.new_empty((0, 3))
            self.contact_face_plane_cache = (supported, points, normals)

    def prepare_contact_target_geometry(self, sample):
        """Cache prescribed next-body geometry once, not the proposal correspondence."""
        if self.contact_target_geometry_cache is not None:
            return
        from utils.cloth_and_material import FaceNormals
        with torch.no_grad():
            body = sample['obstacle']
            faces = body.faces_batch.T
            positions = body.target_pos
            centers = positions[faces].mean(dim=1)
            normals = FaceNormals()(positions.unsqueeze(0), faces.unsqueeze(0))[0]
            original = torch.unique(sample[DIRECT].edge_index[0], sorted=True)
            self.contact_target_geometry_cache = (original, positions, centers, normals)
            self.contact_target_precomputations += 1

    def assemble_target_contact_floor(self, sample, q, mass, local_h, direct, normal, contact_a, internal):
        """Refresh support/planes at q_raw while preserving every original action.

        Discrete KNN choices are detached. The actual one-sided displacement
        differentiates through q_raw. This remains a local face-plane control.
        """
        from pytorch3d.ops import knn_points
        original = super().assemble_contact(sample, q, mass, local_h, direct, normal, contact_a, internal)
        free = sample['cloth'].vertex_type.reshape(-1) != NodeType.HANDLE
        q_raw = q + (internal + original) / mass[:, None] * free[:, None]
        inherited, positions, centers, face_normals = self.contact_target_geometry_cache
        correspondence = self._contact_backward_correspondence
        if correspondence is not None and 'selected' in correspondence:
            ci, points, normals = correspondence['selected']
        else:
            with torch.no_grad():
                near = knn_points(q_raw.detach().unsqueeze(0), positions.unsqueeze(0), K=1)
                allowed = sample['obstacle'].vertex_type.reshape(-1)[near.idx[0, :, 0]] != NodeType.OBSTACLE_OMIT
                near_mask = (near.dists[0, :, 0] <= self.contact_query_radius ** 2) & allowed & free
                added = torch.nonzero(near_mask, as_tuple=False).reshape(-1)
                ci = torch.unique(torch.cat((inherited, added)), sorted=True)
                if len(ci):
                    ids = knn_points(q_raw.detach()[ci].unsqueeze(0), centers.unsqueeze(0), K=1).idx[0, :, 0]
                    points, normals = centers[ids], face_normals[ids]
                else:
                    points = positions.new_empty((0, 3))
                    normals = face_normals.new_empty((0, 3))
                self.contact_target_queries += 1
            if correspondence is not None:
                correspondence['selected'] = ci, points, normals
        gap = ((q_raw[ci] - points) * normals).sum(-1)
        deficit = (self.contact_gap_epsilon - gap).clamp_min(0)
        delta = q.new_zeros(q.shape).index_add(0, ci, deficit[:, None] * normals)
        if getattr(self, 'elastic_contact', False):
            delta = self.coordinate_contact(sample, q_raw, delta, ci, normals)
        if self.record:
            with torch.no_grad():
                valid = free[ci]
                self.contact_readout_record = dict(mode=self.contact_readout,
                    epsilon_m=self.contact_gap_epsilon, query_radius_m=self.contact_query_radius,
                    query_scope='q_raw and prescribed next-body geometry; detached discrete choices',
                    pairs=len(ci), free_pairs=int(valid.sum()), supported_nodes=len(ci),
                    inherited_support_nodes=len(inherited), added_target_support_nodes=len(ci)-len(inherited),
                    face_queries=self.contact_target_queries,
                    target_geometry_precomputations=self.contact_target_precomputations,
                    active_fraction=float((deficit[valid]>0).float().mean()) if bool(valid.any()) else None,
                    gap_raw_min_m=float(gap[valid].min()) if bool(valid.any()) else None,
                    added_normal_rms_m=float((delta*free[:, None]).square().mean().sqrt()),
                    original_external_rms_m=float((original/mass[:, None]*free[:, None]).square().mean().sqrt()))
        else:
            self.contact_readout_record = None
        return original + mass[:, None] * delta

    def assemble_contact_floor(self, sample, q, mass, local_h, direct, normal, contact_a, internal):
        """Keep every original impulse; add only a violated plane's normal deficit.

        This is a one-sided local readout control, not an exact triangle collision
        solver or a guarantee for unsupported nodes / interacting constraints.
        """
        original = super().assemble_contact(sample, q, mass, local_h, direct, normal, contact_a, internal)
        free = sample['cloth'].vertex_type.reshape(-1) != 3
        q_raw = q + (internal + original) / mass[:, None] * free[:, None]
        if self.contact_readout == 'face_floor':
            ci, points, floor_normal = self.contact_face_plane_cache
            weights = q.new_ones(len(ci))
        else:
            ci, bi = direct
            points, floor_normal = sample['obstacle'].target_pos[bi], normal
            count = q.new_zeros(len(q)).index_add(0, ci, q.new_ones(len(ci)))
            weights = count[ci].reciprocal()
        gap = ((q_raw[ci] - points) * floor_normal).sum(-1)
        deficit = (self.contact_gap_epsilon - gap).clamp_min(0)
        delta = q.new_zeros(q.shape).index_add(0, ci, (weights * deficit)[:, None] * floor_normal)
        if self.record:
            with torch.no_grad():
                valid = free[ci]
                self.contact_readout_record = dict(mode=self.contact_readout,
                    epsilon_m=self.contact_gap_epsilon, pairs=len(ci), free_pairs=int(valid.sum()),
                    supported_nodes=int(torch.unique(ci).numel()), face_queries=self.contact_face_queries,
                    active_fraction=float((deficit[valid] > 0).float().mean()) if bool(valid.any()) else None,
                    gap_raw_min_m=float(gap[valid].min()) if bool(valid.any()) else None,
                    added_normal_rms_m=float((delta * free[:, None]).square().mean().sqrt()),
                    original_external_rms_m=float((original / mass[:, None] * free[:, None]).square().mean().sqrt()))
        else:
            self.contact_readout_record = None
        return original + mass[:, None] * delta

    def forward(self, sample, latents, repeat, return_context=False, material_multiplier=None):
        if self.contact_readout == 'face_floor':
            self.prepare_contact_face_plane(sample)
        elif self.contact_readout in ('target_face_floor', 'surface_face_floor'):
            self.prepare_contact_target_geometry(sample)
            if self.contact_readout == 'surface_face_floor' and self.contact_surface_geometry is None:
                from arcon.models.surface_contact import SurfaceGeometry
                with torch.no_grad():
                    self.contact_surface_geometry = SurfaceGeometry(sample)
        if not torch.is_grad_enabled():
            return super().forward(sample, latents, repeat, return_context=return_context,
                                   material_multiplier=material_multiplier)
        # Rest geometry is a detached, once-per-frame quantity. Compute it
        # outside the checkpoint so nested hinge autograd sees the identical
        # operation sequence in the first pass and every recomputation.
        if self.rest_gram is None:
            cloth = sample['cloth']
            with torch.no_grad():
                _, derivative, tri_ids, _ = hinge_basis(
                    cloth.rest_pos, cloth.faces_batch.T,
                    cloth.f_connectivity, cloth.f_connectivity_edges)
                self.rest_gram, _, _ = hinge_mass_gram(
                    derivative, tri_ids, cloth.f_connectivity_edges,
                    cloth.v_mass.reshape(-1), cloth.vertex_type.reshape(-1) != 3)
            self.rest_precomputations += 1
        template = copy.copy(sample)
        captured_gram = self.rest_gram
        captured_target_geometry = self.contact_target_geometry_cache
        captured_surface_geometry = self.contact_surface_geometry
        # Discrete choices are detached in the forward definition. A backward
        # recomputation must reuse that branch, not select another radius mask.
        captured_correspondence = {}
        calls = [0]
        def action(q, *values):
            multiplier = values[0] if material_multiplier is not None else None
            values = values[1:] if material_multiplier is not None else values
            first = calls[0] == 0
            calls[0] += 1
            snapshot = (self.rest_gram, self.rest_precomputations, self.metric_record,
                        list(self.metric_history), self.last, self.record,
                        self.contact_target_geometry_cache, self.contact_target_queries,
                        self.contact_target_precomputations, self.contact_surface_geometry)
            if not first:
                # Recompute against the same frame's rest metric even after the
                # outer wrapper has cleared its mutable per-frame cache.
                self.rest_gram = captured_gram
                self.contact_target_geometry_cache = captured_target_geometry
                self.contact_surface_geometry = captured_surface_geometry
                self.record = False
            proposal = copy.copy(template)
            proposal = add_field_to_pyg_batch(proposal, 'pred_pos', q, 'cloth', 'pos')
            used = dict(zip(('cloth','obstacle','mesh','direct','reverse'), values))
            previous_correspondence = self._contact_backward_correspondence
            self._contact_backward_correspondence = captured_correspondence
            try:
                value, context = GeometryRelationResponse.forward(
                    self, proposal, used, repeat, return_context=True, material_multiplier=multiplier)
                result = value, context['internal_increment'], context['contact_increment']
                return result+(context['material_multiplier'],) if multiplier is not None else result
            finally:
                self._contact_backward_correspondence = previous_correspondence
                if not first:
                    (self.rest_gram, self.rest_precomputations, self.metric_record,
                     self.metric_history, self.last, self.record,
                     self.contact_target_geometry_cache, self.contact_target_queries,
                     self.contact_target_precomputations, self.contact_surface_geometry) = snapshot
        tensors = ((material_multiplier,)+tuple(latents.values()) if material_multiplier is not None
                   else tuple(latents.values()))
        outputs = checkpoint(action, sample['cloth'].pred_pos, *tensors,
                             use_reentrant=False, preserve_rng_state=False)
        if return_context:
            context = dict(internal_increment=outputs[1], contact_increment=outputs[2])
            if material_multiplier is not None:
                context['material_multiplier'] = outputs[3]
            return outputs[0], context
        return outputs[0]


class MeshTransport(nn.Module):
    """One shared latent transport used on either the fine or material patch graph."""
    def __init__(self):
        super().__init__()
        self.message = mlp(384)
        self.output = mlp(256, 128, norm=False, zero=True)
        self.norm = nn.LayerNorm(128)

    def forward(self, nodes, features, edges, repeat):
        src, dst = edges
        state = nodes
        def step(h, e):
            values = self.message(torch.cat((h[src], e, h[dst]), -1))
            total = h.new_zeros(h.shape).index_add(0, dst, values)
            return self.norm(.9*h+total)
        for _ in range(int(repeat)):
            state = (checkpoint(step, state, features, use_reentrant=False,
                                preserve_rng_state=False)
                     if torch.is_grad_enabled() else step(state, features))
        return nodes+self.output(torch.cat((nodes, state), -1))


class MaterialPatches:
    """Disjoint connected geodesic-radius patches on the immutable material graph."""
    def __init__(self, rest, edges, areas, radius):
        begin = time.perf_counter()
        points = rest.detach().cpu().double().numpy()
        edge_array = edges.detach().cpu().T.numpy()
        n = len(points)
        adjacency = [dict() for _ in range(n)]
        for a, b in edge_array:
            a, b = int(a), int(b)
            if a != b:
                length = float(np.linalg.norm(points[a]-points[b]))
                adjacency[a][b] = length
                adjacency[b][a] = length
        ids = np.full(n, -1, dtype=np.int64)
        patch = 0
        for seed in range(n):
            if ids[seed] >= 0:
                continue
            queue, distances = [(0., seed)], {seed: 0.}
            while queue:
                distance, node = heapq.heappop(queue)
                if distance > radius or distance != distances[node] or ids[node] >= 0:
                    continue
                ids[node] = patch
                for neighbor, length in adjacency[node].items():
                    candidate = distance+length
                    if ids[neighbor] < 0 and candidate <= radius and candidate < distances.get(neighbor, float('inf')):
                        distances[neighbor] = candidate
                        heapq.heappush(queue, (candidate, neighbor))
            patch += 1
        self.ids = torch.as_tensor(ids, device=rest.device)
        self.count = patch
        weights = areas.reshape(-1)
        sums = weights.new_zeros(patch).index_add(0, self.ids, weights)
        if bool((sums <= 0).any()):
            raise ValueError('A material patch has no positive rest measure')
        self.weights = weights/sums[self.ids]
        src, dst = edges
        cross = self.ids[src] != self.ids[dst]
        keys = self.ids[src[cross]]*patch+self.ids[dst[cross]]
        unique, self.edge_inverse = torch.unique(keys, sorted=True, return_inverse=True)
        self.edges = torch.stack((unique//patch, unique % patch))
        self.cross_mask = cross
        edge_weights = (rest[dst[cross]]-rest[src[cross]]).norm(dim=-1)
        edge_sums = edge_weights.new_zeros(len(unique)).index_add(0, self.edge_inverse, edge_weights)
        self.edge_weights = edge_weights/edge_sums[self.edge_inverse]
        self.preparation_seconds = time.perf_counter()-begin

    def restrict(self, values):
        return values.new_zeros((self.count, values.shape[1])).index_add(
            0, self.ids, values*self.weights[:, None])

    def restrict_edges(self, features):
        return features.new_zeros((self.edges.shape[1], features.shape[1])).index_add(
            0, self.edge_inverse, features[self.cross_mask]*self.edge_weights[:, None])


class SelfInformation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mlp(8)
        self.message = mlp(384)
        self.output = nn.Linear(128, 128)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.topologies = {}

    def query(self, graph, q):
        cloth = graph['cloth']
        mesh = graph[MESH].edge_index
        key = (str(graph.garment_name), len(q), mesh.shape[1], str(q.device))
        fresh = key not in self.topologies
        if fresh:
            self.topologies[key] = MaterialHopTwo(mesh, len(q))
        topology = self.topologies[key]
        edges, record = nonlocal_self_pairs(q, topology, profile=False)
        record['material_preparation_seconds_this_call'] = topology.preparation_seconds if fresh else 0.
        src, dst = edges
        if not len(src):
            return edges, None, record
        h = cloth.average_edge_length.reshape(-1)[0]
        normal = vertex_normals(q, cloth.faces_batch.T)
        relative = q[dst]-q[src]
        moved = q-cloth.pos
        features = torch.cat((relative/h, relative.norm(dim=-1, keepdim=True)/h,
                              (moved[dst]-moved[src])/h,
                              (normal[src]*normal[dst]).sum(-1, keepdim=True)), -1)
        return edges, self.encoder(features), record

    def apply(self, graph, edges, encoded):
        if encoded is None:
            return graph
        src, dst = edges
        nodes = graph['cloth'].node_features
        active = torch.bincount(dst, minlength=len(nodes)) > 0
        def action(h, e):
            values = self.message(torch.cat((h[src], e, h[dst]), -1))
            return self.output(mean_scatter(values, dst, len(h)))*active[:, None]
        delta = (checkpoint(action, nodes, encoded, use_reentrant=False,
                            preserve_rng_state=False) if torch.is_grad_enabled()
                 else action(nodes, encoded))
        graph['cloth'].node_features = nodes+delta
        return graph


class RelationDirection(nn.Module):
    """One normalized relation-coordinate sweep, applied to the actual candidate delta."""
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == 'learned':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1057)
                self.gain = mlp(139, 1, norm=False, zero=True)
                self.gain[-1].bias.data.fill_(math.log(.1/.9))
        self.record = False
        self.last = None

    def coefficients(self, nodes, ids, c, u, previous, scale, mass, materials, tau, kind):
        if self.mode == 'fixed':
            return c.new_full(c.shape, .1)
        if ids.ndim == 1:
            receivers, receiver_mass = nodes[ids], mass[ids]
        else:
            receivers, receiver_mass = nodes[ids].mean(1), mass[ids].mean(1)
        types = c.new_zeros(len(c), 3)
        types[:, kind] = 1.
        conditions = torch.cat((torch.stack((c/scale, u/scale, previous/scale,
                                             receiver_mass/mass.mean()), -1),
                                 materials.expand(len(c), -1), tau.expand(len(c), 1), types), -1)
        return self.gain(torch.cat((receivers, conditions), -1)).sigmoid().flatten()

    def forward(self, graph, q, delta, previous_response, reference):
        if self.mode == 'none':
            self.last = dict(mode='none') if self.record else None
            return delta
        cloth = graph['cloth']
        mass = cloth.v_mass.reshape(-1)
        free = cloth.vertex_type.reshape(-1) != 3
        inverse = free.to(q.dtype)/mass
        nodes = cloth.node_features
        materials = torch.stack([cloth[k].reshape(-1)[0] for k in
                                 ('lame_mu_input', 'lame_lambda_input', 'bending_coeff_input')])
        tau = cloth.timestep.reshape(-1)[0]
        h = cloth.average_edge_length.reshape(-1)[0]
        correction, rows = torch.zeros_like(delta), []
        def assemble(ids, derivative, c, scale, kind, g0, unilateral=False):
            nonlocal correction
            gradient = derivative if derivative.ndim == 3 else derivative[:, None, :]
            index = ids if ids.ndim == 2 else ids[:, None]
            u = (gradient*delta[index]).sum((1, 2))
            projected = (gradient*previous_response[index]).sum((1, 2))
            gram = (gradient.square().sum(-1)*inverse[index]).sum(1)
            denominator = gram+g0
            valid = denominator > 0
            safe = torch.where(valid, denominator, torch.ones_like(denominator))
            if unilateral:
                worsening = (torch.relu(c+u)-torch.relu(c)).clamp_min(0)
            else:
                predicted = c+u
                worsening = predicted.sign()*(predicted.abs()-c.abs()).clamp_min(0)
            gain = self.coefficients(nodes, ids, c, u, projected, scale, mass, materials, tau, kind)
            scalar = gain*worsening/safe*valid
            values = gradient*scalar[:, None, None]*inverse[index][..., None]
            correction = correction.index_add(0, index.reshape(-1), values.reshape(-1, 3))
            if self.record:
                rows.append(dict(kind=kind, relations=len(c), gain_mean=float(gain.detach().mean()) if len(c) else None,
                    c_rms=float(c.detach().square().mean().sqrt()) if len(c) else None,
                    u_rms=float(u.detach().square().mean().sqrt()) if len(c) else None,
                    worsening_rms=float(worsening.detach().square().mean().sqrt()) if len(c) else None))
        ids, l0, g0edge, theta0, g0hinge = reference
        a, b = ids.T
        length = (q[b]-q[a]).norm(dim=-1)
        direction = torch.nn.functional.normalize(q[b]-q[a], dim=-1)/l0[:, None]
        assemble(ids, torch.stack((-direction, direction), 1), length/l0-1., 1., 0, g0edge)
        theta, derivative, tri_ids, _ = hinge_basis(q, cloth.faces_batch.T,
            cloth.f_connectivity, cloth.f_connectivity_edges)
        unique, gradient = unique_hinge_gradient(derivative, tri_ids, cloth.f_connectivity_edges)
        relative = torch.atan2(torch.sin(theta-theta0), torch.cos(theta-theta0))
        assemble(unique, gradient, relative, 1., 1, g0hinge)
        ci, bi = graph[DIRECT].edge_index
        normal = vertex_normals(graph['obstacle'].target_pos, graph['obstacle'].faces_batch.T)[bi]
        gap = ((q[ci]-graph['obstacle'].target_pos[bi])*normal).sum(-1)
        assemble(ci, -normal, .002-gap, h, 2, inverse[ci]*normal.square().sum(-1), unilateral=True)
        corrected = (delta-correction)*free[:, None]
        self.last = (dict(mode=self.mode, rows=rows,
            raw_rms_m=float(delta.detach().square().mean().sqrt()),
            correction_rms_m=float(correction.detach().square().mean().sqrt()),
            corrected_rms_m=float(corrected.detach().square().mean().sqrt())) if self.record else None)
        return corrected


class AdaptiveResponseCell(nn.Module):
    """A single learned cell reused for three same-endpoint working-state updates."""
    _encode_nodes = NativeCore._encode_nodes
    _encode_edges = NativeCore._encode_edges
    _encode = NativeCore._encode
    _decode = NativeCore._decode

    def __init__(self, mcfg):
        super().__init__()
        self.arm = str(mcfg.cell_arm)
        if self.arm not in ARMS:
            raise ValueError(self.arm)
        self.scale_enabled = self.arm in ('Cell-Scale', 'Cell-Joint', 'Cell-ScaleFixed', 'Cell-NetDir')
        self.net_direction = self.arm == 'Cell-NetDir'
        self.patch_radius = float(mcfg.cell_patch_radius)
        self.n_nodefeatures = int(mcfg.n_nodefeatures)
        self.node_encoder = mlp(self.n_nodefeatures)
        self.decoder = mlp(128, 3, norm=False)
        self.edgeset_encoders = nn.ModuleDict({"mesh": mlp(int(mcfg.n_edgefeatures_mesh)),
                                             "world": mlp(int(mcfg.n_edgefeatures_world))})
        sets = make_edgesets_dict(0)
        self.edge_sets = {k: sets[k] for k in ('world_direct', 'world_inverse', 'mesh')}
        self.transport = MeshTransport()
        self.blocks = nn.ModuleList([GraphNetBlock(lambda: plain_mlp(384), lambda: plain_mlp(384), self.edge_sets, 128)
                                     for _ in range(5)])
        for block in self.blocks:
            block.forward = types.MethodType(recomputed_native_block, block)
        self.self_information = SelfInformation()
        self.stage_feedback = ContextStageFeedback()
        material_arm = getattr(mcfg, 'material_response_arm', 'none')
        if material_arm != 'none':
            from arcon.models.material_response import MaterialReadout
            self.response = MaterialReadout(material_arm, float(mcfg.material_response_thickness))
        else:
            self.response = ProposalRelationResponse()
        self.direction = RelationDirection('learned' if self.arm in ('Cell-Dir', 'Cell-Joint', 'Cell-NetDir') else
                                           'fixed' if self.arm in ('Cell-Fixed', 'Cell-ScaleFixed') else 'none')
        self.patches = {}
        self.capture_stages = False
        self.record = False
        self.position_readout = None
        self.stage_records = []
        self.final_record = None
        self.execution = {}
        self.working_values = None

    def direction_reference(self, graph):
        if self.direction.mode == 'none':
            return None
        cloth = graph['cloth']
        rest, mass = cloth.rest_pos, cloth.v_mass.reshape(-1)
        inverse = (cloth.vertex_type.reshape(-1) != 3).to(rest.dtype)/mass
        src, dst = graph[MESH].edge_index
        keys = torch.unique(torch.minimum(src, dst)*len(rest)+torch.maximum(src, dst), sorted=True)
        ids = torch.stack((keys//len(rest), keys % len(rest)), -1)
        lengths = (rest[ids[:, 1]]-rest[ids[:, 0]]).norm(dim=-1)
        with torch.no_grad():
            theta, derivative, tri_ids, _ = hinge_basis(rest, cloth.faces_batch.T,
                cloth.f_connectivity, cloth.f_connectivity_edges)
            unique, gradient = unique_hinge_gradient(derivative, tri_ids, cloth.f_connectivity_edges)
            g0hinge = (gradient.square().sum(-1)*inverse[unique]).sum(1)
        return ids, lengths, inverse[ids].sum(1)/lengths.square(), theta, g0hinge

    def transfer(self, graph, repeat):
        nodes, features, edges = graph['cloth'].node_features, graph[MESH].features, graph[MESH].edge_index
        if not self.scale_enabled:
            graph['cloth'].node_features = self.transport(nodes, features, edges, repeat)
            return dict(path='fine', repeat=int(repeat))
        cloth = graph['cloth']
        key = (str(graph.garment_name), len(nodes), edges.shape[1], str(nodes.device))
        fresh = key not in self.patches
        if fresh:
            weights = surface_weights(cloth.rest_pos, cloth.faces_batch.T, cloth.f_area)
            self.patches[key] = MaterialPatches(cloth.rest_pos, edges, weights, self.patch_radius)
        patches = self.patches[key]
        coarse = patches.restrict(nodes)
        fine = nodes-coarse[patches.ids]
        physical_span = int(repeat)*float(cloth.average_edge_length.reshape(-1)[0])
        coarse_repeat = math.ceil(physical_span/self.patch_radius) if repeat > 0 else 0
        coarse_next = self.transport(coarse, patches.restrict_edges(features), patches.edges, coarse_repeat)
        graph['cloth'].node_features = fine+coarse_next[patches.ids]
        row = dict(path='fine_residual_plus_patch', radius_m=self.patch_radius,
            patches=patches.count, coarse_edges=patches.edges.shape[1], repeat=coarse_repeat,
            physical_span_m=physical_span,
            preparation_seconds_this_call=patches.preparation_seconds if fresh else 0.)
        if self.record:
            row.update(fine_residual_rms=float(fine.detach().square().mean().sqrt()),
                coarse_delta_rms=float((coarse_next-coarse).detach().square().mean().sqrt()),
                restrict_prolong_max=float((patches.restrict(coarse[patches.ids])-coarse).detach().abs().max()))
        return row

    def refresh_working_state(self, graph, work):
        """Refresh geometric conditions while preserving accumulated graph context.

        During training activation recomputation changes storage, not the update.
        The working positions remain differentiable across the three rounds."""
        if not torch.is_grad_enabled():
            return self.stage_feedback(graph, work)
        template = copy.copy(graph)
        def action(*values):
            current = _assign_latents(copy.copy(template), values[:5])
            proposal = add_field_to_pyg_batch(copy.copy(current), 'pred_pos', values[5], 'cloth', 'pos')
            try:
                current = self.stage_feedback(current, proposal)
                return tuple(capture_latents(current).values())
            finally:
                # There is no global-token consumer in this backbone. Do not
                # retain the hidden context activations outside the checkpoint.
                self.stage_feedback.context = None
        output = checkpoint(action, *capture_latents(graph).values(), work['cloth'].pred_pos,
                            use_reentrant=False, preserve_rng_state=False)
        return _assign_latents(graph, output)

    def forward(self, sample, repeat):
        """Advance one physical step through three geometry-conditioned update rounds.

        Each round refreshes graph conditions at the current working positions,
        queries spatial self relations, propagates the graph representation, and
        combines a proposed displacement with the material/body relation response.
        Working positions are updated between rounds; the normalizer statistics
        follow the original once-per-physical-step accumulation."""
        graph = self._encode(sample)
        free = graph['cloth'].vertex_type.reshape(-1) != 3
        work = self.position_readout(self._decode(graph.clone()), is_training=False)
        # Establish the inertial reference directly, not by subtracting large
        # decoder displacements from global positions. HANDLE retains its native
        # pred_acceleration convention and prescribed position/velocity.
        cloth = work['cloth']
        for key, inertial in (('pred_pos', cloth.pos+cloth.velocity),
                              ('pred_velocity', cloth.velocity),
                              ('pred_acceleration', torch.zeros_like(cloth.pos))):
            values = torch.where(free[:, None], inertial, cloth[key])
            work = add_field_to_pyg_batch(work, key, values, 'cloth', 'pos')
        previous_response = torch.zeros_like(graph['cloth'].pos)
        material_multiplier = (graph['cloth'].pos.new_zeros(len(graph['cloth'].f_connectivity))
                               if getattr(self.response, 'elastic_bending', False) else None)
        reference = self.direction_reference(graph)
        self.stage_records, self.final_record = [], None
        self.execution = dict(organization=self.arm, shared_cells=3, unique_typed_blocks=5,
            executed_typed_blocks=15, response_processors=0, context_tokens=0,
            response_calls=3, self_queries=3, self_message_calls=6,
            single_physical_endpoint=True, scale_enabled=self.scale_enabled,
            relation_feedback_scope='candidate_J' if self.net_direction else 'applied_J',
            direction_scope='complete_candidate' if self.net_direction else 'decoder_innovation',
            activation_recomputation='typed/transport/self/readout/working-state;unchanged operations', cells=[])
        for cell in range(3):
            graph = self.refresh_working_state(graph, work)
            before = self.position_readout(self._decode(graph.clone()), is_training=False)['cloth'].pred_acceleration
            q = work['cloth'].pred_pos
            edges, encoded, query = self.self_information.query(graph, q)
            graph = self.blocks[0](graph)
            graph = self.self_information.apply(graph, edges, encoded)
            transfer = self.transfer(graph, repeat)
            for block in self.blocks[1:]:
                graph = block(graph)
            graph = self.self_information.apply(graph, edges, encoded)
            after = self.position_readout(self._decode(graph.clone()), is_training=False)['cloth'].pred_acceleration
            raw = (after-before)*free[:, None]
            if self.net_direction:
                # J sees the uncorrected decoder proposal. Its output is only a
                # candidate relation hint until the complete increment is accepted.
                proposal = add_response(work.clone(), raw)
                q_hat = proposal['cloth'].pred_pos
                response, context = self.response(proposal, capture_latents(graph), repeat, return_context=True)
                direction_input = raw+response
                delta = self.direction(graph, q, direction_input, previous_response, reference)
                work = add_response(work, delta)
            else:
                # Preserve the five executed arms' exact operation order.
                delta = self.direction(graph, q, raw, previous_response, reference)
                proposal = add_response(work.clone(), delta)
                q_hat = proposal['cloth'].pred_pos
                if material_multiplier is not None:
                    response, context = self.response(proposal, capture_latents(graph), repeat,
                        return_context=True, material_multiplier=material_multiplier)
                    material_multiplier = context['material_multiplier']
                else:
                    response, context = self.response(proposal, capture_latents(graph), repeat, return_context=True)
                work = add_response(proposal, response)
            # For NetDir this deliberately means CANDIDATE assembled response,
            # not the actually applied delta-minus-decoder effective correction.
            previous_response = response
            stage = (cell+1)*5
            if self.capture_stages:
                row = dict(stage=stage, q_hat=q_hat.detach(), refined=work['cloth'].pred_pos.detach(),
                    increment=response.detach(), internal_increment=context['internal_increment'].detach(),
                    contact_increment=context['contact_increment'].detach(), raw_innovation=raw.detach(),
                    direction_innovation=delta.detach())
                if self.net_direction:
                    # Do not label candidate J vectors as executed pure J updates.
                    row = dict(stage=stage, q_hat=q_hat.detach(), refined=work['cloth'].pred_pos.detach(),
                        q_before=q.detach(), raw_innovation=raw.detach(),
                        direction_input_increment=direction_input.detach(), accepted_increment=delta.detach(),
                        candidate_relation_increment=response.detach(),
                        candidate_internal_increment=context['internal_increment'].detach(),
                        candidate_contact_increment=context['contact_increment'].detach(),
                        candidate_response_endpoint=(q_hat+response).detach(),
                        effective_relation_increment=(delta-raw).detach())
                if stage < 15:
                    self.stage_records.append(row)
                else:
                    self.final_record = dict(carried_proposal=q_hat.detach(),
                        refined=work['cloth'].pred_pos.detach(), decoder_innovation=raw.detach(),
                        direction_innovation=delta.detach())
                    if self.net_direction:
                        self.final_record = {key:value for key,value in row.items() if key!='stage'}
                        self.final_record['decoder_innovation'] = raw.detach()
                        self.final_record['carried_proposal'] = q_hat.detach()
            if self.record:
                if self.net_direction:
                    self.direction.last.update(input_scope='decoder_plus_candidate_J',
                        geometry_base='accepted_cell_input_q', response_hint='candidate_not_applied',
                        accepted_writeback_max=float((work['cloth'].pred_pos-q-delta).detach().abs().max()))
                self.execution['cells'].append(dict(stage=stage, query=query, transfer=transfer,
                    direction=self.direction.last, response_metric=self.response.metric_statistics(),
                    contact_readout=self.response.contact_readout_record))
                if getattr(self.response, 'material_arm', None):
                    self.execution['cells'][-1]['material_response'] = self.response.material_statistics()
        self.working_values = {key: work['cloth'][key] for key in
            ('pred_pos', 'pred_velocity', 'pred_acceleration')}
        return graph


class ARCoNModel(NativeModel):
    def forward(self, sample, repeat=None, is_training=True):
        core = self._learned_model
        core.position_readout = self.get_position
        core.response.reset_frame()
        core.response.record = core.record
        core.direction.record = core.record
        core.stage_feedback.record = core.record
        try:
            sample = self.prepare_inputs(sample, is_training=is_training)
            if repeat is None:
                repeat = max(1, int(self.propagation_distance/sample['cloth'].average_edge_length.item()))
            graph = core(sample, int(repeat))
            result = self.get_position(core._decode(graph), is_training=is_training)
            free = result['cloth'].vertex_type.reshape(-1) != 3
            # Sole physical writeback uses the actual working-state increments,
            # without cancellation against a separately translated global q.
            for key, values in core.working_values.items():
                result = add_field_to_pyg_batch(result, key,
                    torch.where(free[:, None], values, result['cloth'][key]), 'cloth', 'pos')
            return result
        finally:
            core.position_readout = None
            core.working_values = None
            core.stage_feedback.context = None
            core.response.rest_gram = None


def EncodeProcessDecode(mcfg):
    return AdaptiveResponseCell(mcfg)
