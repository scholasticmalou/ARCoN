"""Differentiable material and body-contact relation responses.

Relation features combine current geometry with graph context; assembled
contributions update garment positions. Upstream graph primitives are imported
from the attributed third-party implementation."""
from __future__ import annotations
import functools
import torch
from torch import nn
from torch.nn import functional as Fnn
from models.core.pb4u import EncodeProcessDecode as NativeCore, GraphNetBlock, GraphAggrBlock
from models.pb4u import Model as NativeModel
from utils.common import add_field_to_pyg_batch

ACTIVE_MODE = 'R_geom'


def mean_scatter(values, indices, size):
    out = values.new_zeros((size, values.shape[-1])).index_add(0, indices, values)
    count = values.new_zeros(size).index_add(0, indices, values.new_ones(len(indices)))
    return out/count.clamp_min(1)[:, None]


def mlp(nin, nout=128, norm=True, zero=False):
    layers = [nn.Linear(nin, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, nout)]
    if zero:
        nn.init.zeros_(layers[-1].weight); nn.init.zeros_(layers[-1].bias)
    if norm:
        layers.append(nn.LayerNorm(nout))
    return nn.Sequential(*layers)


def face_geometry(q, faces, dm):
    tri = q[faces]
    deformation = torch.stack((tri[:, 0]-tri[:, 2], tri[:, 1]-tri[:, 2]), -1) @ dm
    strain = .5*(deformation.transpose(-1, -2) @ deformation-torch.eye(2, device=q.device, dtype=q.dtype))
    cross = torch.linalg.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    return deformation, strain, cross.norm(dim=-1)/2


def vertex_normals(q, faces):
    tri = q[faces]
    normals = torch.linalg.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    out = q.new_zeros(q.shape)
    for j in range(3):
        out = out.index_add(0, faces[:, j], normals)
    return Fnn.normalize(out, dim=-1)


def hinge_theta(local, slots):
    # Same signed dihedral as the original bending criterion.
    normals = Fnn.normalize(torch.linalg.cross(local[:, :, 1]-local[:, :, 0],
                                               local[:, :, 2]-local[:, :, 0]), dim=-1)
    ids = torch.arange(len(local), device=local.device)
    edge = local[ids, 0, slots[:, 1]]-local[ids, 0, slots[:, 0]]
    tangent = Fnn.normalize(edge, dim=-1)
    sine = (tangent*torch.linalg.cross(normals[:, 0], normals[:, 1])).sum(-1)
    cosine = (normals[:, 0]*normals[:, 1]).sum(-1)
    return torch.atan2(sine, cosine)


def hinge_basis(q, faces, connectivity, shared_edges):
    tri_ids = faces[connectivity]
    slots = (tri_ids[:, 0, :, None] == shared_edges[:, None, :]).long().argmax(1)
    keep_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        local = q[tri_ids]
        if not local.requires_grad:
            local.requires_grad_(True)
        theta = hinge_theta(local, slots)
        derivative, = torch.autograd.grad(theta.sum(), local, create_graph=keep_graph)
    if not keep_graph:
        theta = theta.detach(); derivative = derivative.detach()
    return theta, derivative, tri_ids, slots


class RelationResponse(nn.Module):
    def __init__(self, edge_sets, mode):
        super().__init__()
        self.mode = mode
        self.node_encoder = mlp(15)
        self.edge_encoder = mlp(8)
        self.face_encoder = mlp(13)
        self.world_encoder = mlp(13)
        self.hinge_encoder = mlp(3)
        self.node_fusion = mlp(128*3)
        self.edge_fusion = mlp(128*3)
        self.world_fusion = mlp(128*2)
        self.body_fusion = mlp(128*2)
        node_fn = functools.partial(mlp, 384)
        edge_fn = functools.partial(mlp, 384)
        self.aggr_processor = GraphAggrBlock(node_fn, edge_fn, edge_sets, 128, 2)
        self.processor_steps = nn.ModuleList([GraphNetBlock(node_fn, edge_fn, edge_sets, 128) for _ in range(2)])
        self.edge_head = mlp(384, 1, norm=False, zero=True)
        self.hinge_head = mlp(384, 1, norm=False, zero=True)
        self.contact_head = mlp(384, 4, norm=False, zero=True)
        self.record = False
        self.last = None

    def process_graph(self, graph, repeat):
        graph = self.aggr_processor(graph, repeat)
        for block in self.processor_steps:
            graph = block(graph)
        return graph

    def condition_hinge_coefficient(self, sample, derivative, tri_ids, shared, coefficient):
        # Default path preserves the original readout, without tensor operations.
        return coefficient

    def hinge_record(self):
        return {}

    def assemble_hinge(self, sample, q, mass, rest, theta, derivative, tri_ids,
                       shared, hinge_h, hinge_a, material_multiplier=None):
        inverse_hinge_mass = (1/mass[tri_ids]).sum((1, 2))-(1/mass[shared]).sum(1)
        coefficient = inverse_hinge_mass.reciprocal()*hinge_h*hinge_a
        coefficient = self.condition_hinge_coefficient(sample, derivative, tri_ids, shared, coefficient)
        impulse = derivative*(hinge_h*coefficient)[:, None, None, None]
        return impulse, coefficient, material_multiplier

    def assemble_contact(self, sample, q, mass, local_h, direct, normal, contact_a, internal):
        """Original signed-h contact assembly; subclasses may change only this seam."""
        ci = direct[0]
        tangent = contact_a[:, 1:]-(contact_a[:, 1:]*normal).sum(-1, keepdim=True)*normal
        contact_impulse = mass[ci, None]*local_h[ci, None]*(normal*contact_a[:, :1]+tangent)
        return q.new_zeros(q.shape).index_add(0, ci, contact_impulse)

    def forward(self, sample, latents, repeat, return_context=False, material_multiplier=None):
        """Assemble a position increment from current material and body-contact relations.

        Geometry supplies local relation coordinates; graph features supply learned
        coefficients. Returned context exposes the same assembled increments for
        the next within-step round and optional execution records."""
        cloth, body = sample['cloth'], sample['obstacle']
        q = cloth.pred_pos
        rest, mass = cloth.rest_pos, cloth.v_mass.reshape(-1)
        h = cloth.average_edge_length.reshape(-1)[0]
        faces = cloth.faces_batch.T
        edges = sample['cloth', 'mesh_edge', 'cloth'].edge_index
        src, dst = edges
        n = len(q)
        keys, inverse = torch.unique(torch.minimum(src, dst)*n+torch.maximum(src, dst), sorted=True, return_inverse=True)
        unique = torch.stack((keys//n, keys % n), -1)
        pair_keys = torch.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], dim=-1).values
        face_edge_ids = torch.searchsorted(keys, pair_keys[:, :, 0]*n+pair_keys[:, :, 1])
        lengths0 = (rest[unique[:, 0]]-rest[unique[:, 1]]).norm(dim=-1)
        local_h = mean_scatter(lengths0[:, None].repeat(2, 1), unique.T.reshape(-1), n).flatten()
        mu = cloth.lame_mu.reshape(-1)[0]
        lam = cloth.lame_lambda.reshape(-1)[0]
        materials = torch.stack([cloth.lame_mu_input.reshape(-1)[0], cloth.lame_lambda_input.reshape(-1)[0],
                                 cloth.bending_coeff_input.reshape(-1)[0]])
        deform, strain, area = face_geometry(q, faces, cloth.Dm_inv)
        slot = deform if self.mode == 'R_geom' else deform @ (2*mu*strain+lam*strain.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]*torch.eye(2, device=q.device))/(2*mu+lam)
        face_raw = torch.cat((strain[:, 0, 0:1], strain[:, 1, 1:2], strain[:, 0, 1:2],
                              (area/cloth.f_area.flatten())[:, None], materials.expand(len(faces), -1), slot.reshape(-1, 6)), -1)
        face_lat = self.face_encoder(face_raw)
        face_nodes = mean_scatter(face_lat[:, None].expand(-1, 3, -1).reshape(-1, 128), faces.reshape(-1), n)
        face_edges = mean_scatter(face_lat[:, None].expand(-1, 3, -1).reshape(-1, 128), face_edge_ids.reshape(-1), len(unique))
        normals = vertex_normals(q, faces)
        tau = cloth.timestep.reshape(-1)[0]
        node_raw = torch.cat(((q-cloth.pos)/h, cloth.velocity/h, normals,
                              (cloth.vertex_type.reshape(-1) == 3).to(q.dtype)[:, None],
                              (mass/mass.mean())[:, None], materials.expand(n, -1), tau.expand(n, 1)), -1)
        next_body = body.target_pos
        body_normal = vertex_normals(next_body, body.faces_batch.T)
        body_raw = torch.cat(((next_body-body.pos)/h, body.velocity/h, body_normal,
                              q.new_zeros(len(next_body), 1), -q.new_ones(len(next_body), 1),
                              -q.new_ones(len(next_body), 3), tau.expand(len(next_body), 1)), -1)
        relative = q[dst]-q[src]
        reference = rest[dst]-rest[src]
        mesh_raw = torch.cat((relative/h, reference/h, relative.norm(dim=-1, keepdim=True)/h,
                              reference.norm(dim=-1, keepdim=True)/h), -1)
        direct = sample['cloth', 'world_edge', 'obstacle'].edge_index
        ci, bi = direct
        rel = q[ci]-next_body[bi]
        normal = body_normal[bi]
        motion = (q[ci]-cloth.pos[ci])-(next_body[bi]-body.pos[bi])
        slide = motion-(motion*normal).sum(-1, keepdim=True)*normal
        world_raw = torch.cat((rel/h, normal, (rel*normal).sum(-1, keepdim=True)/h, motion/h, slide/h), -1)
        # Reverse-world ordering is inherited from the original positional graph.
        reverse = sample['obstacle', 'world_edge', 'cloth'].edge_index
        rci, rbi = reverse[1], reverse[0]
        rnormal = body_normal[rbi]
        rrel = q[rci]-next_body[rbi]
        rmotion = (q[rci]-cloth.pos[rci])-(next_body[rbi]-body.pos[rbi])
        rslide = rmotion-(rmotion*rnormal).sum(-1, keepdim=True)*rnormal
        reverse_raw = torch.cat((-rrel/h, rnormal, (rrel*rnormal).sum(-1, keepdim=True)/h, -rmotion/h, -rslide/h), -1)
        graph = sample.clone()
        graph['cloth'].node_features = self.node_fusion(torch.cat((latents['cloth'], self.node_encoder(node_raw), face_nodes), -1))
        graph['obstacle'].node_features = self.body_fusion(torch.cat((latents['obstacle'], self.node_encoder(body_raw)), -1))
        graph['cloth', 'mesh_edge', 'cloth'].features = self.edge_fusion(torch.cat((latents['mesh'], self.edge_encoder(mesh_raw), face_edges[inverse]), -1))
        graph['cloth', 'world_edge', 'obstacle'].features = self.world_fusion(torch.cat((latents['direct'], self.world_encoder(world_raw)), -1))
        graph['obstacle', 'world_edge', 'cloth'].features = self.world_fusion(torch.cat((latents['reverse'], self.world_encoder(reverse_raw)), -1))
        graph = self.process_graph(graph, repeat)
        node = graph['cloth'].node_features
        edge_lat = mean_scatter(graph['cloth', 'mesh_edge', 'cloth'].features, inverse, len(unique))
        edge_a = torch.tanh(self.edge_head(torch.cat((node[unique].mean(1), edge_lat, face_edges), -1))).flatten()
        edge_mass = (1/mass[unique]).sum(-1).reciprocal()
        edge_coefficient = edge_mass*lengths0*edge_a
        edge_vector = q[unique[:, 1]]-q[unique[:, 0]]
        edge_direction = Fnn.normalize(edge_vector, dim=-1)
        edge_impulse = edge_coefficient[:, None]*edge_direction
        internal = q.new_zeros(q.shape).index_add(0, unique[:, 1], edge_impulse).index_add(0, unique[:, 0], -edge_impulse)
        con, shared = cloth.f_connectivity, cloth.f_connectivity_edges
        theta, derivative, tri_ids, slots = hinge_basis(q, faces, con, shared)
        hinge_h = (rest[shared[:, 0]]-rest[shared[:, 1]]).norm(dim=-1)
        hinge_length = (q[shared[:, 0]]-q[shared[:, 1]]).norm(dim=-1)
        hinge_lat = self.hinge_encoder(torch.stack((theta.sin(), theta.cos(), hinge_length/hinge_h), -1))
        hinge_a = torch.tanh(self.hinge_head(torch.cat((node[tri_ids].mean((1, 2)), face_lat[con].mean(1), hinge_lat), -1))).flatten()
        # Pair triangles duplicate the two shared nodes: count them only once in m_e.
        hinge_impulse, hinge_coefficient, next_multiplier = self.assemble_hinge(
            sample, q, mass, rest, theta, derivative, tri_ids, shared,
            hinge_h, hinge_a, material_multiplier)
        internal = internal.index_add(0, tri_ids.reshape(-1), hinge_impulse.reshape(-1, 3))
        contact_a = torch.tanh(self.contact_head(torch.cat((node[ci], graph['obstacle'].node_features[bi],
                                      graph['cloth', 'world_edge', 'obstacle'].features), -1)))
        external = self.assemble_contact(sample, q, mass, local_h, direct, normal, contact_a, internal)
        full = (internal+external)/mass[:, None]
        free = cloth.vertex_type.reshape(-1) != 3
        increment = full*free[:, None]
        if self.record:
            self.last = {'q_hat': q, 'increment': increment, 'internal': internal, 'external': external,
                         'full_increment': full, 'mass': mass, 'free': free, 'unique': unique,
                         'edge_coefficients': edge_coefficient, 'hinge_coefficients': hinge_coefficient,
                         'edge_direction': edge_direction, 'hinge_derivative': derivative, 'hinge_tri_ids': tri_ids,
                         'hinge_slots': slots, 'hinge_h': hinge_h, 'theta': theta, 'repeat': repeat,
                         'face_slot': slot, 'face_raw': face_raw, 'contact_pairs': len(ci),
                         # Actual used tensors for optional scalar tracing; no new model operation.
                         'contact_coefficients': contact_a,
                         'contact_normals': normal, 'contact_slide': slide,
                         'contact_indices': direct}
            self.last.update(self.hinge_record())
        else:
            self.last = None
        if return_context:
            # Actual assembled displacement components, not detached log fields.
            # Default callers retain the original tensor-only return convention.
            context = dict(q_hat=q, internal_increment=internal/mass[:, None]*free[:, None],
                           contact_increment=external/mass[:, None]*free[:, None],
                           strain=strain, area=area, normals=normals,
                           body_normals=body_normal, direct_indices=direct)
            if next_multiplier is not None:
                context['material_multiplier'] = next_multiplier
            return increment, context
        return increment


class EncodeProcessDecode(NativeCore):
    response_class = RelationResponse

    def __init__(self, mcfg):
        super().__init__(mcfg)
        self.response = self.response_class(self.aggr_processor.edge_sets, ACTIVE_MODE)
        self.anchor_latents = None

    def forward(self, sample, repeat):
        sample = self._encode(sample)
        sample = self.aggr_processor(sample, repeat)
        for block in self.processor_steps:
            sample = block(sample)
        self.anchor_latents = {'cloth': sample['cloth'].node_features, 'obstacle': sample['obstacle'].node_features,
            'mesh': sample['cloth', 'mesh_edge', 'cloth'].features,
            'direct': sample['cloth', 'world_edge', 'obstacle'].features,
            'reverse': sample['obstacle', 'world_edge', 'cloth'].features}
        return self._decode(sample)


class RelationResponseModel(NativeModel):
    def forward(self, sample, repeat=None, is_training=True):
        native_statistics = is_training and not getattr(self, 'native_frozen', False)
        sample = self.prepare_inputs(sample, is_training=native_statistics)
        if repeat is None:
            repeat = max(1, int(self.propagation_distance/sample['cloth'].average_edge_length.item()))
        core = self._learned_model
        sample = core(sample, repeat)
        # Exactly one original decoder unnormalization/statistics update.
        sample = self.get_position(sample, is_training=native_statistics)
        increment = core.response(sample, core.anchor_latents, repeat)
        for key in ('pred_pos', 'pred_velocity', 'pred_acceleration'):
            sample = add_field_to_pyg_batch(sample, key, sample['cloth'][key]+increment, 'cloth', 'pos')
        core.anchor_latents = None
        return sample
