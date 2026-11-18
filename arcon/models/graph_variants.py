"""Graph-backbone construction helpers retained for checkpoint-compatible model assembly."""
from __future__ import annotations

import copy
import types
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from models.core.pb4u import EncodeProcessDecode as NativeCore, GraphAggrBlock, GraphNetBlock
from models.core.relation_response import EncodeProcessDecode as ResponseCore, RelationResponseModel, face_geometry, vertex_normals, mean_scatter, mlp
from utils.common import add_field_to_pyg_batch

MESH = ('cloth', 'mesh_edge', 'cloth')
DIRECT = ('cloth', 'world_edge', 'obstacle')
REVERSE = ('obstacle', 'world_edge', 'cloth')


def zero_linear(nin, nout=128):
    layer = nn.Linear(nin, nout)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def capture_latents(sample):
    return dict(cloth=sample['cloth'].node_features,
                obstacle=sample['obstacle'].node_features,
                mesh=sample[MESH].features, direct=sample[DIRECT].features,
                reverse=sample[REVERSE].features)


def add_response(sample, increment):
    # These are native frame-displacement/second-difference coordinates.
    for key in ('pred_pos', 'pred_velocity', 'pred_acceleration'):
        sample = add_field_to_pyg_batch(
            sample, key, sample['cloth'][key] + increment, 'cloth', 'pos')
    return sample



def _assign_latents(sample, values):
    sample['cloth'].node_features, sample['obstacle'].node_features = values[:2]
    sample[MESH].features, sample[DIRECT].features, sample[REVERSE].features = values[2:]
    return sample


def recomputed_native_block(self, sample):
    """Original typed block on independent stores; no mutable readout/state replay."""
    if not torch.is_grad_enabled():
        return GraphNetBlock.forward(self, sample)
    template = copy.copy(sample)
    values = tuple(capture_latents(sample).values())
    def action(*inputs):
        graph = _assign_latents(copy.copy(template), inputs)
        graph = GraphNetBlock.forward(self, graph)
        return tuple(capture_latents(graph).values())
    result = checkpoint(action, *values, use_reentrant=False, preserve_rng_state=False)
    return _assign_latents(sample, result)


def recomputed_native_aggr(self, sample, repeat):
    """Unchanged original K recurrence, checkpointed one pure tensor step at a time."""
    if not torch.is_grad_enabled():
        return GraphAggrBlock.propagate_node_features(self, sample, repeat)
    h = sample['cloth'].node_features
    edge_index, e = sample[MESH].edge_index, sample[MESH].features
    src, dst = edge_index
    size = (len(h), len(h))
    def action(nodes, features):
        messages = self.message_mlp(torch.cat((nodes[src], features, nodes[dst]), -1))
        return self.norm(self.gamma*nodes + self.aggregate_nodes(messages, edge_index, size))
    for _ in range(repeat):
        h = checkpoint(action, h, e, use_reentrant=False, preserve_rng_state=False)
    sample['cloth'].aggregated_features = h
    return sample


def enable_native_recomputation(core):
    core.aggr_processor.propagate_node_features = types.MethodType(recomputed_native_aggr, core.aggr_processor)
    for block in core.processor_steps:
        block.forward = types.MethodType(recomputed_native_block, block)


class SourceInteraction(nn.Module):
    """M/C have identical parameters; only the input relation selection differs."""
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.source_encoder = mlp(384)
        self.output = zero_linear(128)
        if arm != 'E':
            self.transport_gate = mlp(384, norm=False)
            self.exchange_gate = mlp(384, norm=False)
            self.value = nn.Linear(128, 128, bias=False)
            self.pair = nn.Linear(256, 128, bias=False)
            self.carrier_norm = nn.LayerNorm(128, elementwise_affine=False)
        self.record = False
        self.last = None

    def source(self, sample):
        relation = MESH if self.arm == 'M' else REVERSE
        src, dst = sample[relation].edge_index
        nodes = sample[relation[0]].node_features
        target = sample['cloth'].node_features
        features = torch.cat((nodes[src], sample[relation].features, target[dst]), -1)
        messages = self.source_encoder(features)
        source = target.new_zeros(target.shape).index_add(0, dst, messages)
        self.last = None
        if self.record:
            self.last = dict(source_relation=list(relation), source_edges=len(src),
                             source_rms=float(source.detach().square().mean().sqrt()))
        active = torch.zeros(len(target), dtype=torch.bool, device=target.device)
        active[dst] = True
        return source, active


class SourceAggrBlock(GraphAggrBlock):
    def __init__(self, *args, arm, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_interaction = SourceInteraction(arm)

    def _iteration(self, h, c, edge_features, edges, active_indices, record):
        adapter = self.source_interaction
        src, dst = edges
        size = (len(h), len(h))
        message_input = torch.cat((h[src], edge_features, h[dst]), -1)
        original_message = self.message_mlp(message_input)
        transport = torch.sigmoid(adapter.transport_gate(message_input))
        exchange = torch.sigmoid(adapter.exchange_gate(message_input))
        carrier_message = transport * adapter.value(c[src])
        extra = adapter.output(exchange * adapter.pair(torch.cat((c[dst], c[src]), -1)))
        h_next = self.norm(self.gamma*h + self.aggregate_nodes(original_message + extra, edges, size))
        carrier_input = self.gamma*c + self.aggregate_nodes(carrier_message, edges, size)
        # No source has reached the other rows. Their mathematical state and
        # parameter derivative are exactly zero; do not build an epsilon-LN
        # chain for those constants. No threshold and no active-state detach.
        normalized = adapter.carrier_norm(carrier_input[active_indices])
        c_next = c.new_zeros(c.shape).index_copy(0, active_indices, normalized)
        rms = extra.detach().square().mean().sqrt() if record else extra.new_zeros(())
        return h_next, c_next, rms

    def propagate_node_features(self, sample, repeat):
        adapter = self.source_interaction
        c, active = adapter.source(sample)
        if adapter.arm == 'E':
            increment = adapter.output(c)
            sample['cloth'].node_features = sample['cloth'].node_features + increment
            if adapter.record:
                adapter.last.update(K=int(repeat), carrier_iterations=0,
                                    injection_rms=float(increment.detach().square().mean().sqrt()))
            return super().propagate_node_features(sample, repeat)

        h = sample['cloth'].node_features
        edges = sample[MESH].edge_index
        src, dst = edges
        edge_features = sample[MESH].features
        last_extra_rms = h.new_zeros(())
        if adapter.record:
            adapter.last['initial_support_nodes'] = int(active.sum())
        for _ in range(repeat):
            next_active = active.clone()
            next_active[dst[active[src]]] = True
            indices = next_active.nonzero().flatten()
            args = (h, c, edge_features, edges, indices, adapter.record)
            if torch.is_grad_enabled():
                h, c, last_extra_rms = checkpoint(self._iteration, *args,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                h, c, last_extra_rms = self._iteration(*args)
            active = next_active
        sample['cloth'].aggregated_features = h
        # GraphAggrBlock.propagate applies the original fuse_mlp(h0,hK).
        if adapter.record:
            adapter.last.update(K=int(repeat), carrier_iterations=int(repeat),
                                carrier_rms=float(c.detach().square().mean().sqrt()),
                                final_support_nodes=int(active.sum()),
                                message_injection_rms=float(last_extra_rms))
        return sample


class SourceCore(NativeCore):
    def __init__(self, mcfg):
        super().__init__(mcfg)
        # Preserve all native names. Canonical original values are copied by
        # the initialization stage, including this block and outer embeddings.
        from functools import partial
        node_fn = partial(self._make_mlp, input_size=self._latent_size*3,
                          output_size=self._latent_size)
        edge_fn = partial(self._make_mlp, input_size=self._latent_size*3,
                          output_size=self._latent_size)
        self.aggr_processor = SourceAggrBlock(
            node_fn, edge_fn, self.aggr_processor.edge_sets,
            self._latent_size, self._num_layers, arm=mcfg.screen_arm)


class StageFeedback(nn.Module):
    """Re-encode a refined same-time proposal into native graph latents."""
    def __init__(self):
        super().__init__()
        self.node_encoder = mlp(15)
        self.mesh_encoder = mlp(8)
        self.face_encoder = mlp(13)
        self.world_encoder = mlp(13)
        self.node_output = zero_linear(128)
        self.mesh_output = zero_linear(128)
        self.world_output = zero_linear(128)
        self.record = False
        self.last = None

    def forward(self, graph, proposal):
        cloth, body = proposal['cloth'], proposal['obstacle']
        q, rest = cloth.pred_pos, cloth.rest_pos
        h = cloth.average_edge_length.reshape(-1)[0]
        mass = cloth.v_mass.reshape(-1)
        faces = cloth.faces_batch.T
        tau = cloth.timestep.reshape(-1)[0]
        materials = torch.stack([cloth[k].reshape(-1)[0] for k in
                                ('lame_mu_input', 'lame_lambda_input', 'bending_coeff_input')])
        deform, strain, area = face_geometry(q, faces, cloth.Dm_inv)
        face_raw = torch.cat((strain[:, 0, 0:1], strain[:, 1, 1:2], strain[:, 0, 1:2],
                              (area/cloth.f_area.flatten())[:, None],
                              materials.expand(len(faces), -1), deform.reshape(-1, 6)), -1)
        face_lat = self.face_encoder(face_raw)
        face_nodes = mean_scatter(face_lat[:, None].expand(-1, 3, -1).reshape(-1, 128),
                                  faces.reshape(-1), len(q))
        normal = vertex_normals(q, faces)
        node_raw = torch.cat(((q-cloth.pos)/h, cloth.velocity/h, normal,
                              (cloth.vertex_type.reshape(-1) == 3).to(q.dtype)[:, None],
                              (mass/mass.mean())[:, None], materials.expand(len(q), -1),
                              tau.expand(len(q), 1)), -1)
        next_body = body.target_pos
        body_normal = vertex_normals(next_body, body.faces_batch.T)
        body_raw = torch.cat(((next_body-body.pos)/h, body.velocity/h, body_normal,
                              q.new_zeros(len(next_body), 1), -q.new_ones(len(next_body), 1),
                              -q.new_ones(len(next_body), 3), tau.expand(len(next_body), 1)), -1)
        src, dst = graph[MESH].edge_index
        rel, rel0 = q[dst]-q[src], rest[dst]-rest[src]
        mesh_raw = torch.cat((rel/h, rel0/h, rel.norm(dim=-1, keepdim=True)/h,
                              rel0.norm(dim=-1, keepdim=True)/h), -1)
        node_delta = self.node_output(self.node_encoder(node_raw) + face_nodes)
        body_delta = self.node_output(self.node_encoder(body_raw))
        mesh_delta = self.mesh_output(self.mesh_encoder(mesh_raw))
        graph['cloth'].node_features = graph['cloth'].node_features + node_delta
        graph['obstacle'].node_features = graph['obstacle'].node_features + body_delta
        graph[MESH].features = graph[MESH].features + mesh_delta
        world_delta = []
        for relation, reverse in ((DIRECT, False), (REVERSE, True)):
            indices = graph[relation].edge_index
            ci, bi = (indices[1], indices[0]) if reverse else indices
            rel = q[ci]-next_body[bi]
            normal = body_normal[bi]
            motion = (q[ci]-cloth.pos[ci])-(next_body[bi]-body.pos[bi])
            slide = motion-(motion*normal).sum(-1, keepdim=True)*normal
            sign = -1. if reverse else 1.
            raw = torch.cat((sign*rel/h, normal, (rel*normal).sum(-1, keepdim=True)/h,
                             sign*motion/h, sign*slide/h), -1)
            delta = self.world_output(self.world_encoder(raw))
            graph[relation].features = graph[relation].features + delta
            world_delta.append(delta)
        self.last = None
        if self.record:
            rms = lambda x: float(x.detach().square().mean().sqrt()) if x.numel() else 0.
            self.last = dict(node_rms=rms(node_delta), body_rms=rms(body_delta),
                             mesh_rms=rms(mesh_delta), world_rms=[rms(x) for x in world_delta])
        return graph


class StageCore(ResponseCore):
    feedback_class = StageFeedback

    def __init__(self, mcfg):
        super().__init__(mcfg)
        self.stage_feedback = self.feedback_class()
        enable_native_recomputation(self)
        enable_native_recomputation(self.response)
        self.position_readout = None
        self.capture_stages = False
        self.stage_records = []
        self.execution = {}

    def forward(self, sample, repeat):
        self.stage_records = []
        self.execution = dict(intermediate_readouts=0, feedbacks=0, final_endpoints=1)
        sample = self._encode(sample)
        sample = self.aggr_processor(sample, repeat)
        for index, block in enumerate(self.processor_steps, 1):
            sample = block(sample)
            if index not in (5, 10):
                continue
            latents = capture_latents(sample)
            proposal = self.position_readout(self._decode(sample.clone()), is_training=False)
            self.execution['intermediate_readouts'] += 1
            q_hat = proposal['cloth'].pred_pos
            increment = self.response(proposal, latents, repeat)
            proposal = add_response(proposal, increment)
            sample = self.stage_feedback(sample, proposal)
            self.execution['feedbacks'] += 1
            if self.stage_feedback.record:
                self.execution['feedback_'+str(index)] = self.stage_feedback.last
            if self.capture_stages:
                self.stage_records.append(dict(stage=index, q_hat=q_hat.detach(),
                                               refined=proposal['cloth'].pred_pos.detach(),
                                               increment=increment.detach()))
            self.response.last = None
        self.anchor_latents = capture_latents(sample)
        return self._decode(sample)


class TerminalCore(StageCore):
    """Same S modules/calls;all native blocks precede two terminal feedback rounds."""
    def forward(self, sample, repeat):
        self.stage_records = []
        self.execution = dict(intermediate_readouts=0, feedbacks=0, final_endpoints=1,
                              organization='terminal_after_15')
        sample = self._encode(sample)
        sample = self.aggr_processor(sample, repeat)
        for block in self.processor_steps:
            sample = block(sample)
        for response_index in (1, 2):
            latents = capture_latents(sample)
            proposal = self.position_readout(self._decode(sample.clone()), is_training=False)
            self.execution['intermediate_readouts'] += 1
            q_hat = proposal['cloth'].pred_pos
            increment = self.response(proposal, latents, repeat)
            proposal = add_response(proposal, increment)
            sample = self.stage_feedback(sample, proposal)
            self.execution['feedbacks'] += 1
            if self.stage_feedback.record:
                self.execution['feedback_terminal_'+str(response_index)] = self.stage_feedback.last
            if self.capture_stages:
                self.stage_records.append(dict(stage='terminal_'+str(response_index),
                                               q_hat=q_hat.detach(),
                                               refined=proposal['cloth'].pred_pos.detach(),
                                               increment=increment.detach()))
            self.response.last = None
        self.anchor_latents = capture_latents(sample)
        return self._decode(sample)


class StageResponseModel(RelationResponseModel):
    def forward(self, sample, repeat=None, is_training=True):
        core = self._learned_model
        # A temporary bound function, not the parent nn.Module as a child.
        core.position_readout = self.get_position
        try:
            return super().forward(sample, repeat=repeat, is_training=is_training)
        finally:
            core.position_readout = None
            core.anchor_latents = None


def EncodeProcessDecode(mcfg):
    if mcfg.screen_arm in ('E', 'M', 'C'):
        return SourceCore(mcfg)
    if mcfg.screen_arm == 'G':
        return ResponseCore(mcfg)
    if mcfg.screen_arm == 'S':
        return StageCore(mcfg)
    if mcfg.screen_arm == 'T':
        return TerminalCore(mcfg)
    raise ValueError('Native uses its original core, not an architecture adapter')
