"""Graph-context aggregation utilities used by the model construction helpers."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from models.core.graph_variants import StageCore, StageFeedback, StageResponseModel, capture_latents, add_response, MESH, DIRECT, REVERSE, zero_linear
from models.core.relation_response import face_geometry, vertex_normals, mean_scatter


def surface_weights(vertices, faces, face_area=None):
    if face_area is None:
        tri = vertices[faces]
        face_area = torch.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0], dim=-1).norm(dim=-1)*0.5
    weights = vertices.new_zeros(len(vertices)).index_add(
        0, faces.reshape(-1), face_area.flatten()[:, None].expand(-1, 3).reshape(-1)/3.)
    weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    return weights/weights.sum()


class Attention(nn.Module):
    """Multi-head attention with explicit quadrature as an additive log measure."""
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads, self.head_width = heads, width//heads
        self.query, self.key, self.value = (nn.Linear(width, width) for _ in range(3))
        self.output = nn.Linear(width, width)

    def forward(self, query, source, log_measure=None):
        def split(x):
            return x.reshape(1, len(x), self.heads, self.head_width).transpose(1, 2)
        q, k, v = split(self.query(query)), split(self.key(source)), split(self.value(source))
        mask = None if log_measure is None else log_measure[None, None, None, :]
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        return self.output(output.transpose(1, 2).reshape(len(query), -1))


class GlobalContext(nn.Module):
    """Shared compression, token evolution and latent-only writeback at both stages."""
    def __init__(self):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(64, 128)*0.02)
        self.type_embedding = nn.Parameter(torch.randn(2, 128)*0.02)
        self.context_input = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.LayerNorm(128))
        self.compress_norm = nn.LayerNorm(128)
        self.compress = Attention()
        self.token_norm = nn.LayerNorm(128)
        self.token_attention = Attention()
        self.ffn_norm = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 128))
        self.write_norm = nn.LayerNorm(128)
        self.write = Attention()
        self.node_output = zero_linear(128)
        self.record = False

    def exchange(self, graph, h, response, weights, tokens):
        inputs, measures, queries, sizes = [], [], [], []
        for index, kind in enumerate(('cloth', 'obstacle')):
            inputs.append(self.context_input(torch.cat((h[kind], response[kind]), -1))+self.type_embedding[index])
            measures.append((weights[kind]*0.5).log())
            queries.append(graph[kind].node_features+self.type_embedding[index])
            sizes.append(len(queries[-1]))
        before = tokens
        compressed = self.compress(self.compress_norm(tokens), torch.cat(inputs), torch.cat(measures))
        tokens = tokens+compressed
        normalized = self.token_norm(tokens)
        evolved = self.token_attention(normalized, normalized)
        tokens = tokens+evolved
        tokens = tokens+self.ffn(self.ffn_norm(tokens))
        values = self.write(torch.cat(queries), self.write_norm(tokens))
        delta = self.node_output(values)
        cloth_delta, body_delta = delta.split(sizes)
        graph['cloth'].node_features = graph['cloth'].node_features+cloth_delta
        graph['obstacle'].node_features = graph['obstacle'].node_features+body_delta
        record = None
        if self.record:
            rms = lambda x: float(x.detach().square().mean().sqrt())
            record = dict(token_rms=rms(tokens), token_change_rms=rms(tokens-before),
                          compression_rms=rms(compressed), node_writeback_rms=rms(cloth_delta),
                          body_writeback_rms=rms(body_delta), source_nodes=sum(sizes),
                          quadrature_mass=[float(x.sum()) for x in weights.values()],
                          graph_context_rms={k:rms(h[k]) for k in h},
                          response_context_rms={k:rms(response[k]) for k in response})
        return graph, tokens, record


class ContextStageFeedback(StageFeedback):
    """Re-encode a refined same-time proposal into native graph latents."""
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
        node_context = self.node_encoder(node_raw) + face_nodes
        body_context = self.node_encoder(body_raw)
        self.context = dict(cloth=node_context, obstacle=body_context)
        node_delta = self.node_output(node_context)
        body_delta = self.node_output(body_context)
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


class GlobalCore(StageCore):
    feedback_class = ContextStageFeedback

    def __init__(self, mcfg):
        super().__init__(mcfg)
        self.stage_feedback.context = None
        self.global_context = GlobalContext()
        self.arm = mcfg.screen_arm

    def forward(self, sample, repeat):
        self.stage_records = []
        self.execution = dict(intermediate_readouts=0, feedbacks=0, final_endpoints=1,
                              compression_calls=0, token_update_calls=0, global_writebacks=0,
                              organization=self.arm)
        sample = self._encode(sample)
        sample = self.aggr_processor(sample, repeat)
        weights = dict(cloth=surface_weights(sample['cloth'].rest_pos, sample['cloth'].faces_batch.T,
                                             sample['cloth'].f_area),
                       obstacle=surface_weights(sample['obstacle'].target_pos,
                                                sample['obstacle'].faces_batch.T))
        tokens = self.global_context.seeds
        cached_h, cached_response = None, None
        try:
            for index, block in enumerate(self.processor_steps, 1):
                sample = block(sample)
                if index not in (5, 10):
                    continue
                latents = capture_latents(sample)
                # These tensor references precede feedback and are never overwritten in-place.
                current_h = {k:latents[k] for k in ('cloth', 'obstacle')}
                proposal = self.position_readout(self._decode(sample.clone()), is_training=False)
                self.execution['intermediate_readouts'] += 1
                q_hat = proposal['cloth'].pred_pos
                increment = self.response(proposal, latents, repeat)
                proposal = add_response(proposal, increment)
                sample = self.stage_feedback(sample, proposal)
                current_response = self.stage_feedback.context
                self.execution['feedbacks'] += 1
                if index == 5:
                    cached_h, cached_response = current_h, current_response
                selected_h = cached_h if self.arm == 'U' else current_h
                selected_response = current_response if self.arm == 'R' else cached_response
                sample, tokens, record = self.global_context.exchange(
                    sample, selected_h, selected_response, weights, tokens)
                for key in ('compression_calls', 'token_update_calls', 'global_writebacks'):
                    self.execution[key] += 1
                if self.stage_feedback.record:
                    self.execution['feedback_'+str(index)] = self.stage_feedback.last
                if record is not None:
                    self.execution['global_'+str(index)] = dict(record,
                        graph_source_stage=5 if self.arm == 'U' else index,
                        response_source_stage=index if self.arm == 'R' else 5)
                if self.capture_stages:
                    self.stage_records.append(dict(stage=index, q_hat=q_hat.detach(),
                        refined=proposal['cloth'].pred_pos.detach(), increment=increment.detach()))
                self.stage_feedback.context = None
                self.response.last = None
            self.anchor_latents = capture_latents(sample)
            return self._decode(sample)
        finally:
            self.stage_feedback.context = None


def EncodeProcessDecode(mcfg):
    return GlobalCore(mcfg)
