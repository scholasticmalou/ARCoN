"""Graph-context routing utilities used by checkpoint-compatible model construction."""
from __future__ import annotations

import torch
from torch import nn
from models.core.graph_variants import StageCore, StageFeedback, capture_latents, add_response
from models.core.graph_context import GlobalContext, surface_weights
from models.core.relation_response import face_geometry, vertex_normals, mean_scatter, mlp


class ResponseChanges(nn.Module):
    """Node/face changes and associated body-side context at a fixed SI length scale."""
    def __init__(self):
        super().__init__()
        self.register_buffer('length_scale_m', torch.tensor(0.1))
        self.node_encoder = mlp(24)
        self.face_encoder = mlp(7)
        self.record = False
        self.last = None

    def forward(self, refined, context):
        cloth, body = refined['cloth'], refined['obstacle']
        q, before = cloth.pred_pos, context['q_hat']
        faces = cloth.faces_batch.T
        scale = self.length_scale_m.to(dtype=q.dtype)
        internal = context['internal_increment']/scale
        contact = context['contact_increment']/scale
        _, strain, area = face_geometry(q, faces, cloth.Dm_inv)
        ds = strain-context['strain']
        da = (area-context['area'])/cloth.f_area.flatten()
        material = torch.stack([cloth.lame_mu_input.reshape(-1)[0],
                                cloth.lame_lambda_input.reshape(-1)[0],
                                cloth.bending_coeff_input.reshape(-1)[0]])
        face_raw = torch.cat((ds[:, 0, 0:1], ds[:, 1, 1:2], ds[:, 0, 1:2],
                              da[:, None], material.expand(len(faces), -1)), -1)
        face_encoded = self.face_encoder(face_raw)
        face_nodes = mean_scatter(face_encoded[:, None].expand(-1, 3, -1).reshape(-1, 128),
                                  faces.reshape(-1), len(q))
        delta_normal = vertex_normals(q, faces)-context['normals']
        ci, bi = context['direct_indices']
        normal = context['body_normals'][bi]
        delta_relative = (q[ci]-before[ci])/scale
        delta_gap = (delta_relative*normal).sum(-1, keepdim=True)
        delta_slide = delta_relative-delta_gap*normal
        world_change = torch.cat((delta_relative, delta_gap, delta_slide), -1)
        cloth_world = mean_scatter(world_change, ci, len(q))
        body_world = mean_scatter(world_change, bi, len(body.target_pos))
        cloth_count = q.new_zeros(len(q)).index_add(0, ci, q.new_ones(len(ci)))
        body_count = q.new_zeros(len(body.target_pos)).index_add(0, bi, q.new_ones(len(bi)))
        mass = cloth.v_mass.reshape(-1)
        tau = cloth.timestep.reshape(-1)[0]
        edge_scale = cloth.average_edge_length.reshape(-1)[0]/scale
        free_flag = (cloth.vertex_type.reshape(-1) == 3).to(q.dtype)
        cloth_raw = torch.cat((internal, contact, delta_normal, cloth_world,
                               (cloth_count > 0).to(q.dtype)[:, None],
                               (mass/mass.mean())[:, None], material.expand(len(q), -1),
                               tau.expand(len(q), 1), edge_scale.expand(len(q), 1),
                               free_flag[:, None]), -1)
        # Body is externally driven. This source encodes incident cloth response,
        # not a predicted body displacement or a new free body state.
        body_n = len(body.target_pos)
        body_raw = torch.cat((mean_scatter(internal[ci], bi, body_n),
                              mean_scatter(contact[ci], bi, body_n),
                              q.new_zeros(body_n, 3), body_world,
                              (body_count > 0).to(q.dtype)[:, None],
                              -q.new_ones(body_n, 1), material.expand(body_n, -1),
                              tau.expand(body_n, 1), edge_scale.expand(body_n, 1),
                              q.new_zeros(body_n, 1)), -1)
        result = dict(cloth=self.node_encoder(cloth_raw)+face_nodes,
                      obstacle=self.node_encoder(body_raw))
        self.last = None
        if self.record:
            rms = lambda t: float(t.detach().square().mean().sqrt()) if t.numel() else 0.
            self.last = dict(length_scale_m=float(scale), internal_increment_rms_m=rms(internal*scale),
                             contact_increment_rms_m=rms(contact*scale), strain_change_rms=rms(ds),
                             relative_area_change_rms=rms(da), contact_change_rms=rms(world_change),
                             cloth_contact_support=int((cloth_count > 0).sum()),
                             body_contact_support=int((body_count > 0).sum()),
                             assembled_component_sum_max_m=float((context['internal_increment']+
                                 context['contact_increment']-(q-before)).detach().abs().max()))
        return result


class RoutedGlobalContext(GlobalContext):
    """Graph and response sources have separate feature encoders and K/V rows."""
    def __init__(self):
        super().__init__()
        del self.context_input
        self.graph_input = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.LayerNorm(128))
        self.response_input = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.LayerNorm(128))
        self.source_embedding = nn.Parameter(torch.randn(2, 128)*0.02)
        self.response_changes = ResponseChanges()

    def exchange(self, graph, h, response, weights, tokens):
        inputs, measures, queries, sizes = [], [], [], []
        for index, kind in enumerate(('cloth', 'obstacle')):
            # Each object retains 1/2 total quadrature mass. Its two sources
            # share that mass equally, so adding a source does not double it.
            inputs.extend((self.graph_input(h[kind])+self.type_embedding[index]+self.source_embedding[0],
                           self.response_input(response[kind])+self.type_embedding[index]+self.source_embedding[1]))
            measures.extend(((weights[kind]*0.25).log(), (weights[kind]*0.25).log()))
            queries.append(graph[kind].node_features+self.type_embedding[index])
            sizes.append(len(queries[-1]))
        before = tokens
        compressed = self.compress(self.compress_norm(tokens), torch.cat(inputs, 0), torch.cat(measures))
        tokens = tokens+compressed
        normalized = self.token_norm(tokens)
        tokens = tokens+self.token_attention(normalized, normalized)
        tokens = tokens+self.ffn(self.ffn_norm(tokens))
        values = self.write(torch.cat(queries), self.write_norm(tokens))
        cloth_delta, body_delta = self.node_output(values).split(sizes)
        graph['cloth'].node_features = graph['cloth'].node_features+cloth_delta
        graph['obstacle'].node_features = graph['obstacle'].node_features+body_delta
        record = None
        if self.record:
            rms = lambda t: float(t.detach().square().mean().sqrt())
            record = dict(token_rms=rms(tokens), token_change_rms=rms(tokens-before),
                          compression_rms=rms(compressed), node_writeback_rms=rms(cloth_delta),
                          body_writeback_rms=rms(body_delta), source_nodes=2*sum(sizes),
                          quadrature_mass=[float(t.sum()) for t in weights.values()],
                          source_measure_masses=[float((t*0.25).sum()) for t in weights.values() for _ in range(2)],
                          graph_context_rms={k:rms(h[k]) for k in h},
                          response_context_rms={k:rms(response[k]) for k in response},
                          response_changes=self.response_changes.last)
        return graph, tokens, record


class RoutedCore(StageCore):
    feedback_class = StageFeedback

    def __init__(self, mcfg):
        super().__init__(mcfg)
        self.global_context = RoutedGlobalContext()
        self.arm = mcfg.screen_arm

    def forward(self, sample, repeat):
        self.stage_records = []
        self.execution = dict(intermediate_readouts=0, feedbacks=0, final_endpoints=1,
                              compression_calls=0, token_update_calls=0, global_writebacks=0,
                              organization='A', response_representation='assembled_changes',
                              global_sources='separate_graph_and_response_rows')
        sample = self._encode(sample)
        sample = self.aggr_processor(sample, repeat)
        weights = dict(cloth=surface_weights(sample['cloth'].rest_pos, sample['cloth'].faces_batch.T,
                                             sample['cloth'].f_area),
                       obstacle=surface_weights(sample['obstacle'].target_pos,
                                                sample['obstacle'].faces_batch.T))
        tokens = self.global_context.seeds
        for index, block in enumerate(self.processor_steps, 1):
            sample = block(sample)
            if index not in (5, 10):
                continue
            latents = capture_latents(sample)
            current_h = {k:latents[k] for k in ('cloth', 'obstacle')}
            proposal = self.position_readout(self._decode(sample.clone()), is_training=False)
            self.execution['intermediate_readouts'] += 1
            q_hat = proposal['cloth'].pred_pos
            increment, context = self.response(proposal, latents, repeat, return_context=True)
            proposal = add_response(proposal, increment)
            self.global_context.response_changes.record = self.global_context.record
            response = self.global_context.response_changes(proposal, context)
            sample = self.stage_feedback(sample, proposal)
            self.execution['feedbacks'] += 1
            sample, tokens, record = self.global_context.exchange(sample, current_h, response, weights, tokens)
            for key in ('compression_calls', 'token_update_calls', 'global_writebacks'):
                self.execution[key] += 1
            if self.stage_feedback.record:
                self.execution['feedback_'+str(index)] = self.stage_feedback.last
            if record is not None:
                self.execution['global_'+str(index)] = dict(record, graph_source_stage=index,
                                                           response_source_stage=index)
            if self.capture_stages:
                self.stage_records.append(dict(stage=index, q_hat=q_hat.detach(),
                    refined=proposal['cloth'].pred_pos.detach(), increment=increment.detach()))
            self.response.last = None
            self.global_context.response_changes.last = None
        self.anchor_latents = capture_latents(sample)
        return self._decode(sample)


def EncodeProcessDecode(mcfg):
    return RoutedCore(mcfg)
