"""State-conditioned graph-context update utilities."""
from __future__ import annotations

import torch
from torch import nn

from models.core.graph_variants import StageCore, capture_latents, add_response, zero_linear
from models.core.graph_context import GlobalContext, ContextStageFeedback, surface_weights
from models.core.response_routing import ResponseChanges
from models.core.relation_response import mean_scatter


def feature_encoder(width):
    return nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.LayerNorm(128))


class PairedGlobalContext(GlobalContext):
    """One source row per node; preserve R phi and object quadrature."""
    def __init__(self, arm):
        super().__init__()
        if arm not in ('RF', 'RG'):
            raise ValueError('Paired state-innovation arm must be RF or RG')
        self.arm = arm
        self.response_changes = ResponseChanges()
        self.value_encoder = feature_encoder(128 * 2 + 24)
        self.value_output = zero_linear(128)
        self.condition_encoder = feature_encoder(128 * 3 + 24)
        self.condition_output = zero_linear(128)

    def current_conditions(self, refined, context):
        """Shared 24D node conditions; body rows describe prescribed/incident state."""
        cloth, body = refined['cloth'], refined['obstacle']
        q = cloth.pred_pos
        scale = self.response_changes.length_scale_m.to(dtype=q.dtype)
        ci, bi = context['direct_indices']
        normal = context['body_normals'][bi]
        relative = q[ci] - body.target_pos[bi]
        motion = (q[ci] - cloth.pos[ci]) - (body.target_pos[bi] - body.pos[bi])
        gap = (relative * normal).sum(-1, keepdim=True)
        slide = motion - (motion * normal).sum(-1, keepdim=True) * normal
        world = torch.cat((relative / scale, normal, gap / scale,
                           motion / scale, slide / scale), -1)
        cloth_world = mean_scatter(world, ci, len(q))
        body_world = mean_scatter(world, bi, len(body.target_pos))
        cloth_count = q.new_zeros(len(q)).index_add(0, ci, q.new_ones(len(ci)))
        body_count = q.new_zeros(len(body.target_pos)).index_add(0, bi, q.new_ones(len(bi)))
        mass = cloth.v_mass.reshape(-1)
        material = torch.stack([cloth[k].reshape(-1)[0] for k in
                                ('lame_mu_input', 'lame_lambda_input', 'bending_coeff_input')])
        tau = cloth.timestep.reshape(-1)[0]
        edge_scale = cloth.average_edge_length.reshape(-1)[0] / scale
        dynamic_residual = (q - cloth.pos - cloth.velocity) / scale
        cloth_raw = torch.cat((dynamic_residual, cloth_world,
                               (cloth_count > 0).to(q.dtype)[:, None],
                               (mass / mass.mean())[:, None], material.expand(len(q), -1),
                               tau.expand(len(q), 1), edge_scale.expand(len(q), 1),
                               (cloth.vertex_type.reshape(-1) == 3).to(q.dtype)[:, None]), -1)
        body_n = len(body.target_pos)
        # e=0 and mass=-1 are typed sentinels, not a free body inertia model.
        # Material/h metadata belongs to the incident deformable response.
        body_raw = torch.cat((q.new_zeros(body_n, 3), body_world,
                              (body_count > 0).to(q.dtype)[:, None], -q.new_ones(body_n, 1),
                              material.expand(body_n, -1), tau.expand(body_n, 1),
                              edge_scale.expand(body_n, 1), q.new_zeros(body_n, 1)), -1)
        return dict(cloth=cloth_raw, obstacle=body_raw)

    def exchange(self, graph, h, response, weights, tokens):
        state, changes, conditions = (response[k] for k in ('state', 'changes', 'conditions'))
        inputs, measures, queries, sizes, interaction = [], [], [], [], {}
        for index, kind in enumerate(('cloth', 'obstacle')):
            base = self.context_input(torch.cat((h[kind], state[kind]), -1))
            value = self.value_output(self.value_encoder(
                torch.cat((state[kind], changes[kind], conditions[kind]), -1)))
            condition = self.condition_output(self.condition_encoder(
                torch.cat((h[kind], state[kind], changes[kind], conditions[kind]), -1)))
            if self.arm == 'RG':
                gate = torch.sigmoid(condition)
                innovation = gate * value
            else:
                gate = None
                innovation = 0.5 * value + condition
            inputs.append(base + innovation + self.type_embedding[index])
            measures.append((weights[kind] * 0.5).log())
            queries.append(graph[kind].node_features + self.type_embedding[index])
            sizes.append(len(queries[-1]))
            if self.record:
                rms = lambda x: float(x.detach().square().mean().sqrt()) if x.numel() else 0.
                interaction[kind] = dict(base_rms=rms(base), value_rms=rms(value),
                    condition_rms=rms(condition), innovation_rms=rms(innovation),
                    physical_condition_rms=rms(conditions[kind]),
                    gate_mean=float(gate.detach().mean()) if gate is not None else None,
                    gate_min=float(gate.detach().min()) if gate is not None else None,
                    gate_max=float(gate.detach().max()) if gate is not None else None)
        before = tokens
        compressed = self.compress(self.compress_norm(tokens), torch.cat(inputs), torch.cat(measures))
        tokens = tokens + compressed
        normalized = self.token_norm(tokens)
        tokens = tokens + self.token_attention(normalized, normalized)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        values = self.write(torch.cat(queries), self.write_norm(tokens))
        cloth_delta, body_delta = self.node_output(values).split(sizes)
        graph['cloth'].node_features = graph['cloth'].node_features + cloth_delta
        graph['obstacle'].node_features = graph['obstacle'].node_features + body_delta
        record = None
        if self.record:
            rms = lambda x: float(x.detach().square().mean().sqrt()) if x.numel() else 0.
            record = dict(token_rms=rms(tokens), token_change_rms=rms(tokens - before),
                compression_rms=rms(compressed), node_writeback_rms=rms(cloth_delta),
                body_writeback_rms=rms(body_delta), source_nodes=sum(sizes),
                source_rows_per_node=1, quadrature_mass=[float(x.sum()) for x in weights.values()],
                graph_context_rms={k: rms(h[k]) for k in h},
                response_context_rms={k: rms(state[k]) for k in state},
                response_changes=self.response_changes.last, interaction=interaction,
                fusion='conditioned_gate' if self.arm == 'RG' else 'additive_conditioned_ff')
        return graph, tokens, record


class PairedCore(StageCore):
    feedback_class = ContextStageFeedback

    def __init__(self, mcfg):
        super().__init__(mcfg)
        self.stage_feedback.context = None
        self.arm = mcfg.screen_arm
        self.global_context = PairedGlobalContext(self.arm)

    def forward(self, sample, repeat):
        self.stage_records = []
        self.execution = dict(intermediate_readouts=0, feedbacks=0, final_endpoints=1,
            compression_calls=0, token_update_calls=0, global_writebacks=0,
            organization=self.arm, response_representation='full_state_plus_fresh_changes',
            global_sources='paired_same_node_before_compression')
        sample = self._encode(sample)
        sample = self.aggr_processor(sample, repeat)
        weights = dict(cloth=surface_weights(sample['cloth'].rest_pos, sample['cloth'].faces_batch.T,
                                             sample['cloth'].f_area),
                       obstacle=surface_weights(sample['obstacle'].target_pos,
                                                sample['obstacle'].faces_batch.T))
        tokens = self.global_context.seeds
        try:
            for index, block in enumerate(self.processor_steps, 1):
                sample = block(sample)
                if index not in (5, 10):
                    continue
                latents = capture_latents(sample)
                current_h = {k: latents[k] for k in ('cloth', 'obstacle')}
                proposal = self.position_readout(self._decode(sample.clone()), is_training=False)
                self.execution['intermediate_readouts'] += 1
                q_hat = proposal['cloth'].pred_pos
                increment, context = self.response(proposal, latents, repeat, return_context=True)
                proposal = add_response(proposal, increment)
                sample = self.stage_feedback(sample, proposal)
                self.execution['feedbacks'] += 1
                self.global_context.response_changes.record = self.global_context.record
                changes = self.global_context.response_changes(proposal, context)
                response = dict(state=self.stage_feedback.context, changes=changes,
                                conditions=self.global_context.current_conditions(proposal, context))
                sample, tokens, record = self.global_context.exchange(
                    sample, current_h, response, weights, tokens)
                for key in ('compression_calls', 'token_update_calls', 'global_writebacks'):
                    self.execution[key] += 1
                if self.stage_feedback.record:
                    self.execution['feedback_' + str(index)] = self.stage_feedback.last
                if record is not None:
                    self.execution['global_' + str(index)] = dict(record,
                        graph_source_stage=index, response_source_stage=index)
                if self.capture_stages:
                    self.stage_records.append(dict(stage=index, q_hat=q_hat.detach(),
                        refined=proposal['cloth'].pred_pos.detach(), increment=increment.detach(),
                        internal_increment=context['internal_increment'].detach(),
                        contact_increment=context['contact_increment'].detach()))
                self.stage_feedback.context = None
                self.response.last = None
            self.anchor_latents = capture_latents(sample)
            return self._decode(sample)
        finally:
            self.stage_feedback.context = None


def EncodeProcessDecode(mcfg):
    return PairedCore(mcfg)
