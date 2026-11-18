"""Feedback of updated garment geometry into persistent graph representations."""
from __future__ import annotations

import torch
from models.core.relation_response import RelationResponse, hinge_basis
from models.core.graph_variants import capture_latents, add_response
from models.core.graph_context import GlobalCore, surface_weights
from models.pb4u import Model as NativeModel

COEVOLUTION_ARMS = ('W', 'J', 'WJ', 'WJK')


def unique_hinge_gradient(derivative, tri_ids, shared):
    """Merge shared-vertex derivatives before taking a metric or norm."""
    opposite_slot = ((tri_ids != shared[:, None, 0, None]) &
                     (tri_ids != shared[:, None, 1, None])).long().argmax(-1)
    opposite = tri_ids.gather(2, opposite_slot[:, :, None]).squeeze(-1)
    unique = torch.cat((shared, opposite), -1)
    membership = tri_ids.reshape(-1, 6, 1) == unique[:, None, :]
    gradient = (derivative.reshape(-1, 6, 1, 3)*membership[:, :, :, None]).sum(1)
    return unique, gradient


def hinge_mass_gram(derivative, tri_ids, shared, mass, free):
    unique, gradient = unique_hinge_gradient(derivative, tri_ids, shared)
    gram = (gradient.square().sum(-1)*free[unique]/mass[unique]).sum(-1)
    return gram, unique, gradient


class GeometryRelationResponse(RelationResponse):
    def __init__(self, edge_sets, mode):
        super().__init__(edge_sets, mode)
        self.geometry_conditioned = False
        self.source_before_K = False
        self.rest_gram = None
        self.rest_precomputations = 0
        self.metric_record = None
        self.metric_history = []

    def reset_frame(self):
        self.rest_gram = None
        self.rest_precomputations = 0
        self.metric_record = None
        self.metric_history = []

    def process_graph(self, graph, repeat):
        if not self.source_before_K:
            return super().process_graph(graph, repeat)
        graph = self.processor_steps[0](graph)
        graph = self.aggr_processor(graph, repeat)
        return self.processor_steps[1](graph)

    def condition_hinge_coefficient(self, sample, derivative, tri_ids, shared, coefficient):
        if not self.geometry_conditioned:
            return coefficient
        cloth = sample['cloth']
        mass = cloth.v_mass.reshape(-1)
        free = cloth.vertex_type.reshape(-1) != 3
        current, unique, gradient = hinge_mass_gram(derivative, tri_ids, shared, mass, free)
        if self.rest_gram is None:
            with torch.no_grad():
                _, rest_derivative, rest_ids, _ = hinge_basis(
                    cloth.rest_pos, cloth.faces_batch.T,
                    cloth.f_connectivity, cloth.f_connectivity_edges)
                self.rest_gram, _, _ = hinge_mass_gram(rest_derivative, rest_ids, shared, mass, free)
            self.rest_precomputations += 1
        denominator = current+self.rest_gram
        # Zero-Gram, all-HANDLE relations assemble no free-node displacement.
        safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        alpha = 2*self.rest_gram/safe_denominator
        self.metric_record = None
        if self.record:
            self.metric_record = dict(hinge_gram=current, hinge_rest_gram=self.rest_gram,
                hinge_gain=alpha, hinge_unique_ids=unique, hinge_unique_gradient=gradient,
                hinge_unconditioned_coefficients=coefficient,
                rest_precomputations=self.rest_precomputations)
        return coefficient*alpha

    def metric_statistics(self):
        if self.metric_record is None:
            return None
        r = self.metric_record
        g, g0, gain = (r[k].detach() for k in ('hinge_gram','hinge_rest_gram','hinge_gain'))
        active = g0 > 0
        ratio = gain[active]*torch.sqrt(g[active]/g0[active])
        return dict(gain_min=float(gain.min()), gain_mean=float(gain.mean()), gain_max=float(gain.max()),
                    current_gram_max=float(g.max()), rest_gram_max=float(g0.max()),
                    mass_norm_to_rest_bound_max=float(ratio.max()) if len(ratio) else None,
                    rest_precomputations=self.rest_precomputations)

    def hinge_record(self):
        metric = self.metric_statistics()
        if metric is not None:
            self.metric_history.append(metric)
        return self.metric_record or {}


class GeometryFeedbackCore(GlobalCore):
    response_class = GeometryRelationResponse

    def __init__(self, mcfg):
        super().__init__(mcfg)
        self.coevolution_arm = mcfg.screen_arm
        if self.coevolution_arm not in COEVOLUTION_ARMS:
            raise ValueError('Stage co-evolution requires an explicit W/J/WJ/WJK arm')
        # External arm labels must never select the historical U/V cached contexts.
        self.arm = 'R'
        self.carry_enabled = self.coevolution_arm in ('W', 'WJ', 'WJK')
        self.response.geometry_conditioned = self.coevolution_arm in ('J', 'WJ', 'WJK')
        self.response.source_before_K = self.coevolution_arm == 'WJK'
        self.working_carry = None
        self.anchor_acceleration = None
        self.final_record = None

    def forward(self, sample, repeat):
        if not self.carry_enabled:
            result = super().forward(sample, repeat)
            self.execution.update(organization=self.coevolution_arm, working_state_carry=False,
                post_feedback_readouts=0, geometry_conditioned_hinge=True,
                response_order='K,typed1,typed2')
            return result
        self.stage_records = []
        self.execution = dict(intermediate_readouts=0, stage_proposal_readouts=0,
            post_feedback_readouts=0, feedbacks=0, final_endpoints=1,
            compression_calls=0, token_update_calls=0, global_writebacks=0,
            organization=self.coevolution_arm, working_state_carry=True,
            geometry_conditioned_hinge=self.response.geometry_conditioned,
            response_order='typed1,K,typed2' if self.response.source_before_K else 'K,typed1,typed2')
        self.working_carry = sample['cloth'].pos.new_zeros(sample['cloth'].pos.shape)
        free = sample['cloth'].vertex_type.reshape(-1) != 3
        sample = self._encode(sample)
        sample = self.aggr_processor(sample, repeat)
        weights = dict(cloth=surface_weights(sample['cloth'].rest_pos, sample['cloth'].faces_batch.T,
                                             sample['cloth'].f_area),
                       obstacle=surface_weights(sample['obstacle'].target_pos,
                                                sample['obstacle'].faces_batch.T))
        tokens = self.global_context.seeds
        previous_refined = None
        try:
            for index, block in enumerate(self.processor_steps, 1):
                sample = block(sample)
                if index not in (5, 10):
                    continue
                latents = capture_latents(sample)
                current_h = {k: latents[k] for k in ('cloth', 'obstacle')}
                proposal = self.position_readout(self._decode(sample.clone()), is_training=False)
                self.execution['intermediate_readouts'] += 1
                self.execution['stage_proposal_readouts'] += 1
                raw_q, raw_a = proposal['cloth'].pred_pos, proposal['cloth'].pred_acceleration
                innovation = raw_a if self.anchor_acceleration is None else raw_a-self.anchor_acceleration
                proposal = add_response(proposal, self.working_carry)
                q_hat, carried_a = proposal['cloth'].pred_pos, proposal['cloth'].pred_acceleration
                increment = self.response(proposal, latents, repeat)
                proposal = add_response(proposal, increment)
                refined_a = proposal['cloth'].pred_acceleration
                metric = self.response.metric_statistics()
                if metric is not None:
                    self.execution['hinge_metric_'+str(index)] = metric
                sample = self.stage_feedback(sample, proposal)
                self.execution['feedbacks'] += 1
                sample, tokens, record = self.global_context.exchange(
                    sample, current_h, self.stage_feedback.context, weights, tokens)
                for key in ('compression_calls', 'token_update_calls', 'global_writebacks'):
                    self.execution[key] += 1
                # The same decoder/inverse, with unchanged statistics and full gradients.
                anchored = self.position_readout(self._decode(sample.clone()), is_training=False)
                self.anchor_acceleration = anchored['cloth'].pred_acceleration
                self.working_carry = (refined_a-self.anchor_acceleration)*free[:, None]
                self.execution['intermediate_readouts'] += 1
                self.execution['post_feedback_readouts'] += 1
                if self.stage_feedback.record:
                    self.execution['feedback_'+str(index)] = self.stage_feedback.last
                    self.execution['carry_'+str(index)] = dict(
                        carry_rms_m=float(self.working_carry.detach().square().mean().sqrt()),
                        closure_max_m=float(((self.anchor_acceleration+self.working_carry-refined_a)
                                             *free[:, None]).detach().abs().max()))
                if record is not None:
                    self.execution['global_'+str(index)] = dict(record,
                        graph_source_stage=index, response_source_stage=index)
                if self.capture_stages:
                    row = dict(stage=index, raw_proposal=raw_q.detach(), q_hat=q_hat.detach(),
                        carried_proposal=q_hat.detach(), refined=proposal['cloth'].pred_pos.detach(),
                        increment=increment.detach(), raw_acceleration=raw_a.detach(),
                        carried_acceleration=carried_a.detach(), refined_acceleration=refined_a.detach(),
                        anchor_acceleration=self.anchor_acceleration.detach(),
                        working_carry=self.working_carry.detach(),
                        decoder_innovation=(innovation*free[:, None]).detach())
                    if previous_refined is not None:
                        row['representation_error'] = ((carried_a-previous_refined-innovation)
                                                       *free[:, None]).detach()
                    self.stage_records.append(row)
                previous_refined = refined_a
                self.stage_feedback.context = None
                self.response.last = None
            self.anchor_latents = capture_latents(sample)
            return self._decode(sample)
        finally:
            self.stage_feedback.context = None


class GeometryFeedbackModel(NativeModel):
    def forward(self, sample, repeat=None, is_training=True):
        core = self._learned_model
        native_statistics = is_training and not getattr(self, 'native_frozen', False)
        core.response.reset_frame()
        core.working_carry = None
        core.anchor_acceleration = None
        core.final_record = None
        core.position_readout = self.get_position
        try:
            sample = self.prepare_inputs(sample, is_training=native_statistics)
            if repeat is None:
                repeat = max(1, int(self.propagation_distance/sample['cloth'].average_edge_length.item()))
            sample = core(sample, repeat)
            # Final inverse precedes the sole target-statistics accumulation, as native.
            sample = self.get_position(sample, is_training=native_statistics)
            raw_q, raw_a = sample['cloth'].pred_pos, sample['cloth'].pred_acceleration
            if core.carry_enabled:
                sample = add_response(sample, core.working_carry)
            if core.capture_stages:
                core.final_record = dict(raw_proposal=raw_q.detach(),
                    carried_proposal=sample['cloth'].pred_pos.detach(), raw_acceleration=raw_a.detach())
                if core.carry_enabled:
                    free = sample['cloth'].vertex_type.reshape(-1) != 3
                    core.final_record.update(working_carry=core.working_carry.detach(),
                        decoder_innovation=((raw_a-core.anchor_acceleration)*free[:, None]).detach())
            increment = core.response(sample, core.anchor_latents, repeat)
            sample = add_response(sample, increment)
            metric = core.response.metric_statistics()
            if metric is not None:
                core.execution['hinge_metric_15'] = metric
            for stage, row in zip((5, 10, 15), core.response.metric_history):
                core.execution['hinge_metric_'+str(stage)] = row
            core.execution['rest_precomputations'] = core.response.rest_precomputations
            if core.capture_stages:
                core.final_record['refined'] = sample['cloth'].pred_pos.detach()
            return sample
        finally:
            core.position_readout = None
            core.anchor_latents = None
            core.working_carry = None
            core.anchor_acceleration = None
            core.response.rest_gram = None
            core.response.metric_record = None
            core.response.metric_history = []
            core.stage_feedback.context = None


def EncodeProcessDecode(mcfg):
    return GeometryFeedbackCore(mcfg)
