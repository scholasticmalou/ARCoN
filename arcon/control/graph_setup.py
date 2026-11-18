"""Construction utilities for the graph backbone and its checkpoint-compatible variants.

The public ARCoN entry selects one fixed configuration. Upstream data adapters,
physical objectives and runner APIs retain their original namespaces."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import types

import numpy as np
import torch
from arcon.control.reproducibility import install_official_py312_compat, set_seed

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('N', 'E', 'M', 'C', 'G', 'S')
GLOBAL_ARMS = ('U', 'V', 'R')
ROUTING_ARMS = ('A',)
FUSION_ARMS = ('RF', 'RG')
COEVOLUTION_ARMS = ('W', 'J', 'WJ', 'WJK')
SELF_ARMS = ('WJ_L', 'WJ_LS')
SELF_CRITERION = 'pb4u.mataug.self_intersection_energy'
SELF_PREFIXES = tuple('model._learned_model.response.'+name+'.' for name in
                      ('self_encoder', 'self_message', 'self_output'))
ALL_ARMS = ARMS + ('T',) + GLOBAL_ARMS + ROUTING_ARMS + FUSION_ARMS + COEVOLUTION_ARMS + SELF_ARMS
GLOBAL_PREFIX = 'model._learned_model.global_context.'
STEPS = 5000
SOURCE_PREFIX = 'model._learned_model.aggr_processor.source_interaction.'
RESPONSE_PREFIX = 'model._learned_model.response.'
FEEDBACK_PREFIX = 'model._learned_model.stage_feedback.'


def write_json(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(file)


def source_revision():
    return "anonymous-review"


def iteration_budget(args):
    return int(getattr(args, 'iterations', STEPS))


def expected_counts(iterations):
    forwards = sum(min(1+step//500, 5) for step in range(1, iterations+1))
    return iterations, forwards, forwards-min(iterations, 99)


def protocol(arm, iterations=STEPS):
    _, forwards, updates = expected_counts(iterations)
    result = dict(family='pb4u_six_arm_20261010_v1', arm=arm, seed=57,
                original_blocks=15, latent_width=128, from_scratch=True,
                paired_original_initialization=True, trainable_backbone=True,
                data_iterations=iterations, batch=1, roll_init=1, roll_max=5,
                increase_roll_every=500, warmup_steps=100,
                expected_forwards=forwards, expected_optimizer_updates=updates,
                training_K='official Python randint(0,30)',
                evaluation_K='official max(1,floor(D/mean_rest_edge))',
                native_losses_and_collision_ramp=True, per_frame_detach_and_optimizer=True,
                stage_feedback=[5, 10] if arm == 'S' else [],
                response_calls=3 if arm == 'S' else 1 if arm == 'G' else 0)
    if arm == 'T':
        result.update(family='pb4u_terminal_vs_staged_20261010_v1',
                      stage_feedback=['terminal_1', 'terminal_2'], response_calls=3,
                      matched_stage_initialization='S', physical_writebacks=1)
    if arm in GLOBAL_ARMS:
        result.update(family='pb4u_global_context_20261010_v1',
                      stage_feedback=[5, 10], response_calls=3, physical_writebacks=1,
                      matched_stage_initialization='S', global_tokens=64, global_width=128,
                      global_heads=4, global_exchanges=2, shared_global_module=True,
                      quadrature='normalized cloth rest/body target surface area;equal type mass',
                      graph_context='cached5' if arm == 'U' else 'pre_feedback_current',
                      response_context='current' if arm == 'R' else 'cached5',
                      writeback='current node latent;zero output;no new endpoint')
    if arm in ROUTING_ARMS:
        result.update(family='pb4u_response_routing_20261011_v1',
                      stage_feedback=[5, 10], response_calls=3, physical_writebacks=1,
                      matched_stage_initialization='S', matched_global_initialization='R',
                      global_tokens=64, global_width=128, global_heads=4, global_exchanges=2,
                      shared_global_module=True, response_context='fresh_assembled_changes',
                      response_length_scale_m=0.1, graph_context='pre_feedback_current',
                      quadrature='normalized rest/body area;equal object and source mass',
                      routing='separate_graph_and_response_encoders_and_KV_rows',
                      writeback='current node latent;zero output;no new endpoint')
    if arm in FUSION_ARMS:
        result.update(family='pb4u_state_innovation_20261011_v1',
                      stage_feedback=[5, 10], response_calls=3, physical_writebacks=1,
                      matched_stage_initialization='S', matched_global_initialization='R',
                      matched_change_initialization='A', exact_matched_RF_RG_capacity=True,
                      global_tokens=64, global_width=128, global_heads=4, global_exchanges=2,
                      shared_global_module=True, response_context='full_state_plus_fresh_changes',
                      response_length_scale_m=0.1, current_condition_width=24,
                      graph_context='pre_feedback_current',
                      quadrature='normalized rest/body area;equal object mass;one row per node',
                      routing='paired_same_node_before_compression',
                      fusion='additive_conditioned_ff' if arm == 'RF' else 'conditioned_gate',
                      initial_value_and_condition='zero output;gate .5;preserve R source',
                      writeback='current node latent;zero output;no new endpoint')
    if arm in COEVOLUTION_ARMS:
        result.update(family='pb4u_stage_coevolution_20261012_v1',
                      stage_feedback=[5, 10], response_calls=3, physical_writebacks=1,
                      matched_stage_initialization='R', matched_global_initialization='R',
                      global_tokens=64, global_width=128, global_heads=4, global_exchanges=2,
                      shared_global_module=True, graph_context='pre_feedback_current',
                      response_context='current',
                      quadrature='normalized cloth rest/body target surface area;equal type mass',
                      working_state_carry=arm in ('W', 'WJ', 'WJK'),
                      carry_units='native meter-valued acceleration-like frame increment',
                      carry_gradient='full refined_increment-minus-postfeedback_decoder;no detach',
                      geometry_conditioned_hinge=arm in ('J', 'WJ', 'WJK'),
                      hinge_metric='four unique vertex gradients;free inverse-mass Gram',
                      hinge_gain='2*G_rest/(G_current+G_rest);rest once per frame',
                      response_order='typed1,K,typed2' if arm == 'WJK' else 'K,typed1,typed2',
                      new_parameters=0, trainable_parameters=6324702,
                      writeback='final native inverse then carry then shared response;single endpoint')
    if arm in SELF_ARMS:
        result = protocol('WJ', iterations)
        messages = arm == 'WJ_LS'
        result.update(arm=arm, family='pb4u_self_interaction_20261013_v1', base_organization='WJ',
            matched_stage_initialization='original untrained R;not trained WJ',
            original_self_target=dict(source='SENC66e433f9668982aa7d79fc31fd2892c57f19e5ef',
                expression='30000*sum_regions(abs(sum_faces(p0 dot (p1 cross p2))))',
                norm='l1', weight=30000, divide_by_six=False, loss_only_boundary_caps=True,
                failure_is_not_zero=True, criterion_calls_per_forward=1),
            self_messages=messages, self_queries_per_frame=3 if messages else 0,
            self_message_calls_per_frame=6 if messages else 0,
            self_candidates='q_hat K11 including self;1e-7<d<0.02m;unique pairs then mesh exclusion;bidirectional',
            self_features='relative_position/h,distance/h,relative_frame_displacement/h,normal_dot;8',
            self_scale='existing cloth.average_edge_length scalar',
            self_encoder='mlp(8,128,norm=True);3Linear/2ReLU/LayerNorm',
            self_message='mlp(384,128,norm=True);receiver mean;shared across calls',
            self_projection='separate zero Linear(128,128);only receivers with neighbors',
            response_order='K,self,typed1,self,typed2' if messages else 'K,typed1,typed2',
            new_parameters=133504 if messages else 0,
            trainable_parameters=6458206 if messages else 6324702,
            total_capacities_matched=False, geometry_feature_gradients=True,
            adjacency_gradients=False, new_self_force_or_position_head=False,
            random_stream_reset_after_construction=True)
    return result


def install(project, data):
    os.environ['PB4U_PROJECT'] = str(project.resolve())
    os.environ['PB4U_DATA'] = str(data.resolve())
    if str(project.resolve()) not in sys.path:
        sys.path.insert(0, str(project.resolve()))
    if 'utils.arguments' not in sys.modules:
        install_official_py312_compat(project)
    for name in ('relation_response', 'graph_variants', 'graph_context', 'response_routing', 'state_feedback', 'geometry_feedback', 'self_interaction', 'self_contact_targets'):
        key = 'models.core.'+name
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, ROOT/'arcon/models'/(name+'.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
    anchor = sys.modules['models.core.relation_response']
    anchor.ACTIVE_MODE = 'R_geom'
    return anchor, sys.modules['models.core.graph_variants']


def configuration(args, arm, device, run_dir):
    anchor, screen = install(args.project, args.data)
    import models.pb4u as wrapper
    from utils.arguments import load_params
    coevolution = sys.modules['models.core.geometry_feedback']
    self_interaction = sys.modules['models.core.self_interaction']
    if arm in SELF_ARMS and iteration_budget(args) != 7500:
        raise ValueError('The approved self-interaction screen fixes 7500 iterations')
    wrapper.Model = (self_interaction.SelfInteractionModel if arm in SELF_ARMS else
                     coevolution.GeometryFeedbackModel if arm in COEVOLUTION_ARMS else
                     screen.StageResponseModel if arm in ('S', 'T')+GLOBAL_ARMS+ROUTING_ARMS+FUSION_ARMS else
                     anchor.RelationResponseModel if arm == 'G' else anchor.NativeModel)
    modules, cfg = load_params('pb4u', config_dir=str(args.project/'configs'))
    cfg.model.pb4u.core_model = ('pb4u' if arm == 'N' else
                                'self_interaction' if arm in SELF_ARMS else
                                'graph_context' if arm in GLOBAL_ARMS else
                                'response_routing' if arm in ROUTING_ARMS else
                                'state_feedback' if arm in FUSION_ARMS else
                                'geometry_feedback' if arm in COEVOLUTION_ARMS else 'graph_variants')
    if arm in SELF_ARMS:
        from omegaconf import OmegaConf, open_dict
        target = sys.modules['models.core.self_contact_targets']
        modules['criterions'][SELF_CRITERION] = target
        with open_dict(cfg.criterions):
            cfg.criterions[SELF_CRITERION] = OmegaConf.structured(target.Config)
    cfg.model.pb4u.screen_arm = arm
    cfg.device = device
    cfg.step_start = 0
    cfg.runner.pb4u.optimizer.step_start = 0
    cfg.runner.pb4u.roll_init = 1
    cfg.runner.pb4u.increase_roll_every = 500
    cfg.runner.pb4u.roll_max = 5
    cfg.experiment.max_iter = iteration_budget(args)
    cfg.experiment.checkpoint_path = None
    cfg.experiment.save_checkpoint_every = 500
    cfg.experiment.name = 'paired_screen_'+arm
    cfg.run_dir = str(run_dir.resolve())
    return modules, cfg


def compare_values(reference, candidate):
    differences = {key: float((candidate[key].detach().cpu().double()-value.double()).abs().max())
                   for key, value in reference.items()}
    return dict(keys=len(reference), max_absolute_difference=max(differences.values(), default=0.),
                changed_keys=[key for key, delta in differences.items() if delta != 0.])


def counts(runner, native_keys):
    return dict(parameters=sum(x.numel() for x in runner.parameters()),
                trainable_parameters=sum(x.numel() for x in runner.parameters() if x.requires_grad),
                original_trainable=sum(x.numel() for k, x in runner.named_parameters()
                                       if k in native_keys and x.requires_grad),
                added_trainable=sum(x.numel() for k, x in runner.named_parameters()
                                    if k not in native_keys and x.requires_grad))


def prepare(args):
    from omegaconf import OmegaConf
    from utils.arguments import create_runner
    initial_dir = args.output/'initializations'
    initial_dir.mkdir(parents=True, exist_ok=True)
    report = dict(state='preparing', source_revision=source_revision(), seed=args.seed, arms={})
    canonical, paired_source, paired_response = {}, {}, {}
    for arm in ARMS:
        set_seed(args.seed)
        modules, cfg = configuration(args, arm, 'cpu', args.output/'trains'/arm)
        _, runner, aux = create_runner(modules, cfg)
        state = runner.state_dict()
        if arm == 'N':
            canonical = {k: v.detach().cpu().clone() for k, v in state.items()}
        else:
            state.update(canonical)
        if arm == 'M':
            paired_source = {k: v.clone() for k, v in state.items() if k.startswith(SOURCE_PREFIX)}
        elif arm == 'C':
            state.update(paired_source)
        elif arm == 'G':
            paired_response = {k: v.clone() for k, v in state.items() if k.startswith(RESPONSE_PREFIX)}
        elif arm == 'S':
            state.update(paired_response)
        runner.load_state_dict(state, strict=True)
        equality = compare_values(canonical, runner.state_dict())
        if equality['changed_keys']:
            raise ValueError({'original_initialization_mismatch': equality})
        pairing = (compare_values(paired_source, runner.state_dict()) if arm == 'C' else
                   compare_values(paired_response, runner.state_dict()) if arm == 'S' else None)
        record = dict(counts(runner, canonical), original_equality=equality, new_pair_equality=pairing,
                      output_normalizer_accumulations=float(runner.model._output_normalizer._num_accumulations),
                      optimizer_states=len(aux['optimizer'].state),
                      actual_initial_lr=float(aux['optimizer'].param_groups[0]['lr']))
        if pairing and pairing['changed_keys']:
            raise ValueError({'added_initialization_mismatch': pairing})
        payload = dict(training_module=runner.state_dict(), config=OmegaConf.to_container(cfg),
                       architecture_protocol=protocol(arm, iteration_budget(args)), global_step=0, optimizer_updates=0,
                       initialization=record, source_revision=source_revision())
        torch.save(payload, initial_dir/(arm+'.pth'))
        report['arms'][arm] = record
        write_json(initial_dir/'initialization_record.json', report)
        print('INITIALIZED', arm, json.dumps(record), flush=True)
        del runner, aux, payload, state
        gc.collect()
    report['state'] = 'complete'
    report['native_state_keys'] = list(canonical)
    write_json(initial_dir/'initialization_record.json', report)



def prepare_terminal(args):
    """Copy the actual saved S original initialization;never use trained weights."""
    from utils.arguments import create_runner
    from omegaconf import OmegaConf
    set_seed(args.seed)
    modules, cfg = configuration(args, 'T', 'cpu', args.output/'trains/T')
    _, runner, aux = create_runner(modules, cfg)
    directory = args.output/'initializations'
    source = torch.load(directory/'S.pth', map_location='cpu', weights_only=False)
    native = torch.load(directory/'N.pth', map_location='cpu', weights_only=False)
    if source['global_step'] != 0 or source['architecture_protocol'] != protocol('S'):
        raise ValueError('T requires the saved original paired S initialization')
    runner.load_state_dict(source['training_module'], strict=True)
    paired = compare_values(source['training_module'], runner.state_dict())
    original = compare_values(native['training_module'], runner.state_dict())
    record = dict(counts(runner, native['training_module']), original_equality=original,
                  new_pair_equality=paired, matched_S_initialization=paired,
                  output_normalizer_accumulations=float(runner.model._output_normalizer._num_accumulations),
                  optimizer_states=len(aux['optimizer'].state),
                  actual_initial_lr=float(aux['optimizer'].param_groups[0]['lr']),
                  source_initialization=str(directory/'S.pth'))
    if paired['changed_keys'] or original['changed_keys'] or record['parameters'] != source['initialization']['parameters']:
        raise ValueError({'terminal_initialization':record})
    payload=dict(training_module=runner.state_dict(),config=OmegaConf.to_container(cfg),
                 architecture_protocol=protocol('T', iteration_budget(args)),global_step=0,optimizer_updates=0,
                 initialization=record,source_revision=source_revision())
    torch.save(payload,directory/'T.pth')
    write_json(directory/'terminal_initialization_record.json',dict(state='complete',arm='T',**record))
    print('INITIALIZED T',json.dumps(record),flush=True)



def prepare_global(args):
    """Copy every saved S initial field;pair all added global parameters/buffers."""
    from utils.arguments import create_runner
    from omegaconf import OmegaConf
    directory = args.output/'initializations'
    directory.mkdir(parents=True, exist_ok=True)
    source = torch.load(directory/'S.pth', map_location='cpu', weights_only=False)
    native = torch.load(directory/'N.pth', map_location='cpu', weights_only=False)
    if source['global_step'] != 0 or source['architecture_protocol'] != protocol('S'):
        raise ValueError('Global controls require the original saved S initialization')
    report = dict(state='preparing', source_revision=source_revision(), arms={}, seed=args.seed)
    global_state = None
    for arm in GLOBAL_ARMS:
        set_seed(args.seed)
        modules, cfg = configuration(args, arm, 'cpu', args.output/'trains'/arm)
        _, runner, aux = create_runner(modules, cfg)
        state = runner.state_dict()
        state.update(source['training_module'])
        if global_state is None:
            global_state = {k:v.clone() for k,v in state.items() if k.startswith(GLOBAL_PREFIX)}
        else:
            state.update(global_state)
        runner.load_state_dict(state, strict=True)
        original = compare_values(native['training_module'], runner.state_dict())
        paired_s = compare_values(source['training_module'], runner.state_dict())
        paired_global = compare_values(global_state, runner.state_dict())
        if any(x['changed_keys'] for x in (original, paired_s, paired_global)):
            raise ValueError('Paired initial field mismatch')
        record = dict(counts(runner, native['training_module']), original_equality=original,
                      matched_S_initialization=paired_s, global_initialization=paired_global,
                      optimizer_states=len(aux['optimizer'].state),
                      actual_initial_lr=float(aux['optimizer'].param_groups[0]['lr']),
                      output_normalizer_accumulations=float(runner.model._output_normalizer._num_accumulations))
        payload = dict(training_module=runner.state_dict(), config=OmegaConf.to_container(cfg),
                       architecture_protocol=protocol(arm, iteration_budget(args)), global_step=0, optimizer_updates=0,
                       initialization=record, source_revision=source_revision())
        torch.save(payload, directory/(arm+'.pth'))
        report['arms'][arm] = record
        write_json(directory/'global_initialization_record.json', report)
        print('INITIALIZED', arm, json.dumps(record), flush=True)
        del runner, aux, state, payload
        gc.collect()
    report['state'] = 'complete'
    write_json(directory/'global_initialization_record.json', report)


def prepare_routing(args):
    """Keep every saved S field and all shape-compatible initial R global fields."""
    from utils.arguments import create_runner
    from omegaconf import OmegaConf
    directory = args.output/'initializations'
    source = torch.load(directory/'R.pth', map_location='cpu', weights_only=False)
    stage = torch.load(directory/'S.pth', map_location='cpu', weights_only=False)
    native = torch.load(directory/'N.pth', map_location='cpu', weights_only=False)
    if source['global_step'] != 0 or source['architecture_protocol'] != protocol('R'):
        raise ValueError('Routing requires original saved R initialization')
    set_seed(args.seed)
    modules, cfg = configuration(args, 'A', 'cpu', args.output/'trains'/'A')
    _, runner, aux = create_runner(modules, cfg)
    state = runner.state_dict()
    shared = {k:v for k,v in source['training_module'].items() if k in state and state[k].shape == v.shape}
    state.update(shared)
    runner.load_state_dict(state, strict=True)
    original = compare_values(native['training_module'], runner.state_dict())
    paired_s = compare_values(stage['training_module'], runner.state_dict())
    if original['max_absolute_difference'] or paired_s['max_absolute_difference']:
        raise ValueError('Original and stage initialization must be paired')
    record = dict(state='complete', source_revision=source_revision(), arm='A', seed=args.seed,
                  original_initialization=original, stage_initialization=paired_s,
                  shared_R_initial_fields=len(shared),
                  new_fields=[k for k in state if k not in shared],
                  capacity=counts(runner, native['training_module']),
                  paired_optimizer_initialization='new native Adam with no saved trained optimizer',
                  source_initialization=str(directory/'R.pth'))
    torch.save(dict(training_module=runner.state_dict(), config=OmegaConf.to_container(cfg),
                    architecture_protocol=protocol('A'), global_step=0, initialization=record,
                    source_revision=source_revision(), seed=args.seed), directory/'A.pth')
    write_json(directory/'routing_initialization_record.json', record)
    print('ROUTING_INITIALIZATION '+json.dumps(record), flush=True)


def prepare_fusion(args):
    """Only untrained source fields; match every RF/RG parameter and buffer."""
    from utils.arguments import create_runner
    from omegaconf import OmegaConf
    directory = args.output / 'initializations'
    directory.mkdir(parents=True, exist_ok=True)
    sources = {arm: torch.load(directory / (arm + '.pth'), map_location='cpu', weights_only=False)
               for arm in ('N', 'S', 'R', 'A')}
    for arm, payload in sources.items():
        if payload['global_step'] != 0 or payload['architecture_protocol'] != protocol(arm):
            raise ValueError('Fusion needs the saved untrained paired ' + arm + ' initialization')
    reference, report = None, dict(state='preparing', source_revision=source_revision(), arms={}, seed=args.seed)
    for arm in FUSION_ARMS:
        set_seed(args.seed)
        modules, cfg = configuration(args, arm, 'cpu', args.output / 'trains' / arm)
        _, runner, aux = create_runner(modules, cfg)
        state = runner.state_dict()
        shared = {k: v for k, v in sources['R']['training_module'].items()
                  if k in state and state[k].shape == v.shape}
        changes = {k: v for k, v in sources['A']['training_module'].items()
                   if k.startswith(GLOBAL_PREFIX + 'response_changes.') and
                   k in state and state[k].shape == v.shape}
        state.update(shared)
        state.update(changes)
        if reference is not None:
            state.update(reference)
        runner.load_state_dict(state, strict=True)
        original = compare_values(sources['N']['training_module'], runner.state_dict())
        stage = compare_values(sources['S']['training_module'], runner.state_dict())
        global_pair = compare_values(sources['R']['training_module'], runner.state_dict())
        change_pair = compare_values(changes, runner.state_dict())
        matched = compare_values(reference, runner.state_dict()) if reference is not None else None
        if any(x['max_absolute_difference'] for x in (original, stage, global_pair, change_pair)):
            raise ValueError('Fusion original/S/R/change fields must preserve source initialization')
        if matched is not None and matched['max_absolute_difference']:
            raise ValueError('RF/RG full initialization must match')
        record = dict(state='complete', arm=arm, seed=args.seed, original_initialization=original,
            stage_initialization=stage, global_initialization=global_pair, change_initialization=change_pair,
            paired_RF_initialization=matched, shared_R_initial_fields=len(shared),
            shared_A_change_fields=len(changes), capacity=counts(runner, sources['N']['training_module']),
            actual_initial_lr=float(aux['optimizer'].param_groups[0]['lr']),
            source_initialization={k: str(directory / (k + '.pth')) for k in sources},
            paired_optimizer_initialization='new native Adam;no trained weights or optimizer')
        report['arms'][arm] = record
        torch.save(dict(training_module=runner.state_dict(), config=OmegaConf.to_container(cfg),
                        architecture_protocol=protocol(arm, iteration_budget(args)), global_step=0, initialization=record,
                        source_revision=source_revision(), seed=args.seed), directory / (arm + '.pth'))
        if reference is None:
            reference = {k: v.detach().clone() for k, v in runner.state_dict().items()}
        del runner, aux, state
        gc.collect()
        print('FUSION_INITIALIZATION ' + arm + ' ' + json.dumps(record), flush=True)
    capacities = [r['capacity']['parameters'] for r in report['arms'].values()]
    if len(set(capacities)) != 1:
        raise ValueError('RF/RG must have exactly equal parameter counts')
    report['state'] = 'complete'
    write_json(directory / 'fusion_initialization_record.json', report)


def prepare_coevolution(args):
    """Repackage identical untrained R weights/statistics with explicit arm/budget metadata."""
    from omegaconf import OmegaConf
    install(args.project, args.data)
    from utils.arguments import create_runner
    if args.reference_initialization is None:
        raise ValueError('An original untrained R initialization is required')
    source = torch.load(args.reference_initialization, map_location='cpu', weights_only=False)
    if source['global_step'] != 0 or source['architecture_protocol']['arm'] != 'R':
        raise ValueError('Co-evolution initialization must be the saved untrained R bundle')
    directory = args.output/'initializations'
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(state='preparing', source_revision=source_revision(), seed=args.seed,
                  iterations=iteration_budget(args), source_initialization=str(args.reference_initialization), arms={})
    for arm in ('R',)+COEVOLUTION_ARMS:
        set_seed(args.seed)
        modules, cfg = configuration(args, arm, 'cpu', args.output/'trains'/arm)
        _, runner, aux = create_runner(modules, cfg)
        runner.load_state_dict(source['training_module'], strict=True)
        paired = compare_values(source['training_module'], runner.state_dict())
        capacity = counts(runner, source['training_module'])
        if paired['max_absolute_difference'] or capacity['parameters'] != 6324784 or capacity['trainable_parameters'] != 6324702:
            raise ValueError('All five arms must match original R values and full capacity')
        record = dict(state='complete', arm=arm, seed=args.seed, paired_R_initialization=paired,
                      capacity=capacity, actual_initial_lr=float(aux['optimizer'].param_groups[0]['lr']),
                      source_initialization=str(args.reference_initialization),
                      paired_optimizer_initialization='new native Adam;no trained weights or optimizer')
        torch.save(dict(training_module=runner.state_dict(), config=OmegaConf.to_container(cfg),
                        architecture_protocol=protocol(arm, iteration_budget(args)), global_step=0,
                        initialization=record, source_revision=source_revision(), seed=args.seed), directory/(arm+'.pth'))
        report['arms'][arm] = record
        del runner, aux
        gc.collect()
        print('COEVOLUTION_INITIALIZATION '+arm+' '+json.dumps(record), flush=True)
    report['state'] = 'complete'
    write_json(directory/'coevolution_initialization_record.json', report)


def prepare_self(args):
    """Reuse backbone tensors and initialize the additional self-interaction parameters."""
    from omegaconf import OmegaConf
    install(args.project, args.data)
    from utils.arguments import create_runner
    if args.reference_initialization is None:
        raise ValueError('The saved original untrained R initialization is required')
    source = torch.load(args.reference_initialization, map_location='cpu', weights_only=False)
    if source['global_step'] != 0 or source['architecture_protocol']['arm'] != 'R' or source.get('optimizer_updates', 0):
        raise ValueError('Self-screen initialization cannot be a trained R/WJ bundle')
    directory = args.output/'initializations'
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(state='preparing', source_revision=source_revision(), seed=args.seed,
        iterations=iteration_budget(args), source_initialization=str(args.reference_initialization), arms={})
    for arm in SELF_ARMS:
        set_seed(args.seed)
        modules, cfg = configuration(args, arm, 'cpu', args.output/'trains'/arm)
        _, runner, aux = create_runner(modules, cfg)
        values = runner.state_dict()
        original = source['training_module']
        missing = set(original)-set(values)
        added = sorted(set(values)-set(original))
        if missing or any(not key.startswith(SELF_PREFIXES) for key in added):
            raise ValueError(dict(missing_original_keys=sorted(missing), unapproved_added_keys=added))
        for key, value in original.items():
            values[key] = value.clone()
        runner.load_state_dict(values, strict=True)
        paired = compare_values(original, runner.state_dict())
        capacity = counts(runner, original)
        increment = 133504 if arm == 'WJ_LS' else 0
        if paired['max_absolute_difference'] or capacity['parameters'] != 6324784+increment or capacity['trainable_parameters'] != 6324702+increment:
            raise ValueError('Common R values or specified self-module capacity do not match')
        projection_max = None
        if arm == 'WJ_LS':
            output = runner.model._learned_model.response.self_output
            projection_max = max(float(output.weight.detach().abs().max()), float(output.bias.detach().abs().max()))
            if projection_max != 0:
                raise ValueError('The new self residual must start at exact zero')
        if 'self_intersection_energy' not in runner.criterion_dict or len(runner.criterion_dict) != 7:
            raise ValueError('Both the factory and config must register original six plus GIA')
        record = dict(state='complete', arm=arm, seed=args.seed, paired_R_initialization=paired,
            capacity=capacity, actual_initial_lr=float(aux['optimizer'].param_groups[0]['lr']),
            source_initialization=str(args.reference_initialization), added_state_keys=added,
            self_projection_initial_max_absolute=projection_max,
            same_total_capacity_as_target_only=arm == 'WJ_L',
            paired_optimizer_initialization='new native Adam;no trained weights or optimizer',
            required_criterion_registered='self_intersection_energy' in runner.criterion_dict)
        torch.save(dict(training_module=runner.state_dict(), config=OmegaConf.to_container(cfg),
            architecture_protocol=protocol(arm, iteration_budget(args)), global_step=0,
            initialization=record, source_revision=source_revision(), seed=args.seed), directory/(arm+'.pth'))
        report['arms'][arm] = record
        del runner, aux, values
        gc.collect()
        print('SELF_INITIALIZATION '+arm+' '+json.dumps(record), flush=True)
    report['state'] = 'complete'
    write_json(directory/'self_initialization_record.json', report)


def create_training(args, arm, save_configuration=True):
    from utils.arguments import create_modules
    from omegaconf import OmegaConf
    set_seed(args.seed)
    output = args.output/'trains'/arm
    modules, cfg = configuration(args, arm, args.device, output)
    created = create_modules(modules, cfg)
    checkpoint = torch.load(args.output/'initializations'/(arm+'.pth'), map_location='cpu', weights_only=False)
    ckpt_protocol = checkpoint['architecture_protocol']
    expected_protocol = protocol(arm, iteration_budget(args))
    is_resume = getattr(args, 'resume_from', None) is not None
    if is_resume:
        # Allow resume: check everything except budget-dependent fields.
        resume_mismatch = {k: (ckpt_protocol.get(k), expected_protocol[k])
                          for k in expected_protocol
                          if k not in ('data_iterations', 'expected_forwards', 'expected_optimizer_updates',
                                       'extended_from', 'extended_at')
                          and ckpt_protocol.get(k) != expected_protocol[k]}
        if resume_mismatch:
            raise ValueError('Resume protocol mismatch on non-budget fields: '+str(resume_mismatch))
    else:
        if ckpt_protocol != expected_protocol:
            raise ValueError('Initialization protocol mismatch')
    created[2].load_state_dict(checkpoint['training_module'], strict=True)
    if save_configuration:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output/'config_resolved.yaml')
    init = checkpoint['initialization']
    del checkpoint
    # Construction/module count must not move sampling, material, tau or K streams.
    set_seed(args.seed)
    return created, cfg, init


def configure_recording(model, enabled):
    core = model._learned_model
    if hasattr(core.aggr_processor, 'source_interaction'):
        core.aggr_processor.source_interaction.record = enabled
    if hasattr(core, 'response'):
        core.response.record = enabled
    if hasattr(core, 'stage_feedback'):
        core.stage_feedback.record = enabled
    if hasattr(core, 'global_context'):
        core.global_context.record = enabled


def execution_record(model):
    core = model._learned_model
    record = dict(getattr(core, 'execution', {}))
    if hasattr(core.aggr_processor, 'source_interaction'):
        record['source'] = core.aggr_processor.source_interaction.last
    if hasattr(core, 'response') and core.response.last is not None:
        response = core.response.last
        record['response'] = dict(K=response['repeat'], contact_pairs=response['contact_pairs'],
                                 increment_rms=float(response['increment'].detach().square().mean().sqrt()))
        core.response.last = None
    return record


def check_real_sample(args):
    """Exactly one common official training sample, no optimizer or rollout probe."""
    from utils.common import move2device, add_field_to_pyg_batch
    from utils.arguments import create_runner
    created, cfg, _ = create_training(args, 'N', save_configuration=False)
    loader, _, native, _ = created
    data_loader = loader.create_dataloader()
    iterator = iter(data_loader)
    raw = move2device(next(iterator), args.device)
    del iterator, data_loader
    raw = add_field_to_pyg_batch(raw, 'iter', [100], 'cloth', reference_key=None)
    raw = native.add_cloth_obj(raw)
    state = native.collect_sample(raw, 0, None, random_ts=True)
    state = native.collision_solver.solve(state)
    edges = state['cloth', 'mesh_edge', 'cloth'].edge_index
    rest = state['cloth'].rest_pos
    edge_mean = (rest[edges[0]]-rest[edges[1]]).norm(dim=-1).mean().item()
    repeat = max(1, int(native.model.propagation_distance/edge_mean))
    report = dict(state='running', K=repeat, arms={},
                  cloth_nodes=len(state['cloth'].pos), body_nodes=len(state['obstacle'].pos))
    reference = None
    del native, created, raw, loader
    gc.collect()
    torch.cuda.empty_cache()
    try:
        for arm in (args.check_arms or ARMS):
            modules, cfg = configuration(args, arm, args.device, args.output/'trains'/arm)
            _, runner, _ = create_runner(modules, cfg, create_aux_modules=False)
            initial = torch.load(args.output/'initializations'/(arm+'.pth'), map_location='cpu', weights_only=False)
            runner.load_state_dict(initial['training_module'], strict=True)
            del initial
            model = runner.model
            configure_recording(model, True)
            calls = dict(readouts=0, response_calls=0, feedbacks=0)
            def count_readout(*unused): calls['readouts'] += 1
            def count_response(*unused): calls['response_calls'] += 1
            def count_feedback(*unused): calls['feedbacks'] += 1
            hooks = [model._output_normalizer.register_forward_hook(count_readout)]
            core = model._learned_model
            if hasattr(core, 'response'): hooks.append(core.response.register_forward_hook(count_response))
            if hasattr(core, 'stage_feedback'): hooks.append(core.stage_feedback.register_forward_hook(count_feedback))
            before = float(model._output_normalizer._num_accumulations)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                output = model(state.clone(), repeat=repeat, is_training=True)
            actual = {key: output['cloth'][key].detach().clone()
                      for key in ('pred_pos', 'pred_velocity', 'pred_acceleration')}
            if reference is None: reference = {k: v.clone() for k, v in actual.items()}
            delta = {k: float((v-reference[k]).abs().max()) for k, v in actual.items()}
            accumulations = float(model._output_normalizer._num_accumulations)-before
            expected_calls = dict(readouts=3 if arm in ('S', 'T')+GLOBAL_ARMS+ROUTING_ARMS+FUSION_ARMS else 1,
                                  response_calls=3 if arm in ('S', 'T')+GLOBAL_ARMS+ROUTING_ARMS+FUSION_ARMS else 1 if arm == 'G' else 0,
                                  feedbacks=2 if arm in ('S', 'T')+GLOBAL_ARMS+ROUTING_ARMS+FUSION_ARMS else 0)
            row = dict(initial_endpoint_difference=delta, output_normalizer_delta=accumulations,
                       calls=dict(calls), execution=execution_record(model),
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            report['arms'][arm] = row
            write_json(args.output/'real_sample_check.json', report)
            if max(delta.values()) > 1e-7 or accumulations != 1 or calls != expected_calls:
                raise ValueError({'arm': arm, 'direct_check': row, 'expected_calls': expected_calls})
            if arm in GLOBAL_ARMS+ROUTING_ARMS+FUSION_ARMS:
                execution = row['execution']
                if any(execution[k] != 2 for k in ('compression_calls', 'token_update_calls', 'global_writebacks')):
                    raise ValueError('Each arm needs two active global exchanges')
                # Zero output starts upstream gradients at zero. Disposable scratch
                # updates open nested new paths; saved initialization is unchanged.
                scratch_optimizer = torch.optim.Adam(runner.parameters(), lr=5e-5)
                backward = []
                measurements = 4 if arm in FUSION_ARMS else 2
                for measurement in range(measurements):
                    scratch_optimizer.zero_grad(set_to_none=True)
                    tick = time.monotonic()
                    predicted = model(state.clone(), repeat=repeat, is_training=True)
                    losses = runner.criterion_pass(predicted)
                    total = sum(losses.values())
                    total.backward()
                    gradients = gradient_record(runner)
                    backward.append(dict(measurement=measurement, loss=float(total.detach()),
                                         gradients=gradients, elapsed_seconds=time.monotonic()-tick,
                                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
                    if measurement < measurements - 1:
                        torch.nn.utils.clip_grad_norm_(runner.parameters(), runner.mcfg.grad_clip, error_if_nonfinite=True)
                        scratch_optimizer.step()
                names = ('node_output', 'compress', 'token_attention', 'ffn', 'write')
                names += (('graph_input', 'response_input', 'response_changes.node_encoder',
                           'response_changes.face_encoder') if arm in ROUTING_ARMS else ('context_input',))
                if arm in FUSION_ARMS:
                    names += ('response_changes.node_encoder', 'response_changes.face_encoder',
                              'value_encoder', 'value_output', 'condition_encoder', 'condition_output')
                required = tuple(GLOBAL_PREFIX+k for k in names)
                if any(not math.isfinite(backward[-1]['gradients'].get(k) or 0.) or
                       (backward[-1]['gradients'].get(k) or 0.) <= 0 for k in required):
                    raise ValueError('Missing effective global gradient after zero-output opens')
                row['scratch_backward'] = backward
                row['scratch_updates_discarded'] = measurements - 1
                del scratch_optimizer, predicted, losses, total
            for hook in hooks: hook.remove()
            print('CHECKED', arm, json.dumps(row), flush=True)
            del runner, model, core, output, actual
            gc.collect()
            torch.cuda.empty_cache()
        report['state'] = 'complete'
    except BaseException as exc:
        report.update(state='failed', failure=type(exc).__name__+': '+str(exc))
        raise
    finally:
        write_json(args.output/'real_sample_check.json', report)


def gradient_record(runner):
    prefixes = ['model._learned_model.'+name for name in
                ('node_encoder', 'aggr_processor.message_mlp', 'aggr_processor.fuse_mlp', 'decoder')]
    prefixes += ['model._learned_model.processor_steps.'+str(i)+'.' for i in range(15)]
    prefixes += [SOURCE_PREFIX+x for x in ('source_encoder', 'output', 'transport_gate', 'exchange_gate', 'value', 'pair')]
    prefixes += [RESPONSE_PREFIX+x for x in ('node_encoder', 'face_encoder', 'edge_encoder', 'world_encoder',
                  'hinge_encoder', 'aggr_processor', 'processor_steps', 'edge_head', 'hinge_head', 'contact_head')]
    prefixes += list(SELF_PREFIXES)
    prefixes += [FEEDBACK_PREFIX+x for x in ('node_encoder', 'mesh_encoder', 'face_encoder', 'world_encoder',
                                            'node_output', 'mesh_output', 'world_output')]
    prefixes += [GLOBAL_PREFIX+x for x in ('seeds', 'type_embedding', 'context_input', 'compress',
                    'token_attention', 'ffn', 'write', 'node_output', 'graph_input', 'response_input',
                    'source_embedding', 'response_changes.node_encoder', 'response_changes.face_encoder',
                    'value_encoder', 'value_output', 'condition_encoder', 'condition_output')]
    result = {}
    parameters = list(runner.named_parameters())
    for prefix in prefixes:
        matches = [v for key, v in parameters if key.startswith(prefix)]
        if not matches: continue
        values = [v.grad.detach().abs().max() for v in matches if v.grad is not None]
        result[prefix] = float(torch.stack(values).max()) if values else None
    return result


def train(args):
    from omegaconf import OmegaConf
    created, cfg, init = create_training(args, args.arm)
    loader, runner_module, runner, aux = created
    output = args.output/'trains'/args.arm
    counters = dict(iteration=0, data_iterations=0, updates=0, forwards=0)
    if args.arm in SELF_ARMS:
        # The immutable initialization contract still matches exactly. The active
        # numerical readout belongs to runtime evidence, not initialization weights.
        counters['self_target_readout_protocol'] = runner.criterion_dict['self_intersection_energy'].readout_protocol
    start = time.monotonic()
    history_mode = 'a' if getattr(args, 'resume_from', None) is not None else 'w'
    history = (output/'history.jsonl').open(history_mode, encoding='utf-8', buffering=1)
    revision = source_revision()
    execution_policy = dict(activation_recomputation=args.arm in ('M', 'C', 'S', 'T')+GLOBAL_ARMS+ROUTING_ARMS+FUSION_ARMS+COEVOLUTION_ARMS+SELF_ARMS,
                            exact_zero_source_pruning=args.arm in ('M', 'C'),
                            nonfinite_gradient_stops_before_Adam=True)
    def pre_runner(_, inputs):
        sample = inputs[0]
        step = int(torch.as_tensor(sample['cloth'].iter).reshape(-1)[0])
        counters.update(iteration=step, data_iterations=step,
                        rollout_steps=min(1+step//500, 5),
                        training_collision_weight=float(runner.criterion_dict['collision_penalty'].get_weight(step)),
                        sequence_name=sample.get('sequence_name', None), garment_name=sample.get('garment_name', None))
    def pre_model(_, inputs, kwargs):
        counters['forwards'] += 1
        cloth = inputs[0]['cloth']
        counters['material'] = {k: float(cloth[k].reshape(-1)[0]) for k in
                                ('density', 'lame_mu', 'lame_lambda', 'bending_coeff') if k in cloth}
        if args.arm in SELF_ARMS:
            counters['sampled_density'] = float(runner.cloth_obj.material.density.reshape(-1)[0])
        counters['tau'] = float(cloth.timestep.reshape(-1)[0])
        body = inputs[0]['obstacle'].target_pos
        counters['body_input_coordinates'] = body[[0, len(body)//2, len(body)-1]].detach().cpu().tolist()
        repeat = inputs[1] if len(inputs) > 1 else kwargs.get('repeat')
        counters['K'] = None if repeat is None else int(repeat)
        counters['normalizer_before'] = float(runner.model._output_normalizer._num_accumulations)
        configure_recording(runner.model, counters['updates'] < 10 and counters['iteration'] >= 100)
    def post_model(_, inputs, output_sample):
        after = float(runner.model._output_normalizer._num_accumulations)
        counters['normalizer_accumulations'] = after
        if after-counters['normalizer_before'] != 1:
            raise ValueError('Output statistics must accumulate once per real frame')
        if counters['updates'] < 10 and counters['iteration'] >= 100:
            counters['execution'] = execution_record(runner.model)
    def optimizer(self, losses, optimizer=None, scheduler=None):
        if args.arm in SELF_ARMS:
            counters['self_target'] = {key: runner.criterion_dict['self_intersection_energy'].last[key]
                for key in ('original_gia_seconds', 'criterion_forward_seconds', 'region_count',
                            'cgal_enumeration_protocol', 'cgal_normalization_seconds',
                            'original_gia_seconds_without_normalization',
                            'raw_l1_triple_product', 'geometric_volume_m3', 'readout_protocol',
                            'padding_mask', 'padding_slots', 'actual_region_faces',
                            'unmasked_author_loss', 'padding_flipped_region_signs')}
            counters['self_target_recorded_at'] = dict(data_iteration=counters['data_iterations'],
                                                      model_forward=counters['forwards'])
        scalar_losses = {k: float(v.detach()) for k, v in losses.items()}
        if not all(math.isfinite(v) for v in scalar_losses.values()):
            raise FloatingPointError('Nonfinite physical loss: '+str(scalar_losses))
        if optimizer is None: return
        # Same official backward, clipping, Adam and scheduler. Only failure
        # reporting differs: stop before a nonfinite clip can corrupt weights.
        optimizer.zero_grad()
        sum(losses.values()).backward()
        clip_norm = None
        if self.mcfg.grad_clip is not None:
            clip_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.mcfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None: scheduler.step()
        counters['updates'] += 1
        row = dict(counters, losses=scalar_losses, elapsed_seconds=time.monotonic()-start,
                   preclip_gradient_norm=float(clip_norm) if clip_norm is not None else None,
                   learning_rates=[float(group['lr']) for group in optimizer.param_groups])
        if counters['updates'] <= 10:
            row['gradients'] = gradient_record(runner)
            row['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            print('MEASURE '+json.dumps(row, allow_nan=False), flush=True)
        history.write(json.dumps(row, allow_nan=False)+'\n')
        if counters['updates'] <= 10 or counters['updates'] % 100 == 0:
            write_json(output/'progress.json', dict(row, state='training', arm=args.arm,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       data_iterations_budget=iteration_budget(args), expected_optimizer_updates=expected_counts(iteration_budget(args))[2]))
        counters.pop('execution', None)
    hooks = [runner.register_forward_pre_hook(pre_runner),
             runner.model.register_forward_pre_hook(pre_model, with_kwargs=True),
             runner.model.register_forward_hook(post_model)]
    runner.optimizer_step = types.MethodType(optimizer, runner)
    def save(training_module, aux_modules, config, file):
        target = Path(file)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(training_module=training_module.state_dict(), config=OmegaConf.to_container(config),
                       architecture_protocol=protocol(args.arm, iteration_budget(args)), global_step=counters['iteration'],
                       data_iterations=counters['data_iterations'], optimizer_updates=counters['updates'],
                       initialization=init, source_revision=revision, execution_policy=execution_policy, seed=args.seed,
                       random_states=dict(python=random.getstate(), numpy=np.random.get_state(),
                                          torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
        if args.arm in SELF_ARMS:
            criterion = runner.criterion_dict['self_intersection_energy']
            payload['self_target_readout'] = dict(readout_protocol=criterion.readout_protocol,
                cgal_enumeration_protocol=criterion.cgal_enumeration_protocol,
                actual_region_faces_only=True)
        payload.update({k: v.state_dict() for k, v in aux_modules.items() if hasattr(v, 'state_dict')})
        torch.save(payload, target)
    runner_module.save_checkpoint = save
    def stop(signum, frame): raise InterruptedError('Training stopped by signal '+str(signum))
    signal.signal(signal.SIGTERM, stop)
    write_json(output/'initialization_record.json', init)
    write_json(output/'progress.json', dict(counters, state='warmup', arm=args.arm))
    torch.cuda.reset_peak_memory_stats()
    step = 0
    resume_from_step = 0
    if getattr(args, 'resume_from', None) is not None:
        resume_ckpt = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        resume_from_step = int(resume_ckpt.get('global_step', 0))
        runner.load_state_dict(resume_ckpt['training_module'], strict=True)
        if 'optimizer' in aux and hasattr(aux['optimizer'], 'load_state_dict') and 'optimizer' in resume_ckpt:
            aux['optimizer'].load_state_dict(resume_ckpt['optimizer'])
        if 'scheduler' in aux and hasattr(aux['scheduler'], 'load_state_dict') and 'scheduler' in resume_ckpt:
            aux['scheduler'].load_state_dict(resume_ckpt['scheduler'])
        rs = resume_ckpt.get('random_states', {})
        if rs.get('python'): random.setstate(rs['python'])
        if rs.get('numpy') is not None: np.random.set_state(rs['numpy'])
        if rs.get('torch') is not None: torch.set_rng_state(rs['torch'])
        if rs.get('cuda') is not None: torch.cuda.set_rng_state_all(rs['cuda'])
        step = resume_from_step
        counters['updates'] = int(resume_ckpt.get('optimizer_updates', 0))
        counters['forwards'] = expected_counts(resume_from_step)[1]
        del resume_ckpt
        gc.collect()
        print('RESUME_FROM step=%d updates=%d' % (resume_from_step, counters['updates']), flush=True)
    try:
        for epoch in range(int(cfg.experiment.n_epochs)):
            step = runner_module.run_epoch(runner, aux, loader.create_dataloader(), epoch, cfg, global_step=step)
            if step > iteration_budget(args): break
        actual = (counters['data_iterations'], counters['forwards'], counters['updates'])
        full_expected = expected_counts(iteration_budget(args))
        if resume_from_step > 0:
            resume_expected = expected_counts(resume_from_step)
            incremental_expected = tuple(full_expected[i] - resume_expected[i] for i in range(3))
            incremental_actual = (
                counters['data_iterations'] - resume_from_step,
                counters['forwards'] - resume_expected[1],
                counters['updates'] - resume_expected[2],
            )
            if incremental_actual != incremental_expected:
                raise ValueError({'actual_incremental': incremental_actual, 'expected_incremental': incremental_expected,
                                   'resume_from_step': resume_from_step})
        else:
            if actual != full_expected:
                raise ValueError({'actual_counts': actual, 'expected': full_expected})
        summary = dict(counters, state='complete', arm=args.arm, initialization=init,
                       elapsed_seconds=time.monotonic()-start,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       source_revision=revision, execution_policy=execution_policy)
        write_json(output/'train_summary.json', summary)
        write_json(output/'progress.json', summary)
        print('TRAIN_COMPLETE '+json.dumps(summary), flush=True)
    except BaseException as exc:
        summary = dict(counters, state='failed', arm=args.arm,
                       failure=type(exc).__name__+': '+str(exc), elapsed_seconds=time.monotonic()-start,
                       execution_policy=execution_policy, source_revision=revision)
        if 'self_target' in summary:
            summary['last_recorded_self_target'] = summary.pop('self_target')
            summary['last_recorded_self_target_at'] = summary.pop('self_target_recorded_at', None)
        if args.arm in SELF_ARMS:
            geometry = runner.criterion_dict['self_intersection_energy'].failure_geometry
            if geometry is not None:
                # Preserve the actual failing geometry independently of the
                # training checkpoint and any metadata serialization failure.
                summary['self_target_failure_details'] = geometry['details']
                try:
                    torch.save(geometry, output/'self_target_failure_geometry.pth')
                    summary['self_target_failure_geometry'] = str(output/'self_target_failure_geometry.pth')
                except Exception as geometry_error:
                    summary['self_target_failure_geometry_error'] = str(geometry_error)
        try:
            save(runner, aux, cfg, output/'failure_state.pth')
            summary['failure_state'] = str(output/'failure_state.pth')
        except Exception as snapshot_error:
            summary['failure_state_error'] = str(snapshot_error)
        def labelled(value):
            if isinstance(value, float) and not math.isfinite(value): return str(value)
            if isinstance(value, dict): return {k: labelled(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)): return [labelled(v) for v in value]
            return value
        summary = labelled(summary)
        write_json(output/'train_summary.json', summary)
        write_json(output/'progress.json', summary)
        raise
    finally:
        for hook in hooks: hook.remove()
        history.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'prepare-terminal', 'prepare-global', 'prepare-routing', 'prepare-fusion', 'prepare-coevolution', 'prepare-self', 'check', 'train'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Shared matched-architecture result root')
    parser.add_argument('--arm', choices=ALL_ARMS, default='N')
    parser.add_argument('--check-arms', nargs='+', choices=ALL_ARMS)
    parser.add_argument('--seed', type=int, default=57, choices=(57,))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--iterations', type=int, default=STEPS, choices=(5000, 7500, 15000))
    parser.add_argument('--reference-initialization', type=Path)
    parser.add_argument('--resume-from', type=Path, default=None,
                        help='Resume training from an existing checkpoint (e.g. step_0000007500.pth)')
    args = parser.parse_args()
    torch.set_num_threads(4)
    install(args.project, args.data)
    {'prepare':prepare,'prepare-terminal':prepare_terminal,'prepare-global':prepare_global,'prepare-routing':prepare_routing,'prepare-fusion':prepare_fusion,'prepare-coevolution':prepare_coevolution,'prepare-self':prepare_self,'check':check_real_sample,'train':train}[args.action](args)


if __name__ == '__main__':
    main()
