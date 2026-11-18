#!/usr/bin/env python3
"""Material-response training stages and full-state checkpoint restoration.

The public training driver supplies the original absolute iteration boundaries.
The inherited runner, physical objectives and optimizer schedule are unchanged."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import arcon.control.contact_feedback_stage as phase

course, shared, base = phase.course, phase.shared, phase.base
ARMS = ('A', 'B', 'AB', 'C')
READOUTS = {arm: 'target_face_floor' for arm in ARMS}
START, STEPS, MAX_SECONDS = 20000, 22500, 4 * 3600
START_UPDATES = course.expected_counts(START)[2]
RESUME_INFO = {}


def expected_counts(iterations=STEPS):
    return course.expected_counts(iterations)


def protocol(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    import arcon.control.contact_query_stage as parent_query
    value = copy.deepcopy(parent_query.protocol('Early-Query'))
    value.update(family='material_response_short_20261022_v1', arm=arm,
        parent_arm='Face-Early', material_bending_anchor=arm in ('A', 'AB', 'C'),
        material_contact_coordination=arm in ('B', 'AB'),
        material_bending_coordinate='sqrt(kappa/(4*rest_adjacent_area_sum))*current_length*signed_theta',
        bending_current_length_derivative=True, multiplier_state='explicit_per_cell_tensor_reset_each_frame',
        contact_material_iterations=2 if arm in ('B', 'AB') else 0,
        contact_domain='two_rest_mesh_hops_tangent_at_all_query_supports',
        contact_quadratic='mass_z2/(2dt2)+norm(J_elastic*(delta0+z))2/2',
        material_jacobians='exact_local_native_StVK_with_original_thickness_and_current_edge_bending',
        new_parameters=0, new_losses=0, extra_body_queries=0,
        training_walltime_limit_seconds=MAX_SECONDS)
    if arm == 'C':
        value.update(curvature_conditioning='leave_one_out_material_tensor_excess_relaxation',
            curvature_radius_m=.03, curvature_gate='positive_abs_curvature_excess_over_same_sign_neighbor_prediction',
            curvature_unsupported_rule='retain_A_relaxation', curvature_differentiation='full_theta_and_sparse_prediction',
            curvature_changes_native_residual_or_jacobian=False)
    return value


def material_configuration(cfg, arm):
    from omegaconf import open_dict
    with open_dict(cfg.model.pb4u):
        cfg.model.pb4u.material_response_arm = arm
        cfg.model.pb4u.material_response_thickness = float(cfg.criterions['pb4u.mataug.stretching_energy'].thickness)


def configuration(args, arm, device, run_dir, *, training=True):
    modules, cfg = course.configuration(args, 'Cell-ScaleFixed', device, run_dir, training=training)
    cfg.experiment.name = 'material_response_20261022_' + arm
    material_configuration(cfg, arm)
    cfg.experiment.max_iter = getattr(args, 'end_step', getattr(args, 'iterations', STEPS))
    cfg.experiment.save_checkpoint_every = 2500
    return modules, cfg


def configure_readout(runner, arm):
    import arcon.control.contact_query_stage as query
    response = runner.model._learned_model.response
    if response.material_arm != arm:
        raise ValueError('Material arm differs from the constructed module')
    if response.material_thickness != float(runner.criterion_dict['stretching_energy'].thickness):
        raise ValueError('Do not change the actual native membrane thickness')
    return query.configure_readout(runner, 'Early-Query')


def create(args):
    from utils.arguments import create_modules
    from omegaconf import OmegaConf
    global RESUME_INFO
    out = args.output/'trains'/args.arm
    if (out/'history.jsonl').exists() or (out/'train_summary.json').exists():
        raise RuntimeError('Preserve an already attempted continuation')
    parent = (args.resume_checkpoint if getattr(args, 'resume_checkpoint', None) is not None
              else args.face_parent_root/'trains/Face-Early/checkpoints/step_0000020000.pth')
    parent_protocol = protocol(args.arm) if getattr(args, 'resume_checkpoint', None) is not None else phase.protocol('Face-Early')
    parent = getattr(args, 'parent_checkpoint', None) or parent
    payload = torch.load(parent, map_location='cpu', weights_only=False)
    from arcon.compat import relocate_training_config
    relocate_training_config(payload, args.data, args.parent_root/'calibration.json')
    actual = (payload['global_step'], payload['forwards'], payload['optimizer_updates'])
    if actual != expected_counts(START) or payload['architecture_protocol'] != parent_protocol:
        raise ValueError('The complete approved 20k parent identity differs')
    if payload['execution_policy'] != 'outer_only':
        raise ValueError('Retain the parent outer_only recomputation policy')
    if not payload['optimizer']['state'] or payload['scheduler']['last_epoch'] != START_UPDATES:
        raise ValueError('The complete parent Adam/scheduler age is required')
    calibration = args.parent_root/'calibration.json'
    record = json.loads(calibration.read_text())
    saved_calibration = Path(payload['config']['criterions'][shared.TARGET_KEY]['calibration_file'])
    if (record['state'] != 'complete' or record['scale'] != 1722.2913807992006
            or json.loads(saved_calibration.read_text()) != record):
        raise ValueError('Retain the frozen original TRAIN self calibration')
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'calibration.json'
    if target.exists():
        if json.loads(target.read_text()) != record:
            raise ValueError('Existing output calibration differs')
    else:
        shutil.copyfile(calibration, target)
    shared.set_seed(57)
    modules, _ = configuration(args, args.arm, args.device, out)
    cfg = OmegaConf.create(copy.deepcopy(payload['config']))
    cfg.device, cfg.run_dir = args.device, str(out)
    cfg.experiment.name = 'material_response_20261022_' + args.arm
    material_configuration(cfg, args.arm)
    cfg.experiment.max_iter = args.end_step
    cfg.criterions[shared.TARGET_KEY].calibration_file = str(target)
    differences = phase.configuration_differences(payload['config'], OmegaConf.to_container(cfg))
    allowed = {('device',), ('run_dir',), ('experiment','name'), ('experiment','max_iter'),
               ('criterions',shared.TARGET_KEY,'calibration_file'),
               ('model','pb4u','material_response_arm'), ('model','pb4u','material_response_thickness')}
    if any(tuple(row['path']) not in allowed for row in differences):
        raise ValueError('Non-run-control training configuration changed: '+str(differences))
    created = create_modules(modules, cfg)
    runner, aux = created[2:]
    runner.load_state_dict(payload['training_module'], strict=True)
    shared.install(args).configure_mlp_recomputation(runner.model, False)
    configure_readout(runner, args.arm)
    for name in ('optimizer', 'scheduler'):
        aux[name].load_state_dict(payload[name])
    loaded_lr = [float(g['lr']) for g in aux['optimizer'].param_groups]
    if (loaded_lr != [float(g['lr']) for g in payload['optimizer']['param_groups']]
            or aux['scheduler'].last_epoch != START_UPDATES):
        raise ValueError('Restored original learning-rate schedule differs')
    normals = {k:float(v) for k,v in runner.state_dict().items() if k.endswith('_num_accumulations')}
    saved = {k:float(v) for k,v in payload['training_module'].items() if k.endswith('_num_accumulations')}
    if normals != saved:
        raise ValueError('A parent normalizer statistic changed')
    parent_parameters = sum(v.numel() for k,v in payload['training_module'].items()
                            if k in dict(runner.named_parameters()))
    capacity = shared.capacity(runner)
    if parent_parameters != capacity['parameters']:
        raise ValueError('The query update must not introduce trainable parameters')
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out/'config_resolved.yaml')
    RESUME_INFO = dict(parent_checkpoint=str(parent), parent_source_revision=payload['source_revision'],
        control_revision=args.control_revision, model_source_revision=base.source_revision(),
        parent_counters=dict(data_iterations=actual[0], forwards=actual[1], updates=actual[2]),
        target_data_iterations=args.end_step, optimizer_state_entries=len(aux['optimizer'].state),
        learning_rates=loaded_lr, scheduler_epoch=int(aux['scheduler'].last_epoch),
        normalizer_accumulations=normals, configuration_differences=differences,
        all_nonintervention_training_configuration_fields_unchanged=True, original_absolute_course_preserved=True,
        material_response_arm=args.arm, new_losses=0,
        loader_cursor_restored=False, sampling='native TRAIN distribution; restored RNG; new loader pass',
        contact_readout=READOUTS[args.arm], parent_parameters=parent_parameters, capacity=capacity)
    phase.RESUME_INFO = RESUME_INFO
    base.write_json(out/'resume_state.json', dict(state='restored', **RESUME_INFO))
    states = payload['random_states']
    random.setstate(states['python']); np.random.set_state(states['numpy']); torch.set_rng_state(states['torch'])
    if len(states['cuda']) != torch.cuda.device_count():
        raise ValueError('Retain the one-visible-GPU RNG layout')
    torch.cuda.set_rng_state_all(states['cuda'])
    print('RESTORED_CONTINUATION '+json.dumps(RESUME_INFO), flush=True)
    return created, cfg, payload


def save_state(path, runner, aux, cfg, counters, initial, arm, policy):
    from omegaconf import OmegaConf
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(training_module=runner.state_dict(), config=OmegaConf.to_container(cfg),
        architecture_protocol=protocol(arm), global_step=int(counters['data_iterations']),
        optimizer_updates=int(counters['updates']), forwards=int(counters['forwards']),
        counters=dict(counters), initialization=initial, execution_policy=policy,
        source_revision=base.source_revision(), continuation=dict(RESUME_INFO),
        random_states=dict(python=random.getstate(), numpy=np.random.get_state(),
                           torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    payload.update({key:value.state_dict() for key,value in aux.items() if hasattr(value, 'state_dict')})
    temporary = path.with_suffix('.pth.tmp'); torch.save(payload, temporary); temporary.replace(path)


def main():
    global START, START_UPDATES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('resume-check', 'positive-check', 'train'))
    for name in ('project', 'data', 'parent-root', 'face-parent-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--control-revision', required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--resume-step', type=int, choices=(22500,))
    parser.add_argument('--target-step', type=int, choices=(22500, 25000), default=22500)
    parser.add_argument('--max-seconds', type=int, default=MAX_SECONDS)
    args = parser.parse_args()
    if (args.resume_checkpoint is None) != (args.resume_step is None):
        parser.error('Continuation requires both the full checkpoint and absolute resume step')
    if args.resume_checkpoint is not None:
        if args.arm not in ('B', 'C') or args.target_step != 25000:
            parser.error('The approved extra continuation is B/C22500 to25000')
        START = args.resume_step
    elif args.target_step != 22500:
        parser.error('Original screening remains20000 to22500')
    START_UPDATES = expected_counts(START)[2]
    if not 0 < args.max_seconds <= MAX_SECONDS:
        parser.error('Do not exceed the approved four-hour training cap')
    if args.output.resolve() in (args.parent_root.resolve(), args.face_parent_root.resolve()):
        parser.error('Do not overwrite a parent run')
    args.seed, args.iterations = 57, args.target_step
    args.end_step = START+1 if args.mode=='positive-check' else args.target_step
    args.stage_phase = f'material_response_{START}_to_{args.target_step}'
    phase.START, phase.START_UPDATES = START, START_UPDATES
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    shared.install(args)
    if args.mode=='resume-check':
        create(args)
    else:
        phase.train(args, create_fn=create, save_fn=save_state, expected_fn=expected_counts)


if __name__=='__main__':
    main()
