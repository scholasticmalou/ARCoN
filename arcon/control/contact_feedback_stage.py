#!/usr/bin/env python3
"""Training control for the surface-contact phase and later staged updates.

This module retains the physical-step training loop, loss evaluation, gradient
clipping and checkpoint state used by the reported model."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import signal
import time
import types
import numpy as np
import torch
import arcon.control.backbone_stage as course
import arcon.control.contact_support_stage as original_screen

shared = course.shared
base = course.base
ARMS = ('Face-Early', 'Face-Late')
START_STEPS = {'Face-Early': 15000, 'Face-Late': 20000}
EVALUATION_STEPS = {'Face-Early': [17500, 20000, 22500, 25000], 'Face-Late': [22500, 25000]}
TRAINING_CAPS = {'Face-Early': 14*3600, 'Face-Late': 8*3600}
READOUTS = {arm: 'face_floor' for arm in ARMS}
START = None
STEPS = 25000
COURSE = 5000
START_UPDATES = None
RESUME_INFO = {}


def expected_counts(iterations=STEPS):
    return course.expected_counts(iterations)


def protocol(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    start = START_STEPS[arm]
    result = copy.deepcopy(course.protocol('Cell-ScaleFixed'))
    result.update(family='facefloor_phase_courses_20261018_v1', arm=arm,
        parent_arm='Coarse-FaceFloor' if arm=='Face-Early' else 'Cell-ScaleFixed',
        from_scratch=False, initialization='full_model_Adam_scheduler_normalizers_rng_parent',
        parent_data_iterations=start, data_iterations=STEPS,
        floor_activation_iteration=12500 if arm=='Face-Early' else 20000,
        added_data_iterations=STEPS-start,
        added_optimizer_updates=expected_counts(STEPS)[2]-expected_counts(start)[2],
        evaluation_iterations=EVALUATION_STEPS[arm], further_budget_authorized=False,
        sampler='original_TRAIN_distribution_new_loader_after_restored_rng_no_cursor_claim',
        loader_cursor_resume=False,
        expected_forwards=expected_counts()[1], expected_optimizer_updates=expected_counts()[2],
        contact_readout='face_floor', epsilon_m=0.002,
        floor='original_external_plus_mass_times_positive_selected_plane_deficit',
        face_correspondence='native_current_cloth_to_current_body_face_centroid_then_target_face_plane',
        normal_support='original_world_edge_cloth_nodes_only', new_parameters=0,
        training_walltime_limit_seconds=TRAINING_CAPS[arm], pipeline_walltime_limit_seconds=20*3600,
        unchanged='absolute_C5000_H1toH5_loss_self_Adam_LR_normalizers_material_hinge_normal_tangent_decoder100h')
    return result


def configuration(args, arm, device, run_dir, *, training=True):
    modules, cfg = course.configuration(args, 'Cell-ScaleFixed', device, run_dir, training=training)
    cfg.experiment.name = 'facefloor_phase_course_20261018_'+arm
    cfg.experiment.max_iter = getattr(args, 'end_step', STEPS)
    cfg.experiment.save_checkpoint_every = 2500
    return modules, cfg


def configure_readout(runner, arm):
    response = runner.model._learned_model.response
    response.contact_readout = READOUTS[arm]
    response.contact_gap_epsilon = 0.002
    return response.contact_readout


def configuration_differences(left, right, prefix=()):
    differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append(dict(path=list(prefix+(key,)), parent=left.get(key), continued=right.get(key)))
            else:
                differences.extend(configuration_differences(left[key], right[key], prefix+(key,)))
    elif left != right:
        differences.append(dict(path=list(prefix), parent=left, continued=right))
    return differences


def create(args):
    from utils.arguments import create_modules
    from omegaconf import OmegaConf
    global RESUME_INFO
    out = args.output/'trains'/args.arm
    if (out/'history.jsonl').exists() or (out/'train_summary.json').exists():
        raise RuntimeError('Continuation already attempted; preserve its outputs')
    if args.resume_checkpoint is not None:
        parent = args.resume_checkpoint
        expected_protocol = protocol(args.arm)
    elif args.arm == 'Face-Early':
        parent = args.face_parent_root/'trains'/'Coarse-FaceFloor'/'checkpoints'/f'step_{START:010d}.pth'
        expected_protocol = original_screen.protocol('Coarse-FaceFloor')
    else:
        parent = args.parent_root/'trains'/'Cell-ScaleFixed'/'checkpoints'/f'step_{START:010d}.pth'
        expected_protocol = course.protocol('Cell-ScaleFixed')
    parent = getattr(args, 'parent_checkpoint', None) or parent
    payload = torch.load(parent, map_location='cpu', weights_only=False)
    from arcon.compat import relocate_training_config
    relocate_training_config(payload, args.data, args.parent_root/'calibration.json')
    actual = (payload['global_step'], payload['forwards'], payload['optimizer_updates'])
    if actual != expected_counts(START) or payload['architecture_protocol'] != expected_protocol:
        raise ValueError('Parent differs from the approved complete phase-course checkpoint')
    if payload['execution_policy'] != 'outer_only':
        raise ValueError('Preserve original outer_only policy')
    if not payload['optimizer']['state'] or payload['scheduler']['last_epoch'] != START_UPDATES:
        raise ValueError('Complete Adam/scheduler age is required')
    calibration = args.parent_root/'calibration.json'
    record = json.loads(calibration.read_text())
    previous_calibration = Path(payload['config']['criterions'][shared.TARGET_KEY]['calibration_file'])
    if (record['state'] != 'complete' or record['scale'] != 1722.2913807992006
            or json.loads(previous_calibration.read_text()) != record):
        raise ValueError('Preserve the exact original TRAIN self calibration')
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'calibration.json'
    if target.exists():
        if json.loads(target.read_text()) != record:
            raise ValueError('Existing calibration differs')
    else:
        shutil.copyfile(calibration, target)
    shared.set_seed(57)
    modules, _ = configuration(args, args.arm, args.device, out)
    # Reuse the saved config itself, not a new guessed training recipe.
    cfg = OmegaConf.create(copy.deepcopy(payload['config']))
    cfg.device = args.device
    cfg.run_dir = str(out)
    cfg.experiment.name = 'facefloor_phase_course_20261018_'+args.arm
    cfg.experiment.max_iter = args.end_step
    cfg.criterions[shared.TARGET_KEY].calibration_file = str(target)
    differences = configuration_differences(payload['config'], OmegaConf.to_container(cfg))
    allowed = {('device',), ('run_dir',), ('experiment','name'), ('experiment','max_iter'),
               ('criterions',shared.TARGET_KEY,'calibration_file')}
    if any(tuple(row['path']) not in allowed for row in differences):
        raise ValueError('A non-run-control parent configuration field changed: '+str(differences))
    created = create_modules(modules, cfg)
    runner, aux = created[2], created[3]
    runner.load_state_dict(payload['training_module'], strict=True)
    core = shared.install(args)
    core.configure_mlp_recomputation(runner.model, False)
    configure_readout(runner, args.arm)
    for name in ('optimizer','scheduler'):
        aux[name].load_state_dict(payload[name])
    loaded_lr = [float(g['lr']) for g in aux['optimizer'].param_groups]
    saved_lr = [float(g['lr']) for g in payload['optimizer']['param_groups']]
    if loaded_lr != saved_lr or aux['scheduler'].last_epoch != START_UPDATES:
        raise ValueError('The restored original learning-rate schedule differs')
    normals = {name:float(value) for name,value in runner.state_dict().items()
               if name.endswith('_num_accumulations')}
    saved_normals = {name:float(value) for name,value in payload['training_module'].items()
                     if name.endswith('_num_accumulations')}
    if normals != saved_normals:
        raise ValueError('A parent normalizer statistic changed')
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out/'config_resolved.yaml')
    RESUME_INFO = dict(parent_checkpoint=str(parent), parent_source_revision=payload['source_revision'],
        control_revision=args.control_revision, parent_counters=dict(data_iterations=actual[0],
        forwards=actual[1], updates=actual[2]), target_data_iterations=args.end_step,
        optimizer_state_entries=len(aux['optimizer'].state), learning_rates=loaded_lr,
        scheduler_epoch=int(aux['scheduler'].last_epoch), normalizer_accumulations=normals,
        configuration_differences=differences, all_training_configuration_fields_unchanged=True,
        original_absolute_course_preserved=True,
        loader_cursor_restored=False, sampling='original native TRAIN distribution, restored RNG, new loader pass',
        model_source_revision=base.source_revision())
    base.write_json(out/'resume_state.json', dict(state='restored', **RESUME_INFO))
    states = payload['random_states']
    random.setstate(states['python']); np.random.set_state(states['numpy']); torch.set_rng_state(states['torch'])
    if len(states['cuda']) != torch.cuda.device_count():
        raise ValueError('Restore the same one-visible-GPU RNG configuration')
    torch.cuda.set_rng_state_all(states['cuda'])
    print('RESTORED_CONTINUATION '+json.dumps(RESUME_INFO), flush=True)
    return created, cfg, payload


def save_state(path, runner, aux, cfg, counters, initial, arm, policy):
    from omegaconf import OmegaConf
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(training_module=runner.state_dict(),config=OmegaConf.to_container(cfg),
        architecture_protocol=protocol(arm),global_step=int(counters['data_iterations']),
        optimizer_updates=int(counters['updates']),forwards=int(counters['forwards']),
        counters=dict(counters),initialization=initial,execution_policy=policy,
        source_revision=base.source_revision(),continuation=dict(RESUME_INFO),random_states=dict(python=random.getstate(),
            numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    payload.update({key:value.state_dict() for key,value in aux.items() if hasattr(value,'state_dict')})
    temporary=path.with_suffix('.pth.tmp');torch.save(payload,temporary);temporary.replace(path)


def train(args, *, create_fn=None, save_fn=None, expected_fn=None):
    created, cfg, payload = (create_fn or create)(args)
    save_checkpoint = save_fn or save_state
    count_expected = expected_fn or expected_counts
    stage_phase = getattr(args, 'stage_phase', 'facefloor_phase_course_to25000')
    loader, runner_module, runner, aux = created
    out = args.output/'trains'/args.arm
    initial = payload['initialization']
    policy = payload['execution_policy']
    counters = dict(data_iterations=int(payload['global_step']), forwards=int(payload.get('forwards', 0)),
                    updates=int(payload.get('optimizer_updates', 0)))
    del payload
    history = (out/'history.jsonl').open('x', buffering=1)
    start = time.monotonic()
    last_state = [None]
    def pre_runner(_, inputs):
        sample = inputs[0]
        step = int(torch.as_tensor(sample['cloth'].iter).reshape(-1)[0])
        counters.update(data_iterations=step, iteration=step,
            sequence_name=sample.get('sequence_name'), garment_name=sample.get('garment_name'))
    def pre_model(_, inputs, kwargs):
        counters['forwards'] += 1
        cloth = inputs[0]['cloth']
        counters.update(K=int(inputs[1] if len(inputs)>1 else kwargs['repeat']),
            path_ids=cloth['path_id'].detach().cpu().reshape(-1).tolist() if 'path_id' in cloth else None,
            tau=float(cloth.timestep.reshape(-1)[0]),
            material={key:float(cloth[key].reshape(-1)[0]) for key in
                      ('density','lame_mu','lame_lambda','bending_coeff') if key in cloth},
            sampled_density=float(runner.cloth_obj.material.density.reshape(-1)[0]),
            normalizer_before=float(runner.model._output_normalizer._num_accumulations),
            rollout_steps=min(1+counters['data_iterations']//COURSE, 5))
        body = inputs[0]['obstacle'].target_pos
        counters['body_input_coordinates'] = body[[0,len(body)//2,len(body)-1]].detach().cpu().tolist()
        shared.recording(runner.model, counters['data_iterations'] >= 100 and
                  (counters['updates'] < START_UPDATES+5 or counters['updates']%1000==0))
    def post_model(_, inputs, output):
        after = float(runner.model._output_normalizer._num_accumulations)
        if after-counters['normalizer_before'] != 1:
            raise RuntimeError('The sole output-statistics accumulation changed')
        counters['normalizer_accumulations'] = after
        last_state[0] = output
    def optimize(self, loss_dict, optimizer=None, scheduler=None):
        if optimizer is None:
            return
        scalar = {key:float(value.detach()) for key,value in loss_dict.items()}
        if not all(math.isfinite(value) for value in scalar.values()):
            raise FloatingPointError('Nonfinite real training loss')
        optimizer.zero_grad(set_to_none=True)
        sum(loss_dict.values()).backward()
        norms = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.mcfg.grad_clip,
                                               error_if_nonfinite=True)
        observed = shared.gradients(self) if counters['updates'] < START_UPDATES+5 or counters['updates']%1000==0 else None
        if time.monotonic()-start > args.max_seconds:
            raise TimeoutError('Declared per-arm continuation walltime exhausted; no retry authorized')
        optimizer.step()
        scheduler.step()
        counters['updates'] += 1
        target = self.criterion_dict['self_intersection_energy']
        row = dict(counters, losses=scalar, preclip_gradient_norm=float(norms),
            lr=float(optimizer.param_groups[0]['lr']), scheduler_epoch=int(scheduler.last_epoch),
            self_target=target.last, gradients=observed)
        if observed is not None:
            row['execution'] = self.model._learned_model.execution
        history.write(json.dumps(row, allow_nan=False)+'\n')
        if counters['updates']<=START_UPDATES+5 or counters['updates']%100==0:
            base.write_json(out/'progress.json', dict(row, state='training', arm=args.arm, phase=stage_phase,
                                                   elapsed_seconds=time.monotonic()-start))
        if counters['updates']<=START_UPDATES+5:
            print('CONTINUATION_UPDATE '+json.dumps(row,allow_nan=False), flush=True)
    hooks = [runner.register_forward_pre_hook(pre_runner),
        runner.model.register_forward_pre_hook(pre_model, with_kwargs=True),
        runner.model.register_forward_hook(post_model)]
    runner.optimizer_step = types.MethodType(optimize, runner)
    def save(module, auxiliary, config, filename):
        save_checkpoint(filename, module, auxiliary, config, counters, initial, args.arm, policy)
    runner_module.save_checkpoint = save
    def stop(signum, frame):
        raise InterruptedError('Declared training stage stopped by signal '+str(signum))
    signal.signal(signal.SIGTERM, stop)
    torch.cuda.reset_peak_memory_stats()
    base.write_json(out/'progress.json', dict(counters,state='starting',arm=args.arm,phase=stage_phase))
    step = counters['data_iterations']
    end = args.end_step
    try:
        for epoch in range(int(cfg.experiment.n_epochs)):
            step = runner_module.run_epoch(runner, aux, loader.create_dataloader(), epoch, cfg, global_step=step)
            step = counters['data_iterations']
            if step >= end:
                break
        expected = count_expected(end)
        actual = (counters['data_iterations'],counters['forwards'],counters['updates'])
        if actual != expected:
            raise RuntimeError({'actual':actual,'expected':expected})
        save_checkpoint(out/'checkpoints'/f'step_{end:010d}.pth',runner,aux,cfg,counters,initial,args.arm,policy)
        summary = dict(counters,state='positive_check_complete' if args.mode=='positive-check' else 'complete',
            arm=args.arm,phase=stage_phase,elapsed_seconds=time.monotonic()-start,
            capacity=shared.capacity(runner),source_revision=base.source_revision(),continuation=dict(RESUME_INFO),execution_policy=policy,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    except BaseException as exc:
        summary = dict(counters,state='failed',arm=args.arm,phase=stage_phase,
            failure=type(exc).__name__+': '+str(exc),elapsed_seconds=time.monotonic()-start,
            source_revision=base.source_revision())
        try:
            save_checkpoint(out/'failure_full_course.pth',runner,aux,cfg,counters,initial,args.arm,policy)
            if last_state[0] is not None:
                torch.save(last_state[0].cpu(),out/'failure_full_course_actual_state.pth')
        except Exception as secondary:
            summary['failure_save_error'] = str(secondary)
        base.write_json(out/'train_summary.json',summary)
        base.write_json(out/'progress.json',summary)
        raise
    else:
        base.write_json(out/'train_summary.json',summary)
        base.write_json(out/'progress.json',summary)
        print('STAGE_COMPLETE '+json.dumps(summary),flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        history.close()



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('resume-check','positive-check','train'))
    for name in ('project','data','parent-root','face-parent-root','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--control-revision', required=True)
    p.add_argument('--arm', choices=ARMS, default=ARMS[0])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seconds', type=int)
    p.add_argument('--resume-checkpoint', type=Path)
    p.add_argument('--resume-step', type=int)
    global START, START_UPDATES
    a = p.parse_args()
    if (a.resume_checkpoint is None) != (a.resume_step is None):
        p.error('Explicit complete recovery requires both checkpoint and data step')
    START = a.resume_step if a.resume_step is not None else START_STEPS[a.arm]
    if not START_STEPS[a.arm] <= START < STEPS:
        p.error('Recovery must remain inside the original approved course')
    START_UPDATES = expected_counts(START)[2]
    a.max_seconds = a.max_seconds or TRAINING_CAPS[a.arm]
    if not 0 < a.max_seconds <= TRAINING_CAPS[a.arm]:
        p.error('Do not exceed the approved per-arm continuation cap')
    a.seed = 57
    a.iterations = STEPS
    a.end_step = START+1 if a.mode=='positive-check' else STEPS
    if a.output.resolve() in (a.parent_root.resolve(), a.face_parent_root.resolve()):
        raise ValueError('The continuation cannot overwrite the original training root')
    torch.set_num_threads(4)
    shared.install(a)
    if a.mode == 'resume-check':
        create(a)
    else:
        train(a)


if __name__ == '__main__':
    main()
