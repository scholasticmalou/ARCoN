#!/usr/bin/env python3
"""Seeded initialization and the initial graph-network training course.

The release driver stops this course at the original transition checkpoint.
Optimizer, scheduler, normalizers and random states are saved together."""
from __future__ import annotations
import argparse
import copy
import gc
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
import arcon.control.initialization as shared

ARMS = ('Cell-Scale', 'Cell-ScaleFixed')
STEPS = 25000
COURSE = 5000
REFERENCE = Path('/workspace/worktrees/cvpr2027-direction1-shared-cell/results/pb4u_shared_cell_20261014')
base = shared.base


def expected_counts(iterations=STEPS):
    forwards = sum(min(1+step//COURSE, 5) for step in range(1, iterations+1))
    return iterations, forwards, forwards-min(iterations, 99)


def protocol(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    result = copy.deepcopy(shared.protocol(arm))
    for key in ('curriculum_boundary_sampler_seed','final_direction_control','calibration_reused_from','fully_curriculum_trained'):
        result.pop(key, None)
    result.update(family='shared_working_cell_20261015_paired_long_v1',
        data_iterations=STEPS, increase_roll_every=COURSE,
        full_course_planned=True,
        maturity_determined_by='saved_checkpoint_quality_and_convergence_not_budget_alone',
        self_activation_step=8000, self_ramp_steps=4000,
        self_strength_reference='original_Cell-F2000_frozen_TRAIN_calibration',
        self_calibration_scale=1722.2913807992006, body_weight=5000.,
        sampler='continuous_seed57_no_2000_restart', loader_cursor_resume=False,
        base_lr=5e-5, lr_multiplier='0.5**floor(actual_optimizer_updates/50000)+0.01',
        save_every_data_iterations=2500, evaluation_iterations=[20000,22500,25000],
        optional_30000_authorized=False,
        expected_forwards=expected_counts()[1], expected_optimizer_updates=expected_counts()[2])
    return result


def configuration(args, arm, device, run_dir, *, training=True):
    modules, cfg = shared.configuration(args, arm, device, run_dir, training=training)
    cfg.experiment.name = 'shared_cell_long_'+arm
    cfg.experiment.max_iter = STEPS
    cfg.experiment.save_checkpoint_every = 2500
    cfg.runner.pb4u.increase_roll_every = COURSE
    cfg.runner.pb4u.warmup_steps = 100
    cfg.criterions['pb4u.collision_penalty'].weight_start = 5000.
    cfg.criterions['pb4u.collision_penalty'].weight_max = 5000.
    if training:
        cfg.criterions[shared.TARGET_KEY].activation_step = 8000
        cfg.criterions[shared.TARGET_KEY].ramp_steps = 4000
    return modules, cfg


def prepare(args):
    from utils.arguments import create_runner
    from omegaconf import OmegaConf
    directory = args.output/'initializations'
    if (directory/'paired_initialization.json').exists():
        raise RuntimeError('Already prepared;do not overwrite paired fresh initialization')
    directory.mkdir(parents=True, exist_ok=True)
    calibration=json.loads((args.reference/'calibration.json').read_text())
    if calibration['state']!='complete' or calibration['scale']!=1722.2913807992006:
        raise ValueError('Reuse the original frozen TRAIN calibration')
    shutil.copyfile(args.reference/'calibration.json',args.output/'calibration.json')
    profile=json.loads((args.output/'execution_profile.json').read_text())
    if profile['state']!='complete':
        raise RuntimeError('Finish the bounded execution profile before formal preparation')
    policy=profile['selected_policy']
    report=dict(state='preparing',seed=57,source_revision=base.source_revision(),
        execution_policy=policy,calibration_reference=str(args.reference/'calibration.json'),arms={})
    common=None
    for arm in ARMS:
        shared.set_seed(57)
        modules,cfg=configuration(args,arm,'cpu',args.output/'trains'/arm)
        _,runner,aux=create_runner(modules,cfg)
        state=runner.state_dict()
        if common is None:common={k:v.detach().clone() for k,v in state.items()}
        runner.load_state_dict(common,strict=True)
        paired=base.compare_values(common,runner.state_dict())
        record=dict(arm=arm,state='complete',paired_common=paired,capacity=shared.capacity(runner),
            optimizer_state_entries=len(aux['optimizer'].state),
            normalizer_accumulations=float(runner.model._output_normalizer._num_accumulations),
            initial_lr=float(aux['optimizer'].param_groups[0]['lr']),
            initialization_role='new_seed57_parameters_no_trained_checkpoint')
        if paired['max_absolute_difference']!=0 or record['optimizer_state_entries']!=0 or record['normalizer_accumulations']!=0:
            raise RuntimeError('Fresh paired state mismatch')
        torch.save(dict(training_module=runner.state_dict(),config=OmegaConf.to_container(cfg),
            architecture_protocol=protocol(arm),global_step=0,optimizer_updates=0,forwards=0,
            initialization=record,execution_policy=policy,seed=57),directory/(arm+'.pth'))
        report['arms'][arm]=record
        print('INITIALIZED '+json.dumps(record),flush=True)
        del runner,aux;gc.collect()
    report['state']='complete'
    base.write_json(directory/'paired_initialization.json',report)


def create(args):
    from utils.arguments import create_modules
    from omegaconf import OmegaConf
    shared.set_seed(57)
    out=args.output/'trains'/args.arm
    modules,cfg=configuration(args,args.arm,args.device,out)
    created=create_modules(modules,cfg)
    payload=torch.load(args.output/'initializations'/(args.arm+'.pth'),map_location='cpu',weights_only=False)
    if payload['architecture_protocol']!=protocol(args.arm) or payload['global_step']!=0:
        raise ValueError('This entry starts only from the approved fresh initialization')
    created[2].load_state_dict(payload['training_module'],strict=True)
    core=shared.install(args)
    count=core.configure_mlp_recomputation(created[2].model,payload['execution_policy']=='nested')
    payload['initialization']['runtime_mlp_count']=count
    if created[3]['optimizer'].state:
        raise RuntimeError('Formal optimizer must start fresh')
    out.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,out/'config_resolved.yaml')
    shared.set_seed(57)
    return created,cfg,payload


def save_state(path, runner, aux, cfg, counters, initial, arm, policy):
    from omegaconf import OmegaConf
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(training_module=runner.state_dict(),config=OmegaConf.to_container(cfg),
        architecture_protocol=protocol(arm),global_step=int(counters['data_iterations']),
        optimizer_updates=int(counters['updates']),forwards=int(counters['forwards']),
        counters=dict(counters),initialization=initial,execution_policy=policy,
        source_revision=base.source_revision(),random_states=dict(python=random.getstate(),
            numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    payload.update({key:value.state_dict() for key,value in aux.items() if hasattr(value,'state_dict')})
    temporary=path.with_suffix('.pth.tmp');torch.save(payload,temporary);temporary.replace(path)


def train(args):
    created, cfg, payload = create(args)
    loader, runner_module, runner, aux = created
    out = args.output/'trains'/args.arm
    initial = payload['initialization']
    policy = payload['execution_policy']
    counters = dict(data_iterations=int(payload['global_step']), forwards=int(payload.get('forwards', 0)),
                    updates=int(payload.get('optimizer_updates', 0)))
    del payload
    history = (out/'history.jsonl').open('w', buffering=1)
    start = time.monotonic()
    last_state = [None]
    def pre_runner(_, inputs):
        if time.monotonic()-start > args.max_seconds:raise TimeoutError('Training stage wall-time limit reached')
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
                  (counters['updates'] < 10 or counters['updates']%1000==0))
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
        observed = shared.gradients(self) if counters['updates'] < 10 or counters['updates']%1000==0 else None
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
        if counters['updates']<=10 or counters['updates']%100==0:
            base.write_json(out/'progress.json', dict(row, state='training', arm=args.arm, phase='continuous_full_course',
                                                   elapsed_seconds=time.monotonic()-start))
        if counters['updates']<=5:
            print('FORMAL_UPDATE '+json.dumps(row,allow_nan=False), flush=True)
    hooks = [runner.register_forward_pre_hook(pre_runner),
        runner.model.register_forward_pre_hook(pre_model, with_kwargs=True),
        runner.model.register_forward_hook(post_model)]
    runner.optimizer_step = types.MethodType(optimize, runner)
    def save(module, auxiliary, config, filename):
        save_state(filename, module, auxiliary, config, counters, initial, args.arm, policy)
    runner_module.save_checkpoint = save
    def stop(signum, frame):
        raise InterruptedError('Declared training stage stopped by signal '+str(signum))
    signal.signal(signal.SIGTERM, stop)
    torch.cuda.reset_peak_memory_stats()
    base.write_json(out/'progress.json', dict(counters,state='starting',arm=args.arm,phase='continuous_full_course'))
    step = counters['data_iterations']
    end = getattr(args, 'end_step', STEPS)
    try:
        for epoch in range(int(cfg.experiment.n_epochs)):
            step = runner_module.run_epoch(runner, aux, loader.create_dataloader(), epoch, cfg, global_step=step)
            step = counters['data_iterations']
            if step >= end:
                break
        expected = expected_counts(end)
        actual = (counters['data_iterations'],counters['forwards'],counters['updates'])
        if actual != expected:
            raise RuntimeError({'actual':actual,'expected':expected})
        save_state(out/'checkpoints'/f'step_{end:010d}.pth',runner,aux,cfg,counters,initial,args.arm,policy)
        summary = dict(counters,state='complete',
            arm=args.arm,phase='continuous_full_course',elapsed_seconds=time.monotonic()-start,
            capacity=shared.capacity(runner),source_revision=base.source_revision(),execution_policy=policy,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    except BaseException as exc:
        summary = dict(counters,state='failed',arm=args.arm,phase='continuous_full_course',
            failure=type(exc).__name__+': '+str(exc),elapsed_seconds=time.monotonic()-start,
            source_revision=base.source_revision())
        try:
            save_state(out/'failure_full_course.pth',runner,aux,cfg,counters,initial,args.arm,policy)
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
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','train'))
    for name in ('project','data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference',type=Path,default=REFERENCE)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();a.seed=57;a.iterations=STEPS;torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=True);shared.install(a)
    if a.mode=='prepare':prepare(a)
    else:train(a)

if __name__=='__main__':main()
