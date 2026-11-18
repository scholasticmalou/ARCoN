#!/usr/bin/env python3
"""Body-contact readout configurations used by the staged training controls."""
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

shared = course.shared
base = course.base
ARMS = ('Fixed-Control', 'Fixed-Gap')
START = 25000
STEPS = 27500
COURSE = 5000
START_UPDATES = course.expected_counts(START)[2]
RESUME_INFO = {}


def expected_counts(iterations=STEPS):
    return course.expected_counts(iterations)


def protocol(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    result = copy.deepcopy(course.protocol('Cell-ScaleFixed'))
    result.update(family='fixed_contact_readout_20261018_paired_continuation_v1',
        arm=arm, parent_arm='Cell-ScaleFixed', from_scratch=False,
        initialization='same_Fixed25000_model_Adam_scheduler_normalizers_rng',
        data_iterations=STEPS, parent_data_iterations=START,
        added_data_iterations=STEPS-START, added_optimizer_updates=12500,
        evaluation_iterations=[STEPS], further_budget_authorized=False,
        sampler='same_TRAIN_new_loader_after_restored_rng_no_cursor_claim',
        loader_cursor_resume=False, expected_forwards=expected_counts()[1],
        expected_optimizer_updates=expected_counts()[2],
        contact_readout='gap_bounded' if arm=='Fixed-Gap' else 'signed_h',
        epsilon_m=0.002, normal_eta='0.5*(1+tanh(existing_normal_logit))',
        normal_support='uniform_nonnegative_per_node_sum1',
        unchanged='material_hinge_tangent_decoder100h_normalizer_loss_course')
    return result


def configuration(args, arm, device, run_dir, *, training=True):
    modules, cfg = course.configuration(args, 'Cell-ScaleFixed', device, run_dir, training=training)
    cfg.experiment.name = 'fixed_contact_readout_20261018'
    cfg.experiment.max_iter = getattr(args, 'end_step', STEPS)
    cfg.experiment.save_checkpoint_every = 2500
    return modules, cfg


def configure_readout(runner, arm):
    response = runner.model._learned_model.response
    response.contact_readout = 'gap_bounded' if arm=='Fixed-Gap' else 'signed_h'
    response.contact_gap_epsilon = 0.002
    return response.contact_readout


def create(args):
    from utils.arguments import create_modules
    from omegaconf import OmegaConf
    global RESUME_INFO
    out = args.output/'trains'/args.arm
    if (out/'history.jsonl').exists() or (out/'train_summary.json').exists():
        raise RuntimeError('Continuation already attempted; preserve its outputs and inspect before recovery')
    parent = args.parent_root/'trains'/'Cell-ScaleFixed'/'checkpoints'/f'step_{START:010d}.pth'
    payload = torch.load(parent, map_location='cpu', weights_only=False)
    actual = (payload['global_step'], payload['forwards'], payload['optimizer_updates'])
    if actual != expected_counts(START) or payload['architecture_protocol'] != course.protocol('Cell-ScaleFixed'):
        raise ValueError('Parent is not the approved original Fixed25k course checkpoint')
    if payload['execution_policy'] != 'outer_only':
        raise ValueError('Preserve the approved outer_only execution policy')
    if not payload['optimizer']['state'] or payload['scheduler']['last_epoch'] != START_UPDATES:
        raise ValueError('Complete native Adam and scheduler state are required')
    calibration = args.parent_root/'calibration.json'
    record = json.loads(calibration.read_text())
    if record['state'] != 'complete' or record['scale'] != 1722.2913807992006:
        raise ValueError('Frozen original TRAIN self calibration differs')
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'calibration.json'
    if target.exists():
        if json.loads(target.read_text()) != record:
            raise ValueError('Existing continuation calibration differs')
    else:
        shutil.copyfile(calibration, target)
    shared.set_seed(57)
    modules, cfg = configuration(args, args.arm, args.device, out)
    created = create_modules(modules, cfg)
    runner, aux = created[2], created[3]
    runner.load_state_dict(payload['training_module'], strict=True)
    core = shared.install(args)
    core.configure_mlp_recomputation(runner.model, False)
    configure_readout(runner, args.arm)
    for name, saved in ((name,payload[name]) for name in ('optimizer','scheduler')):
        aux[name].load_state_dict(saved)
    loaded_lr = [float(g['lr']) for g in aux['optimizer'].param_groups]
    saved_lr = [float(g['lr']) for g in payload['optimizer']['param_groups']]
    if loaded_lr != saved_lr or aux['scheduler'].last_epoch != START_UPDATES:
        raise ValueError('The restored learning-rate schedule differs from the parent')
    expected_norm = float(payload['counters']['normalizer_accumulations'])
    if float(runner.model._output_normalizer._num_accumulations) != expected_norm:
        raise ValueError('The output statistics must continue from the trained parent')
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out/'config_resolved.yaml')
    RESUME_INFO = dict(parent_checkpoint=str(parent), parent_source_revision=payload['source_revision'],
        control_revision=args.control_revision, parent_counters=dict(data_iterations=actual[0],
        forwards=actual[1], updates=actual[2]), target_data_iterations=getattr(args, 'end_step', STEPS),
        optimizer_state_entries=len(aux['optimizer'].state), learning_rates=loaded_lr,
        scheduler_epoch=int(aux['scheduler'].last_epoch), normalizer_accumulations=expected_norm,
        loader_cursor_restored=False, sampling='same native TRAIN distribution, new loader pass',
        model_source_revision=base.source_revision())
    base.write_json(out/'resume_state.json',dict(state='restored', **RESUME_INFO))
    states = payload['random_states']
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    if len(states['cuda']) != torch.cuda.device_count():
        raise ValueError('Restore the same one-visible-GPU RNG configuration as the original run')
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


def train(args):
    created, cfg, payload = create(args)
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
            base.write_json(out/'progress.json', dict(row, state='training', arm=args.arm, phase='Fixed25000_paired_readout_H5_continuation',
                                                   elapsed_seconds=time.monotonic()-start))
        if counters['updates']<=START_UPDATES+5:
            print('CONTINUATION_UPDATE '+json.dumps(row,allow_nan=False), flush=True)
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
    base.write_json(out/'progress.json', dict(counters,state='starting',arm=args.arm,phase='Fixed25000_paired_readout_H5_continuation'))
    step = counters['data_iterations']
    end = args.end_step
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
        summary = dict(counters,state='positive_check_complete' if args.mode=='positive-check' else 'complete',
            arm=args.arm,phase='Fixed25000_paired_readout_H5_continuation',elapsed_seconds=time.monotonic()-start,
            capacity=shared.capacity(runner),source_revision=base.source_revision(),continuation=dict(RESUME_INFO),execution_policy=policy,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    except BaseException as exc:
        summary = dict(counters,state='failed',arm=args.arm,phase='Fixed25000_paired_readout_H5_continuation',
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('resume-check','positive-check','train'))
    for name in ('project','data','parent-root','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--control-revision', required=True)
    p.add_argument('--arm', choices=ARMS, default=ARMS[0])
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    a.seed = 57
    a.iterations = STEPS
    a.end_step = START+1 if a.mode=='positive-check' else STEPS
    if a.output.resolve() == a.parent_root.resolve():
        raise ValueError('The continuation cannot overwrite the original training root')
    torch.set_num_threads(4)
    shared.install(a)
    if a.mode == 'resume-check':
        create(a)
    else:
        train(a)


if __name__ == '__main__':
    main()
