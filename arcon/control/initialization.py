#!/usr/bin/env python3
"""Graph-network construction, scheduled self-contact supervision and calibration.

The paper course reuses the supplied fixed training calibration. An optional
calibration routine retains the original gradient-based scale calculation."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import gc
import importlib.util
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import types

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import arcon.control.graph_setup as base
from arcon.control.reproducibility import set_seed
from arcon.models import self_supervision as targets

ARMS = ('Cell-F', 'Cell-Dir', 'Cell-Scale', 'Cell-Joint', 'Cell-Fixed')
FOLLOWUP_ARMS = ('Cell-ScaleFixed', 'Cell-NetDir')
ALL_ARMS = ARMS+FOLLOWUP_ARMS
STEPS = 7500
TARGET_KEY = 'pb4u.mataug.shared_cell_self_energy'


@dataclass
class SelfConfig:
    calibration_file: str = ''
    activation_step: int = 2000
    ramp_steps: int = 1000


class ScheduledSelf(torch.nn.Module):
    """A common newly calibrated D, disabled during from-zero geometry warmup."""
    def __init__(self, config):
        super().__init__()
        self.name = 'self_intersection_energy'
        self.path = Path(config.calibration_file)
        self.start = int(config.activation_step)
        self.ramp = int(config.ramp_steps)
        self.target = targets.Criterion(targets.Config(family='D', scale=1.))
        self.scale = None
        self.last = None

    def forward(self, sample):
        step = int(torch.as_tensor(sample['cloth'].iter).reshape(-1)[0])
        ratio = min(1., max(0., (step-self.start)/self.ramp))
        if ratio == 0:
            self.last = dict(enabled=False, ramp=0., data_step=step, scale=None)
            return dict(loss=sample['cloth'].pred_pos.sum()*0.)
        if self.scale is None:
            entry = json.loads(self.path.read_text())
            if entry.get('state') != 'complete' or entry['scale'] <= 0:
                raise ValueError('A positive complete fresh TRAIN calibration is required')
            self.scale = float(entry['scale'])
        loss = self.target(sample)['loss']
        self.last = dict(self.target.last, enabled=True, ramp=ratio, data_step=step,
                         scale=self.scale, actual_weight=ratio*self.scale)
        return dict(loss=loss*(ratio*self.scale))


def install(args):
    sys.path[:] = [x for x in sys.path if x not in ('', str(ROOT))]
    base.install(args.project, args.data)
    key = 'models.core.adaptive_cell'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT/'arcon/models/adaptive_cell.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    from arcon.compat import register_legacy_core
    register_legacy_core(sys.modules[key])
    return sys.modules[key]


def protocol(arm, iterations=STEPS):
    if arm not in ALL_ARMS or iterations != STEPS:
        raise ValueError((arm, iterations))
    result = dict(family='shared_working_cell_20261014_short_v1', arm=arm,
        from_scratch=True, seed=57, data_iterations=7500, warmup_steps=100,
        roll_init=1, roll_max=5, increase_roll_every=500,
        fully_curriculum_trained=False, author_curriculum=False,
        shared_cells=3, unique_typed_blocks=5, executed_typed_blocks=15, latent_width=128,
        old_native_predictor=False, independent_response_processors=0, global_tokens=0,
        direction='learned' if arm in ('Cell-Dir', 'Cell-Joint', 'Cell-NetDir') else 'fixed_0.1' if arm in ('Cell-Fixed','Cell-ScaleFixed') else 'none',
        scale='fine_residual_patch_0.06m' if arm in ('Cell-Scale', 'Cell-Joint', 'Cell-ScaleFixed', 'Cell-NetDir') else 'fine',
        self_source='nonlocal_h2_eligible10', self_queries_per_frame=3, self_messages_per_frame=6,
        self_target='D_detached_local_separation', target_in_inference=False,
        self_activation_step=2000, self_ramp_steps=1000,
        self_strength_reference='fresh_Cell-F2000_TRAIN_stretch_plus_bend_q_gradient',
        curriculum_boundary_sampler_seed=58, loader_cursor_resume=False,
        initialization='fresh_parameters_optimizer_scheduler_normalizers',
        output_statistics_accumulations_per_frame=1,
        expected_forwards=base.expected_counts(STEPS)[1],
        expected_optimizer_updates=base.expected_counts(STEPS)[2])
    if arm in FOLLOWUP_ARMS:
        result.update(direction_scope='complete_candidate' if arm=='Cell-NetDir' else 'decoder_innovation',
            relation_feedback_scope='candidate_J' if arm=='Cell-NetDir' else 'applied_J',
            calibration_reused_from='original_Cell-F2000_frozen_TRAIN_calibration',
            final_direction_control=True)
    return result


def configuration(args, arm, device, run_dir, *, training=True):
    core = install(args)
    construction = copy.copy(args)
    construction.iterations = STEPS
    modules, cfg = base.configuration(construction, 'N', device, run_dir)
    from omegaconf import open_dict, OmegaConf
    import models.pb4u as wrapper
    wrapper.Model = core.ARCoNModel
    with open_dict(cfg.model.pb4u):
        cfg.model.pb4u.core_model = 'adaptive_cell'
        cfg.model.pb4u.cell_arm = arm
        cfg.model.pb4u.cell_patch_radius = .06
    cfg.experiment.name = 'shared_cell_short_'+arm
    cfg.experiment.max_iter = STEPS
    cfg.experiment.save_checkpoint_every = 500
    cfg.runner.pb4u.increase_roll_every = 500
    if training:
        modules['criterions'][TARGET_KEY] = types.SimpleNamespace(
            Config=SelfConfig, create=lambda conf: ScheduledSelf(conf))
        with open_dict(cfg.criterions):
            cfg.criterions[TARGET_KEY] = OmegaConf.structured(SelfConfig(
                calibration_file=str((args.output/'calibration.json').resolve())))
    return modules, cfg


def capacity(runner):
    return dict(parameters=sum(p.numel() for p in runner.parameters()),
                trainable_parameters=sum(p.numel() for p in runner.parameters() if p.requires_grad),
                modules=[name for name, _ in runner.model._learned_model.named_children()])


def prepare(args):
    from utils.arguments import create_runner
    from omegaconf import OmegaConf
    directory = args.output/'initializations'
    directory.mkdir(parents=True, exist_ok=True)
    common, direction = None, None
    report = dict(state='preparing', seed=57, source_revision=base.source_revision(), arms={})
    for arm in ARMS:
        set_seed(57)
        modules, cfg = configuration(args, arm, 'cpu', args.output/'trains'/arm)
        _, runner, aux = create_runner(modules, cfg)
        state = runner.state_dict()
        if common is None:
            common = {key: value.detach().clone() for key, value in state.items()}
        for key, value in common.items():
            if key not in state or state[key].shape != value.shape:
                raise ValueError('Common shared-cell parameter shape changed: '+key)
            state[key] = value.clone()
        gain = {key: value for key, value in state.items() if '.direction.gain.' in key}
        if gain and direction is None:
            direction = {key: value.clone() for key, value in gain.items()}
        if gain:
            state.update(direction)
        runner.load_state_dict(state, strict=True)
        paired = base.compare_values(common, runner.state_dict())
        if paired['max_absolute_difference'] != 0 or aux['optimizer'].state:
            raise ValueError('Fresh paired initialization failed')
        record = dict(arm=arm, state='complete', paired_common=paired, capacity=capacity(runner),
            optimizer_state_entries=len(aux['optimizer'].state),
            normalizer_accumulations=float(runner.model._output_normalizer._num_accumulations),
            initial_lr=float(aux['optimizer'].param_groups[0]['lr']))
        torch.save(dict(training_module=runner.state_dict(), config=OmegaConf.to_container(cfg),
            architecture_protocol=protocol(arm), global_step=0, optimizer_updates=0,
            forwards=0, initialization=record, seed=57), directory/(arm+'.pth'))
        report['arms'][arm] = record
        print('INITIALIZED '+arm+' '+json.dumps(record), flush=True)
        del runner, aux
        gc.collect()
    report['state'] = 'complete'
    base.write_json(directory/'paired_initialization.json', report)


def create(args, arm, *, phase=None, aux=True):
    from utils.arguments import create_modules, create_runner
    from omegaconf import OmegaConf
    set_seed(57)
    out = args.output/'trains'/arm
    modules, cfg = configuration(args, arm, args.device, out)
    if phase == 'first':
        cfg.experiment.max_iter = 2000
    created = create_modules(modules, cfg) if aux else create_runner(modules, cfg, create_aux_modules=False)
    runner = created[2] if aux else created[1]
    source = (out/'checkpoints/step_0000002000.pth' if phase=='second' else
              args.output/'initializations'/(arm+'.pth'))
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if payload['architecture_protocol'] != protocol(arm):
        raise ValueError('Shared-cell checkpoint protocol changed')
    runner.load_state_dict(payload['training_module'], strict=True)
    if aux and phase == 'second':
        for key in ('optimizer', 'scheduler'):
            created[3][key].load_state_dict(payload[key])
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out/'config_resolved.yaml')
    # The declared curriculum boundary restarts the common sampler, not model,
    # optimizer, scheduler or normalizer states. Do not imply loader-cursor replay.
    set_seed(58 if phase=='second' else 57)
    return created, cfg, payload


def gradients(runner):
    prefixes = ('node_encoder', 'decoder', 'blocks', 'transport.message', 'transport.output',
                'self_information.encoder', 'self_information.message', 'self_information.output',
                'stage_feedback', 'response.edge_head', 'response.hinge_head',
                'response.contact_head', 'direction.gain')
    output = {}
    for prefix in prefixes:
        values = [p.grad.detach().abs().max() for name, p in runner.model._learned_model.named_parameters()
                  if name.startswith(prefix) and p.grad is not None]
        if values:
            output[prefix] = float(torch.stack(values).max())
    return output


def recording(model, enabled):
    core = model._learned_model
    core.record = enabled
    core.response.record = enabled
    core.direction.record = enabled
    core.stage_feedback.record = enabled


def save_state(path, runner, aux, cfg, counters, initial, arm):
    from omegaconf import OmegaConf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(training_module=runner.state_dict(), config=OmegaConf.to_container(cfg),
        architecture_protocol=protocol(arm), global_step=int(counters['data_iterations']),
        optimizer_updates=int(counters['updates']), forwards=int(counters['forwards']),
        counters=dict(counters), initialization=initial, source_revision=base.source_revision(),
        random_states=dict(python=random.getstate(), numpy=np.random.get_state(),
                           torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    payload.update({key: value.state_dict() for key, value in aux.items() if hasattr(value, 'state_dict')})
    torch.save(payload, path)


def train(args):
    created, cfg, payload = create(args, args.arm, phase=args.phase)
    loader, runner_module, runner, aux = created
    out = args.output/'trains'/args.arm
    initial = payload['initialization']
    counters = dict(data_iterations=int(payload['global_step']), forwards=int(payload.get('forwards', 0)),
                    updates=int(payload.get('optimizer_updates', 0)))
    if args.phase == 'second':
        calibration = json.loads((args.output/'calibration.json').read_text())
        if calibration['state'] != 'complete':
            raise ValueError('Second phase requires the fresh common calibration')
    del payload
    history = (out/'history.jsonl').open('a' if args.phase=='second' else 'w', buffering=1)
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
            rollout_steps=min(1+counters['data_iterations']//500, 5))
        body = inputs[0]['obstacle'].target_pos
        counters['body_input_coordinates'] = body[[0,len(body)//2,len(body)-1]].detach().cpu().tolist()
        recording(runner.model, counters['data_iterations'] >= 100 and
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
        observed = gradients(self) if counters['updates'] < 10 or counters['updates']%1000==0 else None
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
            base.write_json(out/'progress.json', dict(row, state='training', arm=args.arm, phase=args.phase,
                                                   elapsed_seconds=time.monotonic()-start))
        if counters['updates']<=5:
            print('FORMAL_UPDATE '+json.dumps(row,allow_nan=False), flush=True)
    hooks = [runner.register_forward_pre_hook(pre_runner),
        runner.model.register_forward_pre_hook(pre_model, with_kwargs=True),
        runner.model.register_forward_hook(post_model)]
    runner.optimizer_step = types.MethodType(optimize, runner)
    def save(module, auxiliary, config, filename):
        save_state(filename, module, auxiliary, config, counters, initial, args.arm)
    runner_module.save_checkpoint = save
    def stop(signum, frame):
        raise InterruptedError('Declared training stage stopped by signal '+str(signum))
    signal.signal(signal.SIGTERM, stop)
    torch.cuda.reset_peak_memory_stats()
    base.write_json(out/'progress.json', dict(counters,state='starting',arm=args.arm,phase=args.phase))
    step = counters['data_iterations']
    end = 2000 if args.phase=='first' else STEPS
    try:
        for epoch in range(int(cfg.experiment.n_epochs)):
            step = runner_module.run_epoch(runner, aux, loader.create_dataloader(), epoch, cfg, global_step=step)
            if step >= end:
                break
        expected = base.expected_counts(end)
        actual = (counters['data_iterations'],counters['forwards'],counters['updates'])
        if actual != expected:
            raise RuntimeError({'actual':actual,'expected':expected})
        save_state(out/'checkpoints'/f'step_{end:010d}.pth',runner,aux,cfg,counters,initial,args.arm)
        summary = dict(counters,state='stage_complete' if args.phase=='first' else 'complete',
            arm=args.arm,phase=args.phase,elapsed_seconds=time.monotonic()-start,
            capacity=capacity(runner),source_revision=base.source_revision(),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    except BaseException as exc:
        summary = dict(counters,state='failed',arm=args.arm,phase=args.phase,
            failure=type(exc).__name__+': '+str(exc),elapsed_seconds=time.monotonic()-start,
            source_revision=base.source_revision())
        try:
            save_state(out/('failure_'+args.phase+'.pth'),runner,aux,cfg,counters,initial,args.arm)
            if last_state[0] is not None:
                torch.save(last_state[0].cpu(),out/('failure_'+args.phase+'_actual_state.pth'))
        except Exception as secondary:
            summary['failure_save_error'] = str(secondary)
        base.write_json(out/('stage_'+args.phase+'_summary.json'),summary)
        base.write_json(out/'progress.json',summary)
        raise
    else:
        base.write_json(out/('stage_'+args.phase+'_summary.json'),summary)
        if args.phase == 'second':
            base.write_json(out/'train_summary.json',summary)
        base.write_json(out/'progress.json',summary)
        print('STAGE_COMPLETE '+json.dumps(summary),flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        history.close()


def calibrate(args):
    from utils.common import move2device, add_field_to_pyg_batch
    args.arm = 'Cell-F'
    created, cfg, payload = create(args, args.arm, phase='second')
    loader, _, runner, _ = created
    runner.model.eval()
    set_seed(2057)
    data_loader = loader.create_dataloader()
    iterator = iter(data_loader)
    target = targets.Criterion(targets.Config(family='D',scale=1.))
    rows, common, material_sum, target_sum = [], [], 0., 0.
    directory = args.output/'train_calibration_states'
    directory.mkdir(parents=True,exist_ok=True)
    before = float(runner.model._output_normalizer._num_accumulations)
    start = time.monotonic()
    try:
        for window in range(7):
            raw = move2device(next(iterator),args.device)
            raw = add_field_to_pyg_batch(raw,'iter',[2000],'cloth',reference_key=None)
            raw = runner.add_cloth_obj(raw)
            previous = None
            for frame in range(5):
                sample = runner.collect_sample(raw,frame,previous,random_ts=False)
                if frame==0:
                    sample = runner.collision_solver.solve(sample)
                repeat = random.randint(0,30)
                with torch.no_grad():
                    output = runner.model(sample,repeat,is_training=True)
                previous = output.detach()
                analysis = output.clone()
                q = analysis['cloth'].pred_pos.detach().requires_grad_(True)
                analysis = add_field_to_pyg_batch(analysis,'pred_pos',q,'cloth','pos')
                material = sum(runner.criterion_dict[key](analysis)['loss'] for key in
                               ('stretching_energy','bending_energy'))
                d0 = target(analysis)['loss']
                gm, = torch.autograd.grad(material,q,retain_graph=True)
                gd, = torch.autograd.grad(d0,q)
                cloth = analysis['cloth']
                free = cloth.vertex_type.reshape(-1)!=3
                mass = cloth.v_mass.reshape(-1)
                m2 = float(((gm[free]/mass[free,None]).double().square()).sum())
                d2 = float(((gd[free]/mass[free,None]).double().square()).sum())
                if not math.isfinite(m2+d2):
                    raise FloatingPointError('Nonfinite actual TRAIN calibration gradient')
                index = window*5+frame
                eligible = m2>0 and d2>0
                if eligible:
                    material_sum += m2
                    target_sum += d2
                    common.append(index)
                path = directory/f'state_{index:02d}.pth'
                torch.save(analysis.detach().cpu(),path)
                row = dict(index=index,window=window,frame=frame,eligible=eligible,
                    material_gradient_sq=m2,D_gradient_sq=d2,material_energy=float(material.detach()),
                    D0=float(d0.detach()),garment=analysis.get('garment_name'),
                    sequence=analysis.get('sequence_name'),K=repeat,tau=float(cloth.timestep.reshape(-1)[0]),
                    target_details=target.last,full_state_path=str(path))
                rows.append(row)
                print('TRAIN_CALIBRATION '+json.dumps(row,allow_nan=False),flush=True)
        if not common or material_sum<=0 or target_sum<=0:
            raise ValueError('No common positive fresh TRAIN material/D gradient support')
        result = dict(state='complete',scale=math.sqrt(material_sum/target_sum),states=35,
            common_nonzero_ids=common,material_gradient_sq_sum=material_sum,D_gradient_sq_sum=target_sum,
            reference_checkpoint=str(args.output/'trains/Cell-F/checkpoints/step_0000002000.pth'),
            material_reference='original TRAIN stretching+bending energy q gradient;free inverse mass',
            original_checkpoint_normalizer_accumulations=before,
            isolated_clone_normalizer_delta=float(runner.model._output_normalizer._num_accumulations)-before,
            original_checkpoint_changed=False,model_updates=0,rows=rows,
            elapsed_seconds=time.monotonic()-start,source_revision=base.source_revision())
    except BaseException as exc:
        result = dict(state='failed',failure=type(exc).__name__+': '+str(exc),rows=rows,
                      elapsed_seconds=time.monotonic()-start)
        base.write_json(args.output/'calibration.json',result)
        raise
    else:
        base.write_json(args.output/'calibration.json',result)
        print('CALIBRATION_COMPLETE '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    finally:
        del iterator,data_loader,runner,created,payload
        gc.collect()


def component(args):
    """Positive real-data coarse/fine backward observations, not a correctness theorem."""
    from utils.validation import Config,update_config_for_validation,load_runner,create_one_sequence_dataloader
    from utils.common import move2device,add_field_to_pyg_batch
    report = (json.loads(args.component_from.read_text()) if args.component_from else
              dict(scope='real coarse/fine component training;not formal7500 or H80',
                   scratch_updates_per_arm_case=3,scratch_weights_discarded=True,arms={}))
    revision = base.source_revision()
    if args.component_from:
        previous_revision = (args.component_from.parent/'component_Cell-F_source_commit.txt').read_text().strip()
        report['continuation_from'] = str(args.component_from)
        for rows in report['arms'].values():
            for row in rows:
                row.setdefault('source_revision', previous_revision)
    report.update(state='running',activation_recomputation_only=True,source_revision=revision)
    report.pop('failure',None)
    start = time.monotonic()
    try:
        for arm in ARMS:
            report['arms'].setdefault(arm,[])
            for resolution in ('.5','2.5'):
                expected_garment = 'dress_'+('0.5' if resolution=='.5' else '2.5')
                if any(row.get('garment')==expected_garment and row.get('state')=='complete'
                       for row in report['arms'][arm]):
                    continue
                modules,cfg = configuration(args,arm,args.device,args.output,training=False)
                cfg = update_config_for_validation(cfg,Config(density=.20022,lame_mu=23600.,
                    lame_lambda=44400.,bending_coeff=3.962e-5,separate_arms=True,
                    garment_dict_file='garments_dict_multi_res.pkl',smpl_model='smpl/SMPL_FEMALE.pkl'))
                _,runner = load_runner(modules,cfg)
                initial = torch.load(args.output/'initializations'/(arm+'.pth'),map_location='cpu',weights_only=False)
                runner.load_state_dict(initial['training_module'],strict=True)
                set_seed(57)
                garment = 'dress_'+('0.5' if resolution=='.5' else '2.5')
                loader = create_one_sequence_dataloader(str(args.sequence/'pose_sequences/07_02.pkl'),garment,modules,cfg)
                sequence = move2device(next(iter(loader)),args.device)
                sequence = add_field_to_pyg_batch(sequence,'iter',[100],'cloth',reference_key=None)
                sequence = runner.add_cloth_obj(sequence)
                sample = runner.collect_sample_wholeseq(sequence,0,None)
                sample = runner.collision_solver.solve(sample)
                opt = torch.optim.Adam(runner.parameters(),lr=5e-5)
                recording(runner.model,True)
                observations = []
                torch.cuda.reset_peak_memory_stats()
                for update in range(3):
                    before = float(runner.model._output_normalizer._num_accumulations)
                    torch.cuda.synchronize();begin=time.monotonic()
                    predicted = runner.model(sample.clone(),repeat=3,is_training=True)
                    losses = runner.criterion_pass(predicted)
                    opt.zero_grad(set_to_none=True)
                    sum(losses.values()).backward()
                    norm=torch.nn.utils.clip_grad_norm_(runner.model.parameters(),1.,error_if_nonfinite=True)
                    gs=gradients(runner)
                    opt.step();torch.cuda.synchronize()
                    observations.append(dict(update=update+1,losses={k:float(v.detach()) for k,v in losses.items()},
                        normalizer_delta=float(runner.model._output_normalizer._num_accumulations)-before,
                        gradients=gs,preclip_gradient_norm=float(norm),wall_seconds=time.monotonic()-begin,
                        execution=runner.model._learned_model.execution))
                row=dict(garment=garment,nodes=len(sample['cloth'].pos),state='complete',source_revision=revision,
                    capacity=capacity(runner),observations=observations,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                report['arms'][arm].append(row)
                base.write_json(args.output/'component_evidence.json',report)
                print('REAL_COMPONENT '+arm+' '+garment+' '+json.dumps(row),flush=True)
                del runner,opt,initial,predicted,losses,sample,sequence
                gc.collect();torch.cuda.empty_cache()
        report.update(state='complete',elapsed_seconds=time.monotonic()-start)
    except BaseException as exc:
        report.update(state='failed',failure=type(exc).__name__+': '+str(exc),elapsed_seconds=time.monotonic()-start)
        raise
    finally:
        base.write_json(args.output/'component_evidence.json',report)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('prepare','component','train','calibrate'))
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sequence',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--arm',choices=ALL_ARMS)
    p.add_argument('--phase',choices=('first','second'))
    p.add_argument('--component-from',type=Path)
    a=p.parse_args();a.iterations=STEPS;a.seed=57
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=True)
    install(a)
    globals()[a.mode](a)

if __name__=='__main__':
    main()
