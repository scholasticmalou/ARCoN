"""Reader-facing orchestration of the original ARCoN training stages.

The numerical training loops remain in arcon.control. This module supplies
explicit stage boundaries, relocates inputs, and saves/resumes complete state.
"""
from __future__ import annotations
import copy
import importlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace
from arcon.runtime import ROOT, activate
from arcon.compat import checkpoint_stage, relocate_training_config


def _read_checkpoint(path):
    import torch
    payload=torch.load(path,map_location='cpu',weights_only=False)
    for key in ['training_module','config','global_step','optimizer','scheduler','random_states']:
        if key not in payload:
            raise ValueError(f'Training requires a full saved training checkpoint; missing {key}. The distributed 25k inference weights are for infer.py.')
    checkpoint_stage(payload)
    return payload


def _restore_random_states(payload):
    import numpy as np
    import torch
    states=payload['random_states']
    random.setstate(states['python']);np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    if len(states['cuda'])!=torch.cuda.device_count():
        raise ValueError('Use one visible CUDA GPU, as in the original training course')
    torch.cuda.set_rng_state_all(states['cuda'])


def _initialize_backbone(args,module):
    """Create the same seed-57 initial model without a pretrained checkpoint."""
    import torch
    from omegaconf import OmegaConf
    module.shared.install(args)
    from utils.arguments import create_runner
    module.shared.set_seed(57)
    modules,cfg=module.configuration(args,args.arm,'cpu',args.output/'trains'/args.arm)
    _,runner,aux=create_runner(modules,cfg)
    record=dict(arm=args.arm,state='complete',capacity=module.shared.capacity(runner),
        optimizer_state_entries=len(aux['optimizer'].state),
        normalizer_accumulations=float(runner.model._output_normalizer._num_accumulations),
        initial_lr=float(aux['optimizer'].param_groups[0]['lr']),
        initialization_role='new_seed57_parameters_no_trained_checkpoint')
    directory=args.output/'initializations';directory.mkdir(parents=True)
    torch.save(dict(training_module=runner.state_dict(),config=OmegaConf.to_container(cfg),
        architecture_protocol=module.protocol(args.arm),global_step=0,optimizer_updates=0,forwards=0,
        initialization=record,execution_policy='outer_only',seed=57),directory/(args.arm+'.pth'))
    (directory/'initialization.json').write_text(json.dumps(record,indent=2)+'\n')


def _resume_backbone(args,module):
    """Restore all trainable and stochastic state; start a new native loader pass."""
    from omegaconf import OmegaConf
    module.shared.install(args)
    from utils.arguments import create_modules
    payload=_read_checkpoint(args.resume)
    if payload['architecture_protocol']!=module.protocol(args.arm):
        raise ValueError('Resume checkpoint uses a different backbone training protocol')
    relocate_training_config(payload,args.data,args.output/'calibration.json')
    module.shared.set_seed(57)
    out=args.output/'trains'/args.arm
    modules,_=module.configuration(args,args.arm,args.device,out)
    cfg=OmegaConf.create(copy.deepcopy(payload['config']))
    cfg.device=args.device;cfg.run_dir=str(out);cfg.experiment.max_iter=args.end_step
    created=create_modules(modules,cfg)
    runner,aux=created[2:]
    runner.load_state_dict(payload['training_module'],strict=True)
    module.shared.install(args).configure_mlp_recomputation(runner.model,payload['execution_policy']=='nested')
    for key in ['optimizer','scheduler']:aux[key].load_state_dict(payload[key])
    if aux['scheduler'].last_epoch!=module.expected_counts(payload['global_step'])[2]:
        raise ValueError('Restored scheduler age disagrees with cumulative optimizer updates')
    out.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,out/'config_resolved.yaml')
    _restore_random_states(payload)
    return created,cfg,payload


def _run_stage(args,spec):
    import torch
    if torch.cuda.device_count()!=1:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly one GPU for the original RNG configuration')
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    start=0;parent_kind=None
    if args.resume is not None:
        parent=_read_checkpoint(args.resume);start=int(parent['global_step']);parent_kind=checkpoint_stage(parent);del parent
    if not spec['start']<=start<min(spec['end'],args.until_step):
        raise ValueError(f"Stage {spec['name']} expects a checkpoint in [{spec['start']}, {min(spec['end'],args.until_step)}); got {start}")
    args.output.mkdir(parents=True)
    resources=args.output/'resources';resources.mkdir()
    for directory in [resources,args.output]:shutil.copyfile(ROOT/'assets/calibration.json',directory/'calibration.json')
    activate(args.data_root,args.external_root)
    module=importlib.import_module('arcon.control.'+spec['module'])
    a=SimpleNamespace(**vars(args))
    a.data=args.data_root.resolve();a.project=ROOT/'external/PB4U-GNet-main'
    a.output=args.output.resolve();a.arm=spec['arm'];a.seed=57
    a.end_step=min(spec['end'],args.until_step);a.iterations=25000
    a.parent_root=resources;a.face_parent_root=resources;a.parent_checkpoint=args.resume
    a.control_revision='anonymous-review';a.mode='train';a.stage_phase=spec['name']
    a.max_seconds=args.max_seconds or spec['max_seconds']
    same_stage=parent_kind==spec['name'] or (spec['name']=='final_refinement' and parent_kind=='material_response')
    a.resume_checkpoint=args.resume if same_stage else None
    module.shared.install(a)
    if spec['name']=='backbone':
        original_configuration=module.configuration
        def configuration(*pos,**kw):
            modules,cfg=original_configuration(*pos,**kw);cfg.experiment.max_iter=a.end_step;return modules,cfg
        module.configuration=configuration
        if args.resume is None:_initialize_backbone(a,module)
        else:module.create=lambda unused:_resume_backbone(a,module)
        module.train(a)
    else:
        module.START=start;module.START_UPDATES=module.expected_counts(start)[2]
        if spec['module']=='material_stage':
            module.phase.START=start;module.phase.START_UPDATES=module.START_UPDATES
            module.phase.train(a,create_fn=module.create,save_fn=module.save_state,expected_fn=module.expected_counts)
        else:module.train(a)
    checkpoint=a.output/'trains'/a.arm/'checkpoints'/f'step_{a.end_step:010d}.pth'
    if not checkpoint.is_file():raise RuntimeError('Original training loop did not write its end-of-stage checkpoint')
    summary=json.loads((a.output/'trains'/a.arm/'train_summary.json').read_text())
    result=dict(stage=spec['name'],start_step=start,end_step=a.end_step,checkpoint=str(checkpoint),
        data_iterations=summary['data_iterations'],optimizer_updates=summary['updates'],
        resumed=args.resume is not None,loader_cursor_restored=False,
        physical_training_control='arcon.control.'+spec['module'])
    (a.output/'training_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ARCON_TRAINING_RESULT '+json.dumps(result),flush=True)
    return result


def run(args,course):
    """Execute explicit stages in fresh processes to isolate native model namespaces."""
    if args.stage!='all':
        return _run_stage(args,next(s for s in course['stages'] if s['name']==args.stage))
    step=0;checkpoint=args.resume
    if checkpoint is not None:
        payload=_read_checkpoint(checkpoint);step=int(payload['global_step']);del payload
    if step>=args.until_step:raise ValueError('--until-step must be later than the resume checkpoint')
    args.output.mkdir(parents=True)
    results=[]
    for index,spec in enumerate(course['stages'],1):
        if step>=spec['end'] or args.until_step<=spec['start']:continue
        boundaries=sorted(set(spec.get('loader_restart_steps',[])+[spec['end']]))
        for boundary in boundaries:
            if step>=boundary or step>=args.until_step:continue
            target=min(boundary,args.until_step)
            destination=args.output/'stages'/f"{index:02d}_{spec['name']}_{step:05d}"
            cmd=[sys.executable,str(ROOT/'train.py'),'--data-root',str(args.data_root.resolve()),
                 '--external-root',str(args.external_root.resolve()),'--output',str(destination),
                 '--stage',spec['name'],'--until-step',str(target),'--device',args.device]
            if checkpoint is not None:cmd+=['--resume',str(checkpoint.resolve())]
            if args.max_seconds is not None:cmd+=['--max-seconds',str(args.max_seconds)]
            destination.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(cmd,check=True)
            result=json.loads((destination/'training_result.json').read_text());results.append(result)
            step=result['end_step'];checkpoint=Path(result['checkpoint'])
            (args.output/'training_result.json').write_text(json.dumps(dict(stages=results,end_step=step,checkpoint=str(checkpoint)),indent=2)+'\n')
    return results
