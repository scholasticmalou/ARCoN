#!/usr/bin/env python3
"""Continuation configurations for contact-query checkpoint compatibility."""
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
import arcon.control.contact_query_variants as query25
import arcon.control.contact_query_stage as query20

course, shared, base = phase.course, phase.shared, phase.base
MAX_SECONDS = 3 * 3600
RECIPES = {
    'Fixed-Query30k': dict(start=27500, stop=30000, parent_arm='Fixed-Query', origin='25k'),
    'Early-Query20k25k': dict(start=22500, stop=25000, parent_arm='Early-Query', origin='20k'),
    'Fixed-Query20k25k': dict(start=22500, stop=25000, parent_arm='Fixed-Query', origin='20k'),
}
ARMS = tuple(RECIPES)
READOUTS = dict.fromkeys(ARMS, 'target_face_floor')
RESUME_INFO = {}


def expected_counts(iterations):
    return course.expected_counts(iterations)


def parent_recipe_protocol(arm):
    recipe = RECIPES[arm]
    parent = query25 if recipe['origin'] == '25k' else query20
    return parent.protocol(recipe['parent_arm'])


def protocol(arm):
    recipe = RECIPES[arm]
    value = copy.deepcopy(parent_recipe_protocol(arm))
    end = recipe['stop']
    value.update(family='contact_query_followup_20261020_v1', arm=arm,
        parent_query_family=value['family'], parent_arm=recipe['parent_arm'],
        parent_data_iterations=recipe['start'], data_iterations=end,
        added_data_iterations=end-recipe['start'],
        added_optimizer_updates=expected_counts(end)[2]-expected_counts(recipe['start'])[2],
        expected_forwards=expected_counts(end)[1], expected_optimizer_updates=expected_counts(end)[2],
        evaluation_iterations=[end], further_budget_authorized=False,
        query_first_enabled_at=25000 if recipe['origin']=='25k' else 20000,
        initialization='full_model_Adam_scheduler_normalizers_rng_saved_Query',
        training_walltime_limit_seconds=MAX_SECONDS)
    return value


def configuration(args, arm, device, run_dir, *, training=True):
    modules, cfg = course.configuration(args, 'Cell-ScaleFixed', device, run_dir, training=training)
    cfg.experiment.name = 'contact_query_followup_20261020_' + arm
    cfg.experiment.max_iter = getattr(args, 'end_step', RECIPES[arm]['stop'])
    cfg.experiment.save_checkpoint_every = 2500
    return modules, cfg


def configure_readout(runner, arm):
    return query25.configure_readout(runner, RECIPES[arm]['parent_arm'])


def create(args):
    from utils.arguments import create_modules
    from omegaconf import OmegaConf
    global RESUME_INFO
    out = args.output/'trains'/args.arm
    if (out/'history.jsonl').exists() or (out/'train_summary.json').exists():
        raise RuntimeError('Preserve an already attempted continuation')
    recipe = RECIPES[args.arm]
    start = recipe['start']
    start_updates = course.expected_counts(start)[2]
    parent = args.resume_checkpoint
    parent_protocol = parent_recipe_protocol(args.arm)
    payload = torch.load(parent, map_location='cpu', weights_only=False)
    actual = (payload['global_step'], payload['forwards'], payload['optimizer_updates'])
    if actual != expected_counts(start) or payload['architecture_protocol'] != parent_protocol:
        raise ValueError('The complete approved saved Query parent identity differs')
    if payload['execution_policy'] != 'outer_only':
        raise ValueError('Retain the parent outer_only recomputation policy')
    if not payload['optimizer']['state'] or payload['scheduler']['last_epoch'] != start_updates:
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
    cfg.experiment.name = 'contact_query_followup_20261020_' + args.arm
    cfg.experiment.max_iter = args.end_step
    cfg.criterions[shared.TARGET_KEY].calibration_file = str(target)
    differences = phase.configuration_differences(payload['config'], OmegaConf.to_container(cfg))
    allowed = {('device',), ('run_dir',), ('experiment','name'), ('experiment','max_iter'),
               ('criterions',shared.TARGET_KEY,'calibration_file')}
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
            or aux['scheduler'].last_epoch != start_updates):
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
        all_training_configuration_fields_unchanged=True, original_absolute_course_preserved=True,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('resume-check', 'positive-check', 'train'))
    for name in ('project', 'data', 'parent-root', 'resume-checkpoint', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--control-revision', required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-seconds', type=int, default=MAX_SECONDS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_SECONDS:
        parser.error('Do not exceed the approved three-hour training cap')
    if args.output.resolve() == args.parent_root.resolve():
        parser.error('Do not overwrite a saved Query parent run')
    recipe = RECIPES[args.arm]
    args.seed, args.iterations = 57, recipe['stop']
    args.end_step = recipe['start']+1 if args.mode=='positive-check' else recipe['stop']
    args.stage_phase = 'contact_query_followup_' + args.arm
    phase.START = recipe['start']
    phase.START_UPDATES = expected_counts(recipe['start'])[2]
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    shared.install(args)
    if args.mode=='resume-check':
        create(args)
    else:
        phase.train(args, create_fn=create, save_fn=save_state, expected_fn=expected_counts)


if __name__ == '__main__':
    main()
