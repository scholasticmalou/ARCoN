"""Adapters for author-released garment-simulation baseline implementations."""
from __future__ import annotations
import os
from pathlib import Path
from arcon.control.baseline_runtime import _argument_patches, _bind_packages, _metadata, _native_argv, _runner_patches, _seed
from arcon.control.evaluation_protocol import MATERIAL


def build_author_case(record, spec, output, device):
    import torch
    from omegaconf import OmegaConf
    source=Path(record['source_root']).resolve()
    os.environ.update(HOOD_PROJECT=str(source), HOOD_DATA=str(record['data_root']))
    torch.cuda.set_device(device)
    _bind_packages(source)
    patches=[]
    _argument_patches(source,patches)
    runner_name=record['runner_name']
    if runner_name!='postcvpr':
        _runner_patches(source,'postcvpr',patches)
    _runner_patches(source,runner_name,patches)
    from utils.arguments import load_module, create_runner
    from utils.validation import Config, update_config_for_validation, create_one_sequence_dataloader
    from utils.common import move2device
    payload=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    # Use the actual saved author configuration rather than inferring a model
    # from a filename. The public YAML remains the recorded cross-reference.
    if 'config' in payload:
        cfg=OmegaConf.create(payload['config'])
    elif record['backend']=='msagnet':
        # Some public releases carry state only. Use the actual author YAML
        # and require strict identity; never fill mismatching learned tensors.
        with _native_argv():
            from utils.arguments import load_params
            _,cfg=load_params(Path(record['config_file']).stem)
    else:
        raise ValueError('SENC author checkpoint is missing its saved config')
    cfg.device=device
    modules={k:load_module(kind,cfg[k]) for k,kind in [('model','models'),('runner','runners')]}
    modules['criterions']={name:load_module('criterions',cfg.criterions,name) for name in cfg.criterions}
    modules['dataset']=load_module('datasets',cfg.dataloader.dataset)
    validation=dict(**MATERIAL,separate_arms=True,garment_dict_file=spec['mesh_path'],
        smpl_model='smpl/SMPL_FEMALE.pkl',random_betas=False,zero_betas=True)
    if record['backend']=='senc':
        validation.update(wind=(0.,0.,0.),energy_type_prob='stvk',keep_length=False)
    cfg=update_config_for_validation(cfg,Config(**validation))
    native_cfg=cfg.runner[runner_name]
    native_cfg.initial_ts=1/3
    native_cfg.regular_ts=1/30
    _seed()
    _,runner,_=create_runner(modules,cfg,create_aux_modules=False)
    loaded=runner.load_state_dict(payload['training_module'],strict=True)
    state_count=len(payload['training_module'])
    del payload
    runner.to(device).eval().requires_grad_(False)
    _seed()
    loader=create_one_sequence_dataloader(spec['sequence'],spec['mesh_key'],modules,cfg)
    sequence=move2device(next(iter(loader)),device)
    _metadata(runner,sequence,cfg,source_root=str(source),checkpoint=record['checkpoint'],
        config_source='author checkpoint config',config_path=str(source/'configs'/record['config_file']),
        compatibility_patches=patches,source_initialization_frames=0,
        identity=record['identity'],checkpoint_state_tensors=state_count,
        strict_load_missing_keys=list(loaded.missing_keys),strict_load_unexpected_keys=list(loaded.unexpected_keys),
        motion_source=spec['sequence'],garment_asset=dict(source=spec['mesh_path'],key=spec['mesh_key'],geometry_modified=False),
        initialization_policy='unchanged author runner; common material/body/rest/input',
        separate_arms=True,wind_override=(0.,0.,0.) if record['backend']=='senc' else None,
        native_loss_evaluation=record.get('evaluate_native_losses',True))
    return runner,sequence,cfg


def senc_model_step(runner,state):
    """Exact _rollout pre-model path, including force differentiation and self edges."""
    import torch
    from utils.common import add_field_to_pyg_batch
    state=add_field_to_pyg_batch(state,'pred_pos',state['cloth'].pos,'cloth','pos')
    state['cloth'].pred_pos.requires_grad=True
    with torch.enable_grad():
        energy=sum(runner.criterion_dict[k](state)['loss'] for k in runner.mcfg.force_names)
        grad=torch.autograd.grad(energy,state['cloth'].pred_pos)[0]
        acc=-grad/state['cloth'].v_mass
    state['cloth'].pred_pos.grad=None
    state=add_field_to_pyg_batch(state,'explicit_acc',acc,'cloth','pos')
    with torch.no_grad():
        state=runner.add_space_neighbors(state)
        return runner.model(state,is_training=False)
