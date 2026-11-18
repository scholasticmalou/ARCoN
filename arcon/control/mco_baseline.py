"""Native MCO baseline construction and trajectory export."""
from __future__ import annotations
import os
from pathlib import Path
import re
import sys

ROOT=Path(os.environ.get('ARCON_MCO_ROOT',str(Path.cwd()/'outputs/mco_runtime')))
REV='34e809cab655dd191536179d4a1d793a483d7ef0'
SOURCE=Path(os.environ.get('ARCON_MCO_SOURCE',str(Path(__file__).resolve().parents[2]/'external/MCO')))
PROFILE='contourcraft_armBO_v21aTC_itmw_b0cap_ev2_walkfast'
CHECKPOINT=Path(os.environ.get('ARCON_MCO_CHECKPOINT',str(Path(__file__).resolve().parents[2]/'checkpoints/baselines/contourcraft_frictionless.pth')))
MU=0.5
for _key,_rel in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','TORCH_EXTENSIONS_DIR':'cache/torch_extensions','CUDA_CACHE_PATH':'cache/cuda','TRITON_CACHE_DIR':'cache/triton'}.items():
    os.environ[_key]=str(ROOT/_rel)
MATERIAL=dict(density=0.20022,lame_mu=23600.,lame_lambda=44400.,bending_coeff=3.962e-5)

def _dataclass_compat(name,path,patches):
    from arcon.control.baseline_runtime import _load_compatible
    if name in sys.modules:return sys.modules[name]
    text=Path(path).read_text(encoding='utf-8')
    replacements={}
    for match in re.finditer(r'(?m)^(\s+\w+:\s*(\w+Config)\s*=\s*)\2\(\)',text):
        old=match.group(0).lstrip()
        replacements[old]=old[:old.rfind(match.group(2)+'()')]+'field(default_factory='+match.group(2)+')'
    if replacements:
        replacements['from dataclasses import dataclass']='from dataclasses import dataclass, field'
        return _load_compatible(name,Path(path),replacements,patches)
    __import__(name)
    return sys.modules[name]

def bind_author():
    import torch
    from arcon.control.baseline_runtime import _bind_packages, _seed
    os.environ.setdefault('TMPDIR',str(ROOT/'tmp'))
    os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'cache'))
    torch.set_num_threads(4)
    _seed(57)
    _bind_packages(SOURCE)
    from utils.defaults import DEFAULTS
    DEFAULTS.data_root=os.environ.get('ARCON_MCO_DATA',str(SOURCE/'ccraft_data'))
    DEFAULTS.aux_data=str(Path(DEFAULTS.data_root)/'aux_data')
    DEFAULTS.project_dir=str(SOURCE)
    DEFAULTS.experiment_root=str(ROOT/'experiments')
    DEFAULTS.CMU_root=os.environ.get('ARCON_MCO_MOTION_ROOT',str(Path(DEFAULTS.data_root)/'aux_data/validation_sequences/pose_sequences'))
    # Preserve the existing successful PyTorch -> RAPIDS/Warp/CCCollisions import order.
    import utils.icontour
    import utils.warp_u.proximity
    patches=[]
    _dataclass_compat('utils.arguments',SOURCE/'utils/arguments.py',patches)
    _dataclass_compat('runners.ccraft',SOURCE/'runners/ccraft.py',patches)
    return patches

def build_author_runner(device='cuda:0', *, material=None):
    import torch
    from omegaconf import OmegaConf
    from arcon.control.baseline_runtime import _native_argv
    patches=bind_author()
    from utils.arguments import load_params,create_runner
    from utils.validation import apply_material_params
    with _native_argv():modules,cfg=load_params(PROFILE)
    cfg.device=device
    # Existing licensed SMPL files are stored as NumPy arrays, as in the already-run CC baseline.
    cfg.dataloaders.inference.dataset.ccraft.body_model_root='body_models_numpy'
    if material is None:
        from evaluate import EVAL_MATERIAL
        material=dict(EVAL_MATERIAL)
    cfg=apply_material_params(cfg,material)
    cfg.runner.ccraft.material.friction_mu_override=MU
    _,runner,_=create_runner(modules,cfg,create_aux_modules=False)
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    state=payload['training_module']
    restored=runner.load_state_dict(state,strict=True)
    bo=cfg.runner.ccraft.contact_arm.bolt_on
    if cfg.runner.ccraft.contact_arm.mode!='bolt_on' or not bo.wave_enabled or float(bo.wave_strain_cap)!=0.2:
        raise ValueError('The approved full author MCO configuration changed')
    metadata=dict(identity='author frictionless-retrained ContourCraft + complete released MCO',
        source_root=str(SOURCE),source_revision=REV,checkpoint=str(CHECKPOINT),
        checkpoint_bytes=CHECKPOINT.stat().st_size,checkpoint_keys=list(payload),
        checkpoint_state_tensors=len(state),strict_missing_keys=list(restored.missing_keys),
        strict_unexpected_keys=list(restored.unexpected_keys),parameters=sum(p.numel() for p in runner.parameters()),
        model_parameters=sum(p.numel() for p in runner.model.parameters()),
        profile=PROFILE,friction_mu=MU,material=dict(material),
        actual_operator=OmegaConf.to_container(cfg.runner.ccraft.contact_arm,resolve=True),
        configured_initial_ts=float(cfg.runner.ccraft.initial_ts),configured_regular_ts=float(cfg.runner.ccraft.regular_ts),
        long_rollout_actual_initial_ts=float(cfg.runner.ccraft.regular_ts),
        initialization='author long collect_sample is_short=False; regular_ts on every prediction; two given leading frames',
        source_initialization_frames=2,body_model_storage='existing licensed SMPL NumPy representation',compatibility_patches=patches,
        random_seed=57,numerical_source_modified=False)
    del state,payload
    runner.to(device).eval().requires_grad_(False)
    runner._garment_common_metadata=metadata
    return runner,modules,cfg

def load_author_cell(modules,cfg,garment='tank',motion='05_06',device='cuda:0'):
    import pandas as pd
    from omegaconf import OmegaConf
    from tools.validate_cells import read_eval_set,SPLITS
    import evaluate as evaluation
    from utils.arguments import create_dataloader_modules
    from utils.common import move2device
    folder=Path(os.environ.get('ARCON_MCO_DATA',str(SOURCE/'ccraft_data')))/'aux_data/evaluation_sequences'
    split=read_eval_set(folder)[(garment,motion)]
    csv=evaluation.find_datasplit(folder,SPLITS[split])
    rows=pd.read_csv(csv)
    selected=rows[(rows.garment.astype(str)==garment)&(rows.id.astype(str)==motion)]
    if len(selected)!=1:raise ValueError('Expected exactly one published evaluation cell')
    cfg=OmegaConf.create(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg,False)
    cfg=evaluation._patch_dataset_config(cfg,folder,csv,betas_source='native')
    loader=create_dataloader_modules(modules,cfg)['inference'].create_dataloader(is_eval=True)
    # Select the exact published row before fetching unrelated, unavailable garments.
    index=int(selected.index[0])
    batch=loader.collate_fn([loader.dataset[index]])
    return move2device(batch,device),split


def build_common_case(manifest,case,output,device='cuda:0'):
    """Build full author MCO on the unchanged frozen common garment and motion."""
    import pickle
    import numpy as np
    from omegaconf import OmegaConf
    from arcon.control.evaluation_protocol import input_spec
    from arcon.control.baseline_runtime import _prepare_garment, _numpy_pickle_load, _seed
    runner,modules,cfg=build_author_runner(device,material=manifest['material'])
    from utils.validation import create_postcvpr_one_sequence_dataloader
    from utils.common import move2device
    spec=input_spec(manifest,case,'MCO');key=spec['mesh_key']
    n_levels=int(cfg.dataloaders.inference.dataset.ccraft.n_coarse_levels)
    selected,folder,provenance=_prepare_garment(Path(manifest['library_root']),Path(output),key,n_levels,mesh_path=spec['mesh_path'])
    selected.setdefault('gender','female');selected.setdefault('name',key)
    garments=folder/'cc_garment_dicts';garments.mkdir(parents=True,exist_ok=True)
    with (garments/(key+'.pkl')).open('wb') as stream:pickle.dump(selected,stream,protocol=pickle.HIGHEST_PROTOCOL)
    with Path(spec['sequence']).open('rb') as stream:pose=_numpy_pickle_load(stream)
    pose.setdefault('subsample',1)
    if pose['subsample']!=1:raise ValueError('Frozen common motion must remain 30Hz')
    motion=folder/(manifest['cases'][case]['motion']+'.pkl')
    with motion.open('wb') as stream:pickle.dump(pose,stream,protocol=pickle.HIGHEST_PROTOCOL)
    _seed(57)
    loader=create_postcvpr_one_sequence_dataloader(str(motion),key,config=PROFILE,sequence_loader='hood_pkl',
        garment_dicts_dir=str(garments),obstacle_dict_file='smpl_aux.pkl',gender='female',separate_arms=True,
        body_model_root='body_models_numpy',pinned_verts=True,wholeseq=True,random_betas=False,noise_scale=0.,
        n_coarse_levels=n_levels,nobody_freq=0.)
    cfg.dataloaders.inference.dataset.ccraft=OmegaConf.structured(loader.dataset.loader.mcfg)
    sequence=move2device(next(iter(loader)),device)
    if sequence['obstacle'].lookup.shape[1]!=spec['horizon']:raise ValueError('Full common endpoint axis differs')
    from arcon.control.trajectory_evaluation import canonical_inputs
    common=canonical_inputs(manifest,'MCO',case,sequence,device)
    runner._garment_common_metadata.update(common_kinematic_input=common,garment_asset=provenance,
        motion_source=spec['sequence'],exported_motion_input=str(motion),native_repeat=1,
        native_repeat_source='one full author GNN + full MCO per endpoint',n_coarse_levels=n_levels,
        source_initialization_frames=2,free_future_cloth_input=False)
    return runner,sequence,cfg
