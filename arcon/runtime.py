"""Public path adapter around the evaluated model construction (no new predictor)."""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import os
import random
import sys

ROOT = Path(__file__).resolve().parents[1]

def activate(data_root, external_root=None):
    data_root = Path(data_root).resolve()
    os.environ['PB4U_PROJECT'] = str(ROOT/'external/PB4U-GNet-main')
    os.environ['PB4U_DATA'] = str(data_root)
    os.environ['PB4U_MATRIX_BASE'] = str(Path(external_root).resolve() if external_root is not None else ROOT)
    sys.path.insert(0, str(ROOT))
    from arcon.control import material_stage as control
    return control

def seed():
    import numpy as np
    import torch
    random.seed(57); np.random.seed(57); torch.manual_seed(57)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(57)

def build(data_root, checkpoint, sequence_path, mesh_path, garment_key, output,
          device='cuda:0', external_root=None):
    control = activate(data_root, external_root)
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    seed()
    args = SimpleNamespace(project=ROOT/'external/PB4U-GNet-main',
        data=Path(data_root).resolve(), output=Path(output), arm='C',
        iterations=25000, device=device, seed=57)
    control.phase.shared.install(args)
    from utils import validation
    from utils.common import move2device
    modules, cfg = control.configuration(args, 'C', device, Path(output), training=False)
    cfg = validation.update_config_for_validation(cfg, validation.Config(
        density=.20022, lame_mu=23600., lame_lambda=44400., bending_coeff=3.962e-5,
        separate_arms=True, garment_dict_file=str(Path(mesh_path).resolve()),
        smpl_model='smpl/SMPL_FEMALE.pkl'))
    _, runner = validation.load_runner(modules, cfg)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    from arcon.compat import checkpoint_stage
    stage = checkpoint_stage(payload)
    if stage not in ('material_response', 'final_refinement'):
        raise ValueError('Inference expects an ARCoN material-stage checkpoint; earlier warmup stages use different response settings')
    runner.load_state_dict(payload['training_module'], strict=True)
    control.configure_readout(runner, 'C')
    runner.to(device).eval().requires_grad_(False)
    control.phase.shared.recording(runner.model, False)
    runner.model._learned_model.capture_stages = False
    runner.cloth_obj.cache = defaultdict(dict)
    seed()
    loader = validation.create_one_sequence_dataloader(str(Path(sequence_path).resolve()),
                                                      garment_key, modules, cfg)
    sequence = move2device(next(iter(loader)), device)
    return runner, sequence, cfg
