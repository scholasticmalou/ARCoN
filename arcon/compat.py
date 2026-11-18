"""Compatibility for inherited loader namespaces and historical checkpoint metadata.

The model implementation is ARCoN-named. The ``pb4u`` configuration namespaces
belong to the attributed upstream loader/runner API and are kept for reuse.
"""
from pathlib import Path
import sys


def register_legacy_core(module):
    """Allow old saved configurations to resolve the renamed ARCoN core."""
    sys.modules['models.core.shared_working_cell'] = module


def relocate_training_config(payload, data_root, calibration_file):
    """Relocate data/calibration inputs without changing tensors or training settings."""
    cfg = payload['config']
    cfg['dataloader']['dataset']['pb4u']['data_root'] = str(Path(data_root).resolve()/'vto_dataset/smpl_parameters')
    cfg['criterions']['pb4u.mataug.shared_cell_self_energy']['calibration_file'] = str(Path(calibration_file).resolve())
    cfg['model']['pb4u']['core_model'] = 'adaptive_cell'
    return payload


def checkpoint_stage(payload):
    """Read the original architecture protocol rather than guessing from a filename."""
    protocol = payload.get('architecture_protocol', {})
    family = str(protocol.get('family', ''))
    step = int(payload['global_step'])
    if family.startswith('material_response') and protocol.get('arm') == 'C':
        return 'final_refinement' if step > 22500 else 'material_response'
    if family.startswith('facefloor_phase'):
        return 'contact_feedback'
    if family.startswith('coarse_contact_interface'):
        return 'surface_contact'
    if family.startswith('shared_working_cell'):
        return 'backbone'
    raise ValueError('Checkpoint is not from the documented ARCoN training course: '+family)
