"""Adapters for the native graph-simulator baselines and their released checkpoints."""
from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import pickle
import random
import sys


MATERIAL = {
    "density": 0.20022,
    "lame_mu": 23600.0,
    "lame_lambda": 44400.0,
    "bending_coeff": 3.962e-5,
}
INITIAL_TS = 1.0 / 3.0
REGULAR_TS = 1.0 / 30.0
MODELS = ("fine15", "hood", "contourcraft")
GARMENTS = ("tank", "tshirt", "longsleeve", "dress")
MOTIONS = ("01_01", "07_02")
RESOLUTIONS = ("0.5", "1.0", "1.5", "2.5")


def _seed(seed=57):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def _native_argv():
    # CC's load_params always merges OmegaConf.from_cli(), even with a name.
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        yield
    finally:
        sys.argv = saved


def _bind_packages(source):
    """Bind the native namespace without importing a different project tree."""
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    for name in ("models", "datasets", "runners", "utils", "criterions"):
        existing = sys.modules.get(name)
        if existing is not None:
            origins = list(getattr(existing, "__path__", ()))
            if origins and all(Path(p).resolve().is_relative_to(source) for p in origins):
                continue
            raise RuntimeError(
                f"Native package {name!r} already belongs to another source tree; "
                "build each backend in a fresh process")
        spec = importlib.machinery.PathFinder.find_spec(name, [str(source)])
        if spec is None:
            raise ImportError(f"Missing native package {name}: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        if spec.loader is not None:
            spec.loader.exec_module(module)


def _load_compatible(name, path, replacements, patches):
    """Same in-memory dataclass repair as the already-run native probes."""
    text = path.read_text(encoding="utf-8")
    applied = []
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(f"Unexpected native compatibility target {path}: {old}")
        text = text.replace(old, new)
        applied.append({"original": old, "replacement": new})
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(text, str(path), "exec"), module.__dict__)
    patches.append({"source": str(path), "kind": "Python dataclass default_factory",
                    "changes": applied, "source_file_modified": False})
    return module


def _argument_patches(source, patches, contourcraft=False):
    replacements = {
        "from dataclasses import dataclass": "from dataclasses import dataclass, field",
        "dataloader: DataConfig = DataConfig()":
            "dataloader: DataConfig = field(default_factory=DataConfig)",
        "experiment: ExperimentConfig = ExperimentConfig()":
            "experiment: ExperimentConfig = field(default_factory=ExperimentConfig)",
    }
    if contourcraft:
        replacements["restart: RestartConfig = RestartConfig()"] = (
            "restart: RestartConfig = field(default_factory=RestartConfig)")
    _load_compatible("utils.arguments", source / "utils/arguments.py", replacements, patches)


def _runner_patches(source, name, patches):
    replacements = {
        "from dataclasses import dataclass": "from dataclasses import dataclass, field",
        "optimizer: OptimConfig = OptimConfig()":
            "optimizer: OptimConfig = field(default_factory=OptimConfig)",
        "material: MaterialConfig = MaterialConfig()":
            "material: MaterialConfig = field(default_factory=MaterialConfig)",
    }
    if name == "ccraft":
        replacements["safecheck: SafecheckConfig = SafecheckConfig()"] = (
            "safecheck: SafecheckConfig = field(default_factory=SafecheckConfig)")
    _load_compatible("runners." + name, source / "runners" / (name + ".py"),
                     replacements, patches)


def _numpy_pickle_load(stream):
    """Read the same numeric arrays across NumPy1/2 module-name changes."""
    import numpy as np
    old_numpy = int(np.__version__.split(".")[0]) < 2
    class NumericModuleReader(pickle.Unpickler):
        def find_class(self, module, name):
            if old_numpy and module.startswith("numpy._core"):
                module = "numpy.core" + module[len("numpy._core"):]
            return super().find_class(module, name)
    return NumericModuleReader(stream).load()


def _prepare_garment(root, output, garment_key, n_levels, *, mesh_path=None):
    """Export ONE original mesh; add only exact native hierarchy metadata.

    Raw multi_res is authoritative for rest positions, faces, node order, LBS
    and handles. The published hierarchical cache is reusable only for the
    exact same faces. Missing cache entries use the native graph/center/coarse
    definitions, not a coarse mesh substituted for the fine geometry.
    """
    import numpy as np
    raw_path = (Path(mesh_path).resolve() if mesh_path is not None else
                root / "data/pb4u_official/aux_data/garments_dict_multi_res.pkl")
    hierarchy_path = raw_path.with_name("garments_dict_multi_res_hier.pkl")
    with raw_path.open("rb") as stream:
        all_garments = _numpy_pickle_load(stream)
    if garment_key not in all_garments:
        raise KeyError(f"Original multi-resolution mesh missing: {garment_key}: {raw_path}")
    selected = all_garments[garment_key]
    del all_garments
    provenance = {"mesh_source": str(raw_path), "mesh_key": garment_key,
                  "geometry_or_lbs_modified": False, "native_coarse_levels": n_levels}
    # Fine15's original dataset also selects a graph center when n_levels=0.
    # Supply that native metadata rather than patching/skipping its data path.
    cached_complete = ("center" in selected and
        (not n_levels or all(center in selected.get("coarse_edges", {}) and
            all(i in selected["coarse_edges"][center] for i in range(n_levels))
            for center in selected["center"])))
    needs_hierarchy = (bool(n_levels) or "center" not in selected) if mesh_path is None else not cached_complete
    if needs_hierarchy:
        cached = None
        if hierarchy_path.is_file():
            with hierarchy_path.open("rb") as stream:
                hierarchy = _numpy_pickle_load(stream)
            cached = hierarchy.get(garment_key)
            del hierarchy
        # Some published middle-grid caches describe a different remeshing.
        # Keep the canonical raw geometry and regenerate only native hierarchy
        # metadata when that cache cannot refer to these exact indexed faces.
        cache_matches = cached is not None and (
            np.array_equal(selected["faces"], cached["faces"])
            and len(selected["rest_pos"]) == len(cached["rest_pos"]))
        if cached is not None and not cache_matches:
            provenance["unused_hierarchy_cache"] = {
                "source": str(hierarchy_path),
                "reason": "different indexed topology; original raw mesh retained"}
        if cache_matches:
            selected["center"] = cached["center"]
            selected["coarse_edges"] = {
                center: levels for center, levels in cached.get("coarse_edges", {}).items()
                if all(i in levels for i in range(n_levels))}
            provenance["hierarchy"] = {
                "source": str(hierarchy_path), "cached_centers": len(selected["coarse_edges"]),
                "missing_selected_center": "native dataset make_coarse_edges"}
            del cached
        else:
            import networkx as nx
            from utils.coarse import make_graph_from_faces
            # usebounds is an EXACT eccentricity-bound acceleration of nx.center;
            # it neither approximates the center set nor changes native graph edges.
            selected["center"] = nx.center(make_graph_from_faces(selected["faces"]),
                                           usebounds=True)
            selected["coarse_edges"] = {}
            provenance["hierarchy"] = {
                "source": "native utils.coarse.make_graph_from_faces + exact nx.center(usebounds=True)",
                "cached_centers": 0, "missing_selected_center": "native dataset make_coarse_edges",
                "approximate_center": False}
    folder = output / "inference_inputs"
    folder.mkdir(parents=True, exist_ok=True)
    return selected, folder, provenance


def _load_weights(runner, checkpoint):
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    runner.load_state_dict(payload["training_module"], strict=True)
    del payload


def _native_repeat(runner, config):
    name = next(iter(config.model))
    core = getattr(runner.model, "_learned_model", None)
    repeat = getattr(core, "_message_passing_steps", None)
    if repeat is None:
        repeat = config.model[name].get("message_passing_steps")
    if repeat is None:
        raise RuntimeError("Native model has no message_passing_steps; do not infer Pb4U K")
    return int(repeat)


def _metadata(runner, sequence, config, **values):
    name = next(iter(config.model))
    model_config = config.model[name]
    coarse_counts = {edge_type[1]: int(sequence[edge_type].edge_index.shape[1])
                     for edge_type in sequence.edge_types if edge_type[1].startswith("coarse_edge")}
    runner._garment_common_metadata = dict(
        material=dict(MATERIAL), native_repeat=_native_repeat(runner, config),
        native_repeat_source="native learned core _message_passing_steps, else its own model Config.message_passing_steps",
        native_hierarchy={"n_coarse_levels": int(model_config.get("n_coarse_levels", 0)),
                          "coarse_edge_counts": coarse_counts,
                          "architecture": model_config.get("architecture"),
                          "graph_edges_or_levels_reduced": False},
        prediction_time_indexing="0-based converted motion frames: prediction j targets pose frame j+2",
        **values)
    return runner


def build_pb4u_case(model, library_root, output, motion, garment, resolution, device="cuda:0", *, input_spec=None):
    """Pb4U/Fine15/HOOD: their own model, runner, dataset, YAML and weights."""
    if model not in ("fine15", "hood", "pb4u"):
        raise ValueError(model)
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    root, output = Path(library_root).resolve(), Path(output).resolve()
    source, data = root / "external/PB4U-GNet-main", root / "data/pb4u_official"
    os.environ["PB4U_PROJECT"], os.environ["PB4U_DATA"] = str(source), str(data)
    _bind_packages(source)
    patches = []
    _argument_patches(source, patches)
    runner_name = "baselines" if model == "fine15" else model
    _runner_patches(source, runner_name, patches)
    from utils.arguments import load_params
    from utils.validation import Config, update_config_for_validation, load_runner
    from utils.validation import create_one_sequence_dataloader
    from utils.common import move2device
    with _native_argv():
        modules, config = load_params(model)
    config.device = device
    config = update_config_for_validation(config, Config(
        **MATERIAL, separate_arms=True, garment_dict_file="garments_dict_multi_res.pkl",
        smpl_model="smpl/SMPL_FEMALE.pkl"))
    dataset_name = next(iter(config.dataloader.dataset))
    data_config = config.dataloader.dataset[dataset_name]
    key = input_spec["mesh_key"] if input_spec else garment + "_" + resolution
    selected, folder, mesh_provenance = _prepare_garment(
        root, output, key, int(data_config.n_coarse_levels),
        mesh_path=input_spec["mesh_path"] if input_spec else None)
    garment_path = Path(input_spec["mesh_path"]) if input_spec else folder / "garments_dict.pkl"
    if input_spec is None:
        with garment_path.open("wb") as stream:
            pickle.dump({key: selected}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    del selected
    data_config.garment_dict_file = str(garment_path)
    native_cfg = config.runner[runner_name]
    # These are the approved common values and the native YAML/default values.
    native_cfg.initial_ts, native_cfg.regular_ts = INITIAL_TS, REGULAR_TS
    _seed()
    _, runner = load_runner(modules, config)
    checkpoint = (Path(input_spec["checkpoint"]) if input_spec and input_spec.get("checkpoint")
                  else data / "trained_models" / (model + ".pth"))
    _load_weights(runner, checkpoint)
    runner.to(device).eval().requires_grad_(False)
    motion_path = (Path(input_spec["sequence"]) if input_spec else
        root / "data/hood_data/aux_data/validation_sequences/pose_sequences" / (motion + ".pkl"))
    _seed()
    loader = create_one_sequence_dataloader(str(motion_path), key, modules, config)
    sequence = move2device(next(iter(loader)), device)
    _metadata(runner, sequence, config, source_root=str(source), checkpoint=str(checkpoint),
              config_path=str(source / "configs" / (model + ".yaml")),
              model_module=modules["model"].__name__, runner_module=modules["runner"].__name__,
              dataset_module=modules["dataset"].__name__, compatibility_patches=patches,
              motion_source=str(motion_path), garment_asset=mesh_provenance,
              exported_garment_input=str(garment_path), seed=57, separate_arms=True,
              smpl_model=str(data / "aux_data/smpl/SMPL_FEMALE.pkl"),
              initialization_policy="unchanged native runner", source_initialization_frames=0,
              identity=("MGN-style Fine15 released with Pb4U; not asserted to be the original paper's MGN weight"
                        if model == "fine15" else "HOOD author model and weight released with Pb4U"
                        if model == "hood" else "official mature Pb4U checkpoint"),
              body_export="per-prediction native state obstacle.target_pos; native returned array is also saved")
    return runner, sequence, config


def build_contourcraft_case(library_root, output, motion, garment, resolution, device="cuda:0", *, input_spec=None):
    """Full released ContourCraft runner and four-level graph on ORIGINAL meshes.

    CC ships only separate tshirt/pants garment files. Exporting the selected
    original multi_res garment in that same file format supplies the requested
    mesh without switching models or using the wrong-resolution native tshirt.
    Its own dataset, contour gradients, proximity, collision preparation and
    collision counting remain untouched. No global no_grad is introduced.
    """
    import torch  # Bind PyTorch's CUDA libraries before RAPIDS, as in the native probe.
    from omegaconf import OmegaConf
    root, output = Path(library_root).resolve(), Path(output).resolve()
    source = root / "external/ContourCraft-main"
    data = root / "data/contourcraft_official"
    _bind_packages(source)
    from utils.defaults import DEFAULTS
    DEFAULTS.data_root = str(data)
    DEFAULTS.aux_data = str(data / "aux_data")
    DEFAULTS.project_dir = str(source)
    DEFAULTS.experiment_root = str(output)
    DEFAULTS.CMU_root = str(root / "data/hood_data/aux_data/validation_sequences/pose_sequences")
    # Keep the successful native RAPIDS/Warp/CUDA-extension import order.
    import utils.icontour
    import utils.warp_u.proximity
    patches = []
    _argument_patches(source, patches, contourcraft=True)
    _runner_patches(source, "ccraft", patches)
    from utils.arguments import load_params, create_runner
    from utils.validation import apply_material_params, create_postcvpr_one_sequence_dataloader
    from utils.common import move2device
    with _native_argv():
        modules, config = load_params("contourcraft")
    config.device = device
    config = apply_material_params(config, dict(MATERIAL))
    config.experiment.initial_ts, config.experiment.regular_ts = INITIAL_TS, REGULAR_TS
    config.runner.ccraft.initial_ts, config.runner.ccraft.regular_ts = INITIAL_TS, REGULAR_TS
    torch.cuda.set_device(device)
    _seed()
    _, runner, _ = create_runner(modules, config, create_aux_modules=False)
    checkpoint = data / "trained_models/contourcraft.pth"
    _load_weights(runner, checkpoint)
    runner.to(device).eval().requires_grad_(False)
    key = input_spec["mesh_key"] if input_spec else garment + "_" + resolution
    n_levels = int(config.dataloaders.inference.dataset.ccraft.n_coarse_levels)
    selected, folder, mesh_provenance = _prepare_garment(root, output, key, n_levels,
        mesh_path=input_spec["mesh_path"] if input_spec else None)
    selected.setdefault("gender", "female")
    selected.setdefault("name", key)
    garment_folder = folder / "cc_garment_dicts"
    garment_folder.mkdir(parents=True, exist_ok=True)
    garment_path = garment_folder / (key + ".pkl")
    with garment_path.open("wb") as stream:
        pickle.dump(selected, stream, protocol=pickle.HIGHEST_PROTOCOL)
    del selected
    motion_path = (Path(input_spec["sequence"]) if input_spec else
        root / "data/hood_data/aux_data/validation_sequences/pose_sequences" / (motion + ".pkl"))
    with motion_path.open("rb") as stream:
        pose = _numpy_pickle_load(stream)
    added_subsample = "subsample" not in pose
    if added_subsample:
        # Native CC Loader requires this metadata; no frames are subsampled.
        pose["subsample"] = 1
    if pose["subsample"] != 1:
        raise ValueError(f"Expected original 30Hz input without subsampling: {motion_path}")
    exported_motion = folder / (motion + ".pkl")
    with exported_motion.open("wb") as stream:
        pickle.dump(pose, stream, protocol=pickle.HIGHEST_PROTOCOL)
    del pose
    _seed()
    loader = create_postcvpr_one_sequence_dataloader(
        str(exported_motion), key, config="contourcraft", sequence_loader="hood_pkl",
        garment_dicts_dir=str(garment_folder), obstacle_dict_file="smpl_aux.pkl",
        gender="female", separate_arms=True, body_model_root="body_models_numpy",
        pinned_verts=True, wholeseq=True, random_betas=False, noise_scale=0.,
        n_coarse_levels=n_levels, nobody_freq=0.)
    # Return the ACTUAL dataloader config, not the unrelated CMU defaults in YAML.
    config.dataloaders.inference.dataset.ccraft = OmegaConf.structured(loader.dataset.loader.mcfg)
    sequence = move2device(next(iter(loader)), device)
    _metadata(runner, sequence, config, source_root=str(source), checkpoint=str(checkpoint),
              config_path=str(source / "configs/contourcraft.yaml"),
              model_module=modules["model"].__name__, runner_module=modules["runner"].__name__,
              dataset_module=modules["datasets"]["inference"].__name__,
              compatibility_patches=patches, motion_source=str(motion_path),
              exported_motion_input=str(exported_motion),
              motion_metadata_patch={"added_subsample": added_subsample, "value": 1,
                                     "motion_arrays_or_frame_order_modified": False},
              garment_asset=mesh_provenance, exported_garment_input=str(garment_path),
              garment_format_patch="one original garment dict per .pkl; gender/name metadata only",
              seed=57, separate_arms=True,
              smpl_model=str(data / "aux_data/body_models_numpy"),
              identity="released ContourCraft model/checkpoint, full native ccraft runner",
              initialization_policy="unchanged CC long valid_rollout: two initial frames then regular-ts endpoints",
              source_initialization_frames=2,
              temporal_protocol_difference="native CC collect_sample(..., is_short=False) uses regular_ts even for the first prediction; initial_ts is configured but not used by long inference",
              native_step={"prepare_once": "runner.prepare_sample(sequence)",
                           "collect": "runner.collect_sample(sequence, index, previous_output)",
                           "first_step": "runner.update_sample_1st_step(sample_step, sequence)",
                           "forward": "runner.model(sample_step)",
                           "inference_context": "torch.enable_grad must enclose the complete native model call; geometry autograd in add_icontour cannot run under outer no_grad alone",
                           "prepare_autograd": "prepare_sample uses masks/material/cloth geometry without autograd.grad; no_grad is permissible",
                           "first_step_autograd": "update_sample_1st_step uses discrete collision masks, not autograd.grad; no_grad is permissible",
                           "forbidden_context": "torch.inference_mode cannot be repaired by nested enable_grad for native contour tensors"},
              warm_repeat_state_policy={
                  "template": "prepare a fresh case sequence once; preserve the initialized native cutout masks for that case",
                  "first_step": "when use_safecheck=True, run the real i0 update once and keep its returned sequence masks; do not recompute i0 masks from F20/F80 states",
                  "each_independent_repeat": "collect from an unchanged prepared template and inject the same prestate; never reuse the preceding repeat output as prev_out",
                  "mutated_graph": "model modifies node/edge features and contour fields; use a fresh collected/detached-cloned sample for every repeat",
                  "within_real_rollout": "carry detached prev_out and persistent initialized cutouts exactly as native _rollout does",
                  "reusable_static_state": "fixed weights/eval normalizers and same-garment rest-geometry cloth cache; no cache clearing per warm repeat"},
              collision_policy="native YAML/defaults, no disabling of contour/selfcoll/proximity/preparation",
              body_export="native target_pos for each prediction; two original body init frames saved separately")
    return runner, sequence, config


def build_native_case(model, library_root, output, motion, garment, resolution, device="cuda:0", *, input_spec=None):
    """Return (native runner, device sequence, full native OmegaConf config).

    No inference, optimizer, scoring, queue, or cost loop runs here. Input
    serialization/hierarchy preparation may populate output/inference_inputs.
    Use a separate process for each backend, exactly as with the native probes.
    """
    resolution = str(resolution)
    if input_spec is None:
        if model not in MODELS or motion not in MOTIONS or garment not in GARMENTS or resolution not in RESOLUTIONS:
            raise ValueError(f"Unsupported legacy native case: {(model, motion, garment, resolution)}")
    else:
        if model not in MODELS + ("pb4u",):
            raise ValueError(model)
        for key in ("sequence", "mesh_path", "mesh_key", "horizon"):
            if key not in input_spec:
                raise ValueError("Explicit full-sequence input missing: " + key)
    if model == "contourcraft":
        return build_contourcraft_case(library_root, output, motion, garment, resolution, device, input_spec=input_spec)
    return build_pb4u_case(model, library_root, output, motion, garment, resolution, device, input_spec=input_spec)
