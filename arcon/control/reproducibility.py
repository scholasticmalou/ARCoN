"""Random-state and configuration utilities used by the training controls."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_CORE = REPO_ROOT / 'arcon/models' / "pb4u_ctied_core.py"


def load_py312_compatible(name: str, source: Path, replacements: dict[str, str]):
    """Load an unchanged official module with Python 3.12 dataclass defaults fixed.

    Pb4U's released source targets the older dataclass behavior used by its
    original environment.  The experiment environment is Python 3.12, where
    those nested dataclass instances are rejected as mutable defaults.  This
    loader applies only the two compatibility substitutions already used by
    the official evaluation wrapper; it does not alter the official checkout.
    """
    text = source.read_text(encoding="utf-8")
    for original, compatible in replacements.items():
        count = text.count(original)
        if count != 1:
            raise RuntimeError(
                f"Unexpected official source {source}: {original!r} occurs {count} times"
            )
        text = text.replace(original, compatible)
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official module {name} from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(text, str(source), "exec"), module.__dict__)
    return module


def install_official_py312_compat(official_root: Path) -> None:
    """Preload the official config/runner modules without editing their files."""
    load_py312_compatible(
        "utils.arguments",
        official_root / "utils" / "arguments.py",
        {
            "from dataclasses import dataclass":
                "from dataclasses import dataclass, field",
            "dataloader: DataConfig = DataConfig()":
                "dataloader: DataConfig = field(default_factory=DataConfig)",
            "experiment: ExperimentConfig = ExperimentConfig()":
                "experiment: ExperimentConfig = field(default_factory=ExperimentConfig)",
        },
    )
    load_py312_compatible(
        "runners.pb4u",
        official_root / "runners" / "pb4u.py",
        {
            "from dataclasses import dataclass":
                "from dataclasses import dataclass, field",
            "optimizer: OptimConfig = OptimConfig()":
                "optimizer: OptimConfig = field(default_factory=OptimConfig)",
            "material: MaterialConfig = MaterialConfig()":
                "material: MaterialConfig = field(default_factory=MaterialConfig)",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["native", "L", "S", "R", "SC", "C_tied_phys"], required=True)
    parser.add_argument("--shared", type=int, choices=[0, 1], default=1)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--response-scale", type=float, default=0.10)
    parser.add_argument("--state-decay", type=float, default=0.90)
    parser.add_argument("--type-embedding-dim", type=int, default=8)
    parser.add_argument("--native-residual-scale", type=float, default=1.0)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--measure-response", action="store_true")
    parser.add_argument("--response-writeback", choices=["full", "gainmatched_full", "gainmatched_increment"], default="full")
    parser.add_argument("--seed", type=int, default=57)
    return parser.parse_args()


def install_runtime(args: argparse.Namespace) -> None:
    official_root = args.official_root.resolve()
    if not official_root.exists():
        raise FileNotFoundError(f"official Pb4U root does not exist: {official_root}")
    if not VENDOR_CORE.exists():
        raise FileNotFoundError(f"tracked response core does not exist: {VENDOR_CORE}")

    os.environ["PB4U_PROJECT"] = str(official_root)
    os.environ["PB4U_DATA"] = str(args.data_root.resolve())
    # The official project must win over any sibling research package named
    # models or utils.
    sys.path.insert(0, str(official_root))
    install_official_py312_compat(official_root)

    module_name = "models.core.pb4u_ctied"
    spec = importlib.util.spec_from_file_location(module_name, VENDOR_CORE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load tracked core from {VENDOR_CORE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(args: argparse.Namespace):
    from utils.arguments import create_modules, load_params

    modules, config = load_params(
        config_name="pb4u", config_dir=str(args.official_root.resolve() / "configs")
    )
    config.device = args.device
    config.model.pb4u.core_model = "pb4u_ctied"
    config.model.pb4u.ctied_mode = args.mode
    config.model.pb4u.ctied_shared = bool(args.shared)
    config.model.pb4u.response_writeback = args.response_writeback
    config.model.pb4u.response_scale = args.response_scale
    config.model.pb4u.state_decay = args.state_decay
    config.model.pb4u.type_embedding_dim = args.type_embedding_dim
    config.model.pb4u.native_residual_scale = args.native_residual_scale
    if args.mode == "C_tied_phys":
        config.model.pb4u.contact_temperature = 0.005
    config.experiment.max_iter = args.max_iter
    config.experiment.save_checkpoint_every = args.save_every
    config.experiment.name = (
        f"pb4u_{args.mode.lower()}_{'shared' if args.shared else 'unshared'}"
    )
    config.run_dir = str(args.run_dir.resolve())
    args.run_dir.mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf

    OmegaConf.save(config, args.run_dir / "config_resolved.yaml")
    created = create_modules(modules, config)
    # The core isolates extra-module RNG so outer native embeddings match.
    # Separately reset the seed here to match data order and worker seeds.
    set_seed(args.seed)
    return created, config


def _move_sample(sample, device):
    from utils.common import add_field_to_pyg_batch, move2device

    sample = move2device(sample, device)
    return add_field_to_pyg_batch(
        sample, "iter", [1] * sample.num_graphs, "cloth", reference_key=None
    )


def run_probe(args: argparse.Namespace) -> None:
    (dataloader_m, runner_module, training_module, aux_modules), config = configure(args)
    del runner_module, config
    dataloader = dataloader_m.create_dataloader()
    sample = _move_sample(next(iter(dataloader)), args.device)
    optimizer = aux_modules["optimizer"]
    cores = [m for m in training_module.modules() if hasattr(m, "record_response_diagnostics")]
    for core in cores:
        core.record_response_diagnostics = args.measure_response

    def response_gradients():
        return {
            name: (None if parameter.grad is None
                   else float(parameter.grad.detach().abs().max().cpu()))
            for name, parameter in training_module.named_parameters()
            if any(part in name for part in
                   ("response_stage", "response_readout", "source_carrier",
                    "physical_response"))
        }

    def max_gradient(entries, component=None):
        return max((value for name, value in entries.items()
                    if value is not None and (component is None or component in name)),
                   default=0.0)

    def native_gradients():
        return {
            component: max((float(parameter.grad.detach().abs().max().cpu())
                            for name, parameter in training_module.named_parameters()
                            if f".{component}." in name and parameter.grad is not None),
                           default=0.0)
            for component in ("node_encoder", "aggr_processor", "processor_steps", "decoder")
        }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        torch.cuda.synchronize()
    measurement_rng = None
    if args.measure_response:
        measurement_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    start = time.perf_counter()
    losses = training_module(sample.clone(), roll_steps=1, optimizer=optimizer, scheduler=None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    initial_response = [list(core.response_diagnostics) for core in cores]
    initial_physical_response = [list(core.physical_response_diagnostics) for core in cores]
    response_grads = response_gradients()
    initial_native_grads = native_gradients()

    # In S/SC the first backward opens zero-initialized source maps; in the
    # physical candidate it opens the zero scalar gain. The second real-batch
    # backward measures the newly reachable carrier or response-law/adapter.
    # These probe-only optimizer steps never affect the formal training seed.
    # This process never saves training weights; formal training starts afresh.
    post_update_losses = None
    second_response_grads = {}
    second_elapsed = None
    second_native_grads = {}
    if args.measure_response:
        random.setstate(measurement_rng[0])
        np.random.set_state(measurement_rng[1])
        torch.set_rng_state(measurement_rng[2])
        if measurement_rng[3] is not None:
            torch.cuda.set_rng_state_all(measurement_rng[3])
        second_start = time.perf_counter()
        post_update_losses = training_module(
            sample.clone(), roll_steps=1, optimizer=optimizer, scheduler=None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        second_elapsed = time.perf_counter() - second_start
        second_response_grads = response_gradients()
        second_native_grads = native_gradients()
    post_update_response = [list(core.response_diagnostics) for core in cores]
    post_update_physical_response = [list(core.physical_response_diagnostics) for core in cores]

    result = {
        "mode": args.mode,
        "shared": bool(args.shared),
        "response_writeback": args.response_writeback,
        "initialization_policy": training_module.model._learned_model.initialization_policy,
        "writeback_gain": [core.response_writeback_gain for core in cores],
        "initial_response_diagnostics": initial_response,
        "initial_physical_response_diagnostics": initial_physical_response,
        "post_update_response_diagnostics": post_update_response,
        "post_update_physical_response_diagnostics": post_update_physical_response,
        "post_update_losses": post_update_losses,
        "losses": losses,
        "elapsed_sec": elapsed,
        "second_backward_elapsed_sec": second_elapsed,
        "probe_optimizer_updates": 2 if args.measure_response else 1,
        "parameter_count": sum(p.numel() for p in training_module.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in training_module.parameters() if p.requires_grad
        ),
        "response_gradient_max": max_gradient(response_grads),
        "response_gradient_entries": response_grads,
        "initial_native_gradient_max": initial_native_grads,
        "second_native_gradient_max": second_native_grads,
        "second_response_gradient_entries": second_response_grads,
        "second_source_gradient_max": max_gradient(second_response_grads, "response_stage"),
        "second_carrier_gradient_max": max_gradient(second_response_grads, "source_carrier"),
        "second_physical_response_gradient_max": max_gradient(
            second_response_grads, "physical_response"
        ),
        "initial_physical_law_gradient_max": max_gradient(
            response_grads, "physical_response.response_law"
        ),
        "initial_physical_adapter_gradient_max": max_gradient(
            response_grads, "physical_response.adapter"
        ),
        "second_physical_law_gradient_max": max_gradient(
            second_response_grads, "physical_response.response_law"
        ),
        "second_physical_adapter_gradient_max": max_gradient(
            second_response_grads, "physical_response.adapter"
        ),
        "physical_response_parameter_count": sum(
            p.numel() for name, p in training_module.named_parameters()
            if "physical_response" in name
        ),
        "source_carrier_parameter_count": sum(
            p.numel() for name, p in training_module.named_parameters()
            if "source_carrier" in name
        ),
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(torch.device(args.device))
            if torch.cuda.is_available()
            else None
        ),
    }
    if args.mode == "C_tied_phys":
        initial_stages = [stage for group in initial_physical_response for stage in group]
        refreshed_stages = [stage for group in post_update_physical_response for stage in group]
        required_stages = sum(len(core.processor_steps) for core in cores)
        result["probe_gate"] = {
            "all_stages_measured": len(initial_stages) == len(refreshed_stages) == required_stages,
            "zero_initial_writeback": bool(initial_stages) and all(
                stage["response_gain"] == 0.0 and stage["injected_rms"] == 0.0
                for stage in initial_stages
            ),
            "internal_balance": bool(initial_stages) and all(
                stage["internal_balance_relative"] < 1e-5
                for stage in initial_stages + refreshed_stages
            ),
            "finite_losses_and_scale": all(np.isfinite(value) for value in losses.values())
            and all(np.isfinite(value) for value in (post_update_losses or {}).values())
            and bool(initial_stages) and all(
                np.isfinite(value) for stage in initial_stages + refreshed_stages
                for value in stage.values()
            ),
            "native_path_gradients": all(np.isfinite(value) and value > 0.0
                                         for value in initial_native_grads.values()),
            "response_law_gradients": np.isfinite(result["second_physical_law_gradient_max"])
            and result["second_physical_law_gradient_max"] > 0.0,
            "adapter_gradients": np.isfinite(result["second_physical_adapter_gradient_max"])
            and result["second_physical_adapter_gradient_max"] > 0.0,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    (args.run_dir / "probe_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    if "probe_gate" in result and not all(result["probe_gate"].values()):
        raise RuntimeError(f"Physical response real-batch gate incomplete: {result['probe_gate']}")


def run_training(args: argparse.Namespace) -> None:
    (dataloader_m, runner_module, training_module, aux_modules), config = configure(args)
    global_step = 0
    start = time.perf_counter()
    for epoch in range(int(config.experiment.n_epochs)):
        dataloader = dataloader_m.create_dataloader()
        global_step = runner_module.run_epoch(
            training_module,
            aux_modules,
            dataloader,
            epoch,
            config,
            global_step=global_step,
        )
        if global_step > args.max_iter:
            break
    elapsed = time.perf_counter() - start
    summary = {
        "mode": args.mode,
        "shared": bool(args.shared),
        "response_writeback": args.response_writeback,
        "initialization_policy": training_module.model._learned_model.initialization_policy,
        "max_iter": args.max_iter,
        "global_step": global_step,
        "elapsed_sec": elapsed,
        "parameter_count": sum(p.numel() for p in training_module.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in training_module.parameters() if p.requires_grad
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    (args.run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    install_runtime(args)
    if args.probe:
        run_probe(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
