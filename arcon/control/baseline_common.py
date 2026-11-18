"""Common trajectory export for native baseline implementations."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import time

from arcon.control.baseline_runtime import GARMENTS, INITIAL_TS, MATERIAL, MODELS, MOTIONS, REGULAR_TS, RESOLUTIONS, build_native_case


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _json_value(value):
    """Serialize native auxiliary metrics, without inventing missing entries."""
    import numpy as np
    import torch
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _array(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _frame_zero(value):
    value = _array(value)
    if value.ndim == 3:
        return value[:, 0].copy()
    if value.ndim == 2:
        return value
    raise ValueError(f"Unexpected native position shape: {value.shape}")


def _sequence_context(sequence):
    """Preserve the original data order and RAW shared initial geometry."""
    cloth, obstacle = sequence["cloth"], sequence["obstacle"]
    return {
        "rest_pos": _array(cloth.rest_pos),
        "initial_pos": _frame_zero(cloth.pos),
        "previous_pos": _frame_zero(cloth.prev_pos),
        "cloth_faces": _array(cloth.faces_batch).T,
        "obstacle_faces": _array(obstacle.faces_batch).T,
        "mesh_edges": _array(sequence["cloth", "mesh_edge", "cloth"].edge_index),
        "vertex_type": _array(cloth.vertex_type),
        "initial_obstacle": _frame_zero(obstacle.pos),
        "previous_obstacle": _frame_zero(obstacle.prev_pos),
    }


def normalizer_snapshot(runner):
    """Only copy real checkpoint normalizer statistics; no hashes or calibration."""
    fields = {"_acc_count", "_num_accumulations", "_acc_sum", "_acc_sum_squared"}
    return {name: value.detach().cpu().clone()
            for name, value in runner.state_dict().items()
            if "normalizer" in name and name.rsplit(".", 1)[-1] in fields}


def normalizer_comparison(before, after):
    import torch
    keys = sorted(set(before) | set(after))
    changed = [k for k in keys if k not in before or k not in after
               or not torch.equal(before[k], after[k])]
    counts = lambda snapshot: {k: float(v.reshape(-1)[0]) for k, v in snapshot.items()
                               if k.endswith("._num_accumulations") or k.endswith("._acc_count")}
    return {"changed": bool(changed), "changed_keys": changed,
            "statistics_tensors": len(keys), "before_counts": counts(before),
            "after_counts": counts(after),
            "comparison": "direct equality of all saved normalizer statistics tensors"}


class _NativeCapture:
    """Read-only hooks for actual prediction-time body indices and native dt.

    Fine15's native return array stores obstacle.pos, while its predicted cloth
    and pinned targets correspond to obstacle.target_pos. Keeping BOTH arrays
    makes that distinction explicit without changing its native runner.
    """
    def __init__(self):
        self.bodies, self.predictions, self.timesteps, self.is_init = [], [], [], []
        self.initial = {}

    def before(self, module, inputs):
        state = inputs[0]
        cloth, body = state["cloth"], state["obstacle"]
        self.bodies.append(_array(body.target_pos))
        self.timesteps.append(float(cloth.timestep.reshape(-1)[0].detach().cpu())
                              if "timestep" in cloth else None)
        self.is_init.append(bool(cloth.is_init.reshape(-1)[0].detach().cpu())
                            if "is_init" in cloth else None)
        if not self.initial:
            self.initial = {
                "native_initial_pos": _array(cloth.pos),
                "native_previous_pos": _array(cloth.prev_pos),
                "native_initial_obstacle": _array(body.pos),
                "native_previous_obstacle": _array(body.prev_pos),
            }

    def after(self, module, inputs, state):
        self.predictions.append(_array(state["cloth"].pred_pos))


def _loss_series(metrics, frames):
    """Keep actual native per-frame loss values; only remove CC's hood/ namespace.

    CC's own add_metrics reports its native weight-divided losses. Values are
    NOT rescaled into Pb4U weighted losses here. The original dictionary and
    key mapping are exported separately for provenance.
    """
    losses, key_map = {}, {}
    for native_key, values in metrics.items():
        if not native_key.endswith("_loss"):
            continue
        if not isinstance(values, (list, tuple)) or len(values) != frames:
            raise ValueError(f"Native loss is not a {frames}-frame series: {native_key}")
        key = native_key.removeprefix("hood/")
        if key in losses:
            raise ValueError(f"Ambiguous native loss namespace: {native_key}")
        losses[key] = [float(v) for v in values]
        key_map[key] = native_key
    if frames and not losses:
        raise RuntimeError("Native bare=False rollout returned no per-frame losses")
    return losses, key_map


def _save_rollout(output, context, capture, pred, obstacle, native_obstacle=None):
    import numpy as np
    frames = len(pred)
    payload = dict(context, **capture.initial, pred=pred, obstacle=obstacle,
                   body_pose_frame_index=np.arange(2, frames + 2, dtype=np.int64))
    if native_obstacle is not None:
        payload["native_obstacle"] = native_obstacle
    if len(capture.timesteps) >= frames and all(v is not None for v in capture.timesteps[:frames]):
        payload["native_timestep"] = np.asarray(capture.timesteps[:frames], dtype=np.float64)
    if len(capture.is_init) >= frames and all(v is not None for v in capture.is_init[:frames]):
        payload["native_is_init"] = np.asarray(capture.is_init[:frames], dtype=np.bool_)
    np.savez_compressed(output / "rollout.npz", **payload)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion", choices=MOTIONS, required=True)
    parser.add_argument("--garment", choices=GARMENTS, required=True)
    parser.add_argument("--resolution", choices=RESOLUTIONS, required=True)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be a positive number of predicted endpoints")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("rollout.npz", "metrics.json", "summary.json")):
        raise FileExistsError(f"Case artifacts already exist; choose a fresh --output: {args.output}")
    total_start = time.monotonic()
    summary = {
        "state": "building", "model": args.model,
        "case": args.motion + "_" + args.garment + "_" + args.resolution,
        "motion": args.motion, "garment": args.garment, "resolution": args.resolution,
        "requested_frames": args.horizon, "frames": 0, "pred_shape": None,
        "native_repeat": None, "material": dict(MATERIAL),
        "device": args.device, "elapsed_seconds": 0.0, "peak_allocated_gib": None,
        "normalizer_changed": None, "scope": "native endpoint-only rollout",
        "is_complete_H80": False,
        "elapsed_scope": "native rollout including native losses/collision scoring and export hooks; not online-only cost",
        "peak_scope": "torch allocated peak during native rollout, not whole-process GPU memory",
        "evaluation_evidence": "H80 only if actually completed; shorter horizon is integration evidence",
        "common_initial_ts": INITIAL_TS, "common_regular_ts": REGULAR_TS,
    }
    _write_json(args.output / "summary.json", summary)
    runner, context, capture, before = None, None, _NativeCapture(), None
    handles, rollout_start = [], None
    try:
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        torch.set_num_threads(4)
        # CC initializes RAPIDS/Warp before selecting its torch context in the builder.
        if args.model != "contourcraft":
            torch.cuda.set_device(args.device)
        runner, sequence, config = build_native_case(
            args.model, args.library_root, args.output, args.motion, args.garment,
            args.resolution, args.device)
        context = _sequence_context(sequence)
        meta = runner._garment_common_metadata
        summary.update({"state": "running", "checkpoint": meta["checkpoint"],
                        "native_repeat": meta["native_repeat"],
                        "config": OmegaConf.to_container(config, resolve=True),
                        "provenance": meta, "setup_elapsed_seconds": time.monotonic() - total_start,
                        "initial_array_convention": "raw native dataloader frame0, same node order; native prepared prestate saved separately"})
        _write_json(args.output / "summary.json", summary)
        before = normalizer_snapshot(runner)
        handles = [runner.model.register_forward_pre_hook(capture.before),
                   runner.model.register_forward_hook(capture.after)]
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        rollout_start = time.monotonic()
        # CC's add_icontour sets detached geometry requires_grad=True and calls
        # autograd.grad. Explicit enable_grad preserves native inference even if
        # this entry is invoked inside a caller's no_grad; never use inference_mode.
        inference_context = torch.enable_grad() if args.model == "contourcraft" else contextlib.nullcontext()
        with inference_context:
            trajectory = runner.valid_rollout(sequence, n_steps=args.horizon, bare=False)
        torch.cuda.synchronize(args.device)
        summary["elapsed_seconds"] = time.monotonic() - rollout_start
        for handle in handles:
            handle.remove()
        handles = []
        native_pred = _array(trajectory["pred"])
        native_body = _array(trajectory["obstacle"])
        prefix = int(meta["source_initialization_frames"])
        pred, body = native_pred[prefix:], native_body[prefix:]
        frames = len(pred)
        if frames != len(capture.predictions) or frames != len(capture.bodies):
            raise ValueError("Native predicted frames and captured body-time indices disagree")
        # This is a true input-body alignment, not a fabricated model endpoint.
        obstacle = np.stack(capture.bodies, axis=0)
        if not np.isfinite(pred).all() or not np.isfinite(obstacle).all():
            raise FloatingPointError("Native rollout contains nonfinite geometry")
        if pred.ndim != 3 or pred.shape[1:] != context["rest_pos"].shape:
            raise ValueError("Native prediction no longer matches original rest node order/shape")
        native_metrics = _json_value(dict(trajectory["metrics"]))
        losses, key_map = _loss_series(native_metrics, frames)
        _save_rollout(args.output, context, capture, pred, obstacle, native_obstacle=body)
        if prefix:
            # Preserve CC's two original initialization frames, but never count
            # them among the H predicted endpoints passed to common scoring.
            np.savez_compressed(args.output / "native_initialization.npz",
                                pred=native_pred[:prefix], obstacle=native_body[:prefix])
        _write_json(args.output / "metrics.json", losses)
        _write_json(args.output / "native_metrics.json", native_metrics)
        normalizers = normalizer_comparison(before, normalizer_snapshot(runner))
        observed_ts = capture.timesteps[:frames]
        summary.update({
            "state": "complete" if frames == args.horizon else "partial",
            "frames": frames, "pred_shape": list(pred.shape),
            "source_saved_frames": len(native_pred), "source_initialization_frames": prefix,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device) / 2**30,
            "normalizer_changed": normalizers["changed"], "normalizer": normalizers,
            "native_loss_key_map": key_map,
            "loss_value_convention": ("native CC add_metrics values, dividing by criterion-reported weights"
                                      if args.model == "contourcraft" else "unchanged native runner criterion values"),
            "native_timestep": observed_ts,
            "prediction_body_pose_frame_index": list(range(2, frames + 2)),
            "initial_timestep_matches_common": bool(observed_ts and observed_ts[0] is not None
                                                      and abs(observed_ts[0] - INITIAL_TS) < 1e-7),
            "body_alignment": "one obstacle.target_pos from each actual native model pre-hook; no frame resampling",
            "native_body_return_matches_target": bool(body.shape == obstacle.shape and np.array_equal(body, obstacle)),
            "is_complete_H80": frames == args.horizon == 80,
            "total_elapsed_seconds": time.monotonic() - total_start,
        })
        _write_json(args.output / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("state", "case", "model", "frames", "pred_shape",
              "native_repeat", "elapsed_seconds", "peak_allocated_gib", "normalizer_changed")}), flush=True)
        if frames != args.horizon:
            raise RuntimeError(f"Native sequence yielded {frames} predictions, requested {args.horizon}")
    except BaseException as exc:
        for handle in handles:
            handle.remove()
        summary.update(state="failed", error=repr(exc),
                       total_elapsed_seconds=time.monotonic() - total_start,
                       captured_prediction_frames=len(capture.predictions), is_complete_H80=False)
        if rollout_start is not None and not summary["elapsed_seconds"]:
            summary["elapsed_seconds"] = time.monotonic() - rollout_start
        if runner is not None and before is not None:
            comparison = normalizer_comparison(before, normalizer_snapshot(runner))
            summary.update(normalizer=comparison, normalizer_changed=comparison["changed"])
        if context is not None and capture.predictions and not (args.output / "rollout.npz").exists():
            import numpy as np
            partial_pred = np.stack(capture.predictions)
            partial_body = np.stack(capture.bodies[:len(partial_pred)])
            _save_rollout(args.output, context, capture, partial_pred, partial_body)
        if capture.predictions:
            summary.update(frames=len(capture.predictions),
                           pred_shape=[len(capture.predictions), *capture.predictions[0].shape])
        if runner is not None:
            import torch
            summary["peak_allocated_gib"] = torch.cuda.max_memory_allocated(args.device) / 2**30
        _write_json(args.output / "summary.json", summary)
        raise


if __name__ == "__main__":
    main()
