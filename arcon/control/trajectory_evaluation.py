#!/usr/bin/env python3
"""Complete autoregressive trajectory evaluation with common native inputs."""
from __future__ import annotations
import argparse
from collections import defaultdict
import contextlib
import copy
import json
import math
import os
import signal
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from arcon.control.evaluation_protocol import MATERIAL, SIX, case_dir, finite_mean, input_spec, now, read, write
from arcon.control.baseline_runtime import build_native_case, _metadata, _seed
from arcon.control.baseline_common import _array, _sequence_context, _NativeCapture, _loss_series, _json_value, _save_rollout, normalizer_snapshot, normalizer_comparison


def canonical_inputs(manifest, identity, case, sequence, device):
    """Bind one native kinematic input, not a reference prediction, to all models.

    This removes backend-dependent repeated skinning differences while retaining
    each method's graph hierarchy, material assembly and complete native rollout.
    """
    import numpy as np
    import torch
    path=Path(manifest["output"])/"inputs/kinematics"/(case+".npz")
    if not path.exists():
        if identity!=manifest["reference_model"]:
            raise FileNotFoundError("Canonical native INPUT is not prepared: "+str(path))
        values={obj+"_"+key:_array(sequence[obj][key])
                for obj in ("cloth","obstacle") for key in ("prev_pos","pos","target_pos")}
        values.update(rest_pos=_array(sequence["cloth"].rest_pos),
            cloth_faces=_array(sequence["cloth"].faces_batch),
            obstacle_faces=_array(sequence["obstacle"].faces_batch),
            vertex_type=_array(sequence["cloth"].vertex_type))
        # An already saved, valid pilot input must not change under a cache repair.
        previous=Path(manifest["output"])/"inputs/context"/(case+".npz")
        if previous.exists():
            with np.load(previous) as z:
                for key,array in (("initial_pos",values["cloth_pos"][:,0]),
                    ("previous_pos",values["cloth_prev_pos"][:,0]),
                    ("obstacle",values["obstacle_target_pos"].transpose(1,0,2))):
                    if not np.array_equal(z[key],array):
                        raise ValueError("New canonical INPUT differs from the existing pilot input: "+key)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(".tmp.npz")
        np.savez_compressed(temp,**values);temp.replace(path)
        write(path.with_suffix(".json"),dict(state="prepared",source="native Pb4U dataloader BEFORE any prediction",
            sequence=manifest["sequences"][manifest["cases"][case]["motion"]]["converted"],
            mesh=manifest["cases"][case]["prepared_mesh"],seed=57,
            scope="common raw kinematics; not neural predictions or simulated cloth GT",created_at=now()))
    with np.load(path) as z:values={k:z[k] for k in z.files}
    if not np.array_equal(_array(sequence["cloth"].rest_pos),values["rest_pos"]):
        raise ValueError("Shared original rest geometry differs before kinematic binding")
    if not np.array_equal(_array(sequence["cloth"].vertex_type),values["vertex_type"]):
        raise ValueError("Shared handles differ before kinematic binding")
    backend=manifest["models"][identity]["backend"]
    differences={}
    for obj in ("cloth","obstacle"):
        keys={"prev_pos":"prev_pos","pos":"pos","lookup":"target_pos"} if backend=="contourcraft" else {k:k for k in ("prev_pos","pos","target_pos")}
        for native_key,source_key in keys.items():
            wanted=values[obj+"_"+source_key]
            if backend=="contourcraft" and native_key!="lookup":wanted=wanted[:,0]
            current=sequence[obj][native_key]
            if tuple(current.shape)!=wanted.shape:raise ValueError("Native input layout mismatch: "+obj+"/"+native_key)
            differences[obj+"/"+native_key]=float(np.max(np.abs(_array(current)-wanted)))
            sequence[obj][native_key]=torch.as_tensor(wanted,device=device,dtype=current.dtype).clone()
    return dict(source=str(path),source_kind="native kinematic INPUT; no neural output or GT reset",
        raw_native_max_absolute_difference_m=differences,common_input_arrays_bound=True)


def build(manifest, identity, case, output, device):
    """Construct the unchanged native model with this manifest's explicit input."""
    import torch
    record, spec = manifest["models"][identity], input_spec(manifest, case, identity)
    item = manifest["cases"][case]
    backend = record["backend"]
    if backend != "shared_cell":
        runner, sequence, config = build_native_case(backend, Path(manifest["library_root"]), output,
            item["motion"], item["garment"], item["level"], device, input_spec=spec)
    else:
        source = Path(manifest["source_work"])
        sys.path.insert(0, str(source / 'arcon/control'))
        import arcon.control.backbone_stage as course
        args = SimpleNamespace(project=Path(manifest["library_root"])/"external/PB4U-GNet-main",
            data=Path(manifest["library_root"])/"data/pb4u_official", output=output,
            arm=record["arm"], iterations=record["step"], device=device, seed=57)
        torch.cuda.set_device(device)
        _seed()
        course.shared.install(args)
        from utils.validation import Config, update_config_for_validation, load_runner, create_one_sequence_dataloader
        from utils.common import move2device
        if record["source"] in ("project_gap_readout_2500", "project_fixed_contact_readout_2500", "project_coarse_contact_short_2500", "project_facefloor_phase_course", "project_contact_query_continuation", "project_contact_query_20k_continuation", "project_contact_query_followup", "project_eq_surface_continuation", "project_material_response_screen"):
            if record["source"] == "project_material_response_screen":
                import arcon.control.material_stage as readout_course
            elif record["source"] == "project_eq_surface_continuation":
                import arcon.control.surface_variants as readout_course
            elif record["source"] == "project_contact_query_followup":
                import arcon.control.contact_query_followup as readout_course
            elif record["source"] == "project_contact_query_20k_continuation":
                import arcon.control.contact_query_stage as readout_course
            elif record["source"] == "project_contact_query_continuation":
                import arcon.control.contact_query_variants as readout_course
            elif record["source"] == "project_facefloor_phase_course":
                import arcon.control.contact_feedback_stage as readout_course
            elif record["source"] == "project_coarse_contact_short_2500":
                import arcon.control.contact_support_stage as readout_course
            elif record["source"] == "project_fixed_contact_readout_2500":
                import arcon.control.contact_readout_variants as readout_course
            else:
                import arcon.control.gap_readout_variants as readout_course
            modules, config = readout_course.configuration(args, record["arm"], device, output, training=False)
        else:
            modules, config = course.configuration(args, record["arm"], device, output, training=False)
        config = update_config_for_validation(config, Config(**MATERIAL, separate_arms=True,
            garment_dict_file=spec["mesh_path"], smpl_model="smpl/SMPL_FEMALE.pkl"))
        _, runner = load_runner(modules, config)
        payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
        if record["source"] in ("project_gap_readout_2500", "project_fixed_contact_readout_2500", "project_coarse_contact_short_2500", "project_facefloor_phase_course", "project_contact_query_continuation", "project_contact_query_20k_continuation", "project_contact_query_followup", "project_eq_surface_continuation", "project_material_response_screen"):
            expected = (readout_course.phase.protocol("Face-Early")
                        if record.get("zero_update_diagnostic") else readout_course.protocol(record["arm"]))
        elif record["source"] == "project_approved_continuation":
            import arcon.control.backbone_continuation as continuation
            expected = continuation.protocol(record["arm"])
        else:
            expected = course.protocol(record["arm"])
        if payload["global_step"] != record["step"] or payload["architecture_protocol"] != expected:
            raise ValueError("Checkpoint identity/protocol does not match manifest")
        runner.load_state_dict(payload["training_module"], strict=True)
        if record["source"] in ("project_gap_readout_2500", "project_fixed_contact_readout_2500", "project_coarse_contact_short_2500", "project_facefloor_phase_course", "project_contact_query_continuation", "project_contact_query_20k_continuation", "project_contact_query_followup", "project_eq_surface_continuation", "project_material_response_screen"):
            readout_course.configure_readout(runner, record["arm"])
        del payload
        runner.to(device).eval().requires_grad_(False)
        course.shared.recording(runner.model, False)
        runner.model._learned_model.capture_stages = False
        runner.cloth_obj.cache = defaultdict(dict)
        _seed()
        loader=create_one_sequence_dataloader(spec["sequence"],spec["mesh_key"],modules,config)
        sequence=move2device(next(iter(loader)),device)
        _metadata(runner,sequence,config,checkpoint=record["checkpoint"], source_root=str(source),
            source_initialization_frames=0, motion_source=spec["sequence"],
            garment_asset=dict(source=spec["mesh_path"], key=spec["mesh_key"], geometry_modified=False),
            initialization_policy="unchanged native Pb4U runner", identity=identity,
            architecture_protocol=expected)
    meta=runner._garment_common_metadata
    if hasattr(runner.model,"propagation_distance"):
        rest=sequence["cloth"].rest_pos
        edges=sequence["cloth","mesh_edge","cloth"].edge_index
        mean_edge=float((rest[edges[1]]-rest[edges[0]]).norm(dim=-1).mean())
        meta["native_repeat"]=max(1,int(runner.model.propagation_distance/mean_edge))
        meta["native_repeat_source"]="unchanged native floor(propagation_distance/mean_rest_edge)"
    length=(sequence["obstacle"].lookup.shape[1] if backend=="contourcraft"
            else sequence["obstacle"].pos.shape[1])
    if length!=item["horizon"]:
        raise ValueError(f"Native complete time axis differs: {length} vs {item['horizon']}")
    meta["common_kinematic_input"]=canonical_inputs(manifest,identity,case,sequence,device)
    return runner,sequence,config


def indices(horizon):
    return sorted({max(1,math.ceil(horizon*f)-1) for f in (.25,.5,.75,1.)})


class FullCapture(_NativeCapture):
    def __init__(self,horizon,record_prestates=False):
        super().__init__()
        self.record_prestates=record_prestates
        self.requested=horizon
        self.cost_indices=indices(horizon)
        self.prestates=[]
        self.native_losses=[]
        self.first_nonfinite=None

    def before(self,module,inputs):
        i=len(self.predictions)
        super().before(module,inputs)
        if self.record_prestates and i in self.cost_indices:
            c=inputs[0]["cloth"]
            self.prestates.append(dict(index=i,pos=_array(c.pos),prev_pos=_array(c.prev_pos),velocity=_array(c.velocity)))

    def after(self,module,inputs,state):
        import numpy as np
        super().after(module,inputs,state)
        if not np.isfinite(self.predictions[-1]).all():
            self.first_nonfinite=len(self.predictions)-1
            raise FloatingPointError(f"Nonfinite model endpoint at prediction index {self.first_nonfinite}")


def compare_input(manifest,identity,case,context,body):
    import numpy as np
    reference=Path(manifest["output"])/"inputs/context"/(case+".npz")
    if not reference.exists():raise FileNotFoundError("Complete input reference missing: "+str(reference))
    differences={}
    with np.load(reference) as z:
        for key in ("rest_pos","initial_pos","previous_pos","initial_obstacle","previous_obstacle"):
            a,b=context[key],z[key]
            if a.shape!=b.shape:raise ValueError("Shared input shape differs: "+key)
            delta=float(np.max(np.abs(a-b)));differences[key]=delta
            if delta>1e-6:raise ValueError(f"Shared native input differs {key}: {delta}")
        for key in ("cloth_faces","obstacle_faces"):
            # Face order alone may differ between native loaders; vertex
            # correspondence and oriented indexed triangles may not.
            def oriented_faces(faces):
                offsets=(np.argmin(faces,axis=1)[:,None]+np.arange(3))%3
                faces=np.take_along_axis(faces,offsets,axis=1)
                return faces[np.lexsort(faces.T[::-1])]
            a,b=oriented_faces(context[key]),oriented_faces(z[key])
            if not np.array_equal(a,b):raise ValueError("Shared indexed surface differs: "+key)
        if body.shape!=z["obstacle"].shape:raise ValueError("Shared complete body axis differs")
        differences["body_max_abs_m"]=float(np.max(np.abs(body-z["obstacle"])))
        if differences["body_max_abs_m"]>1e-6:raise ValueError("Shared full native body driver differs")
    return dict(state="complete",reference=str(reference),direct_max_absolute_m=differences)


def benchmark(manifest,identity,case,runner,sequence,output,device):
    import numpy as np
    import torch
    from utils.common import add_field_to_pyg_batch
    backend=manifest["models"][identity]["backend"]
    reference=case_dir(manifest["output"],manifest["reference_model"],case)/"common_prestates.npz"
    if not reference.is_file():raise FileNotFoundError(reference)
    sequence=add_field_to_pyg_batch(sequence,"iter",[0],"cloth",reference_key=None)
    with (torch.enable_grad() if backend=="contourcraft" else torch.no_grad()):
        if backend=="contourcraft":
            sequence=runner._add_cloth_obj(sequence)
            sequence=runner.prepare_sample(sequence)
            first=runner.collect_sample(sequence,0,None)
            first,sequence=runner.update_sample_1st_step(first,sequence)
            del first
        elif backend=="fine15":
            sequence=runner._add_cloth_obj(sequence);sequence["cloth"].cloth_obj=runner.cloth_obj
        else:
            sequence=runner.add_cloth_obj(sequence)
    with np.load(reference) as z:
        states={key:z[key] for key in ("index","pos","prev_pos","velocity")}
    rows=[]
    for j,index in enumerate(states["index"]):
        q=torch.as_tensor(states["pos"][j],device=device)
        prev=torch.as_tensor(states["prev_pos"][j],device=device)
        velocity=torch.as_tensor(states["velocity"][j],device=device)
        def prepared():
            if backend=="contourcraft":sample=runner.collect_sample(sequence,int(index),None)
            elif backend=="fine15":sample=runner.collect_sample_wholeseq(sequence,int(index),int(index),None)
            else:sample=runner.collect_sample_wholeseq(sequence,int(index),None)
            sample["cloth"].pos=q.clone();sample["cloth"].prev_pos=prev.clone();sample["cloth"].velocity=velocity.clone()
            return sample
        def step(sample):
            return runner.model(sample) if backend=="contourcraft" else runner.model(sample,is_training=False)
        measurements=[]
        with (torch.enable_grad() if backend=="contourcraft" else torch.no_grad()):
            warm=step(prepared());torch.cuda.synchronize();del warm
            for repeat in range(3):
                torch.cuda.synchronize();begin=time.perf_counter()
                sample=prepared();torch.cuda.synchronize();prep_ms=(time.perf_counter()-begin)*1000
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();model_start=time.perf_counter();predicted=step(sample);b.record();b.synchronize()
                model_ms=(time.perf_counter()-model_start)*1000
                measurements.append(dict(preparation_wall_ms=prep_ms,model_wall_ms=model_ms,
                    online_wall_ms=(time.perf_counter()-begin)*1000,model_cuda_ms=a.elapsed_time(b),
                    finite=bool(torch.isfinite(predicted["cloth"].pred_pos).all())))
                del predicted,sample
        rows.append(dict(prediction_index=int(index),measurements=measurements))
    if len(rows)!=4:raise ValueError("Complete relative-time cost states missing")
    result=dict(state="complete",model=identity,case=case,reference=str(reference),rows=rows,
        means={key:finite_mean([x[key] for r in rows for x in r["measurements"]])
               for key in ("preparation_wall_ms","model_wall_ms","online_wall_ms","model_cuda_ms")},
        warmups_per_state=1,repeats_per_state=3,gpu=torch.cuda.get_device_name(),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        scope=manifest["cost"],finished_at=now())
    write(output/"online_cost.json",result)
    return result


def infer(args):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    manifest=read(args.manifest);item=manifest["cases"][args.case]
    output=case_dir(manifest["output"],args.model,args.case)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"summary.json").exists():raise FileExistsError("Preserve the earlier attempt: "+str(output))
    summary=dict(state="building",model=args.model,case=args.case,protocol=manifest["protocol"],
        checkpoint=manifest["models"][args.model]["checkpoint"],expected_frames=item["horizon"],frames=0,
        started_at=now(),control_revision=args.control_revision,complete_sequence=False,
        material=manifest["material"],scope="Complete native autoregressive rollout; no reset or horizon cap")
    write(output/"summary.json",summary)
    def interrupted(signum, frame):
        raise TimeoutError(f"Terminated by signal {signum}; preserve captured frames")
    signal.signal(signal.SIGTERM,interrupted)
    capture=FullCapture(item["horizon"],args.model==manifest["reference_model"]);handles=[];runner=None;before=None;context=None;start=time.monotonic()
    try:
        torch.set_num_threads(4)
        runner,sequence,cfg=build(manifest,args.model,args.case,output,args.device)
        cost_sequence=sequence.clone()
        context=_sequence_context(sequence)
        if len(context["rest_pos"])!=item["vertices"]:raise ValueError("Manifest original mesh differs")
        if args.model==manifest["reference_model"]:
            target=Path(manifest["output"])/"inputs/context"/(args.case+".npz")
            target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,**context,
                obstacle=np.transpose(_array(sequence["obstacle"].target_pos),(1,0,2)))
        meta=runner._garment_common_metadata
        summary.update(state="running",provenance=meta,native_repeat=meta["native_repeat"],
            config=OmegaConf.to_container(cfg,resolve=True),setup_seconds=time.monotonic()-start)
        write(output/"summary.json",summary)
        before=normalizer_snapshot(runner)
        handles=[runner.model.register_forward_pre_hook(capture.before),runner.model.register_forward_hook(capture.after)]
        torch.cuda.reset_peak_memory_stats()
        backend=manifest["models"][args.model]["backend"]
        begin=time.monotonic()
        # -1 means the WHOLE native sequence in every supported author runner.
        with (torch.enable_grad() if backend=="contourcraft" else torch.no_grad()):
            trajectory=runner.valid_rollout(sequence,n_steps=-1,bare=False)
        torch.cuda.synchronize();summary["rollout_seconds"]=time.monotonic()-begin
        for h in handles:h.remove()
        handles=[]
        prefix=int(meta["source_initialization_frames"])
        native_pred,native_body=_array(trajectory["pred"]),_array(trajectory["obstacle"])
        pred=native_pred[prefix:];returned_body=native_body[prefix:]
        body=np.stack(capture.bodies)
        if len(pred)!=item["horizon"] or len(pred)!=len(capture.predictions):
            raise ValueError("Native full trajectory length does not match manifest")
        if not np.isfinite(pred).all() or not np.isfinite(body).all():raise FloatingPointError("Nonfinite geometry")
        _save_rollout(output,context,capture,pred,body,native_obstacle=returned_body)
        if prefix:
            np.savez_compressed(output/"native_initialization.npz",pred=native_pred[:prefix],obstacle=native_body[:prefix])
        # Persist trajectory before any reporting/metric failure.
        native_metrics=_json_value(dict(trajectory["metrics"]))
        losses,key_map=_loss_series(native_metrics,len(pred))
        invalid={key:[i for i,x in enumerate(v) if not math.isfinite(x)] for key,v in losses.items()}
        invalid={k:v for k,v in invalid.items() if v}
        np.savez_compressed(output/"native_losses.npz",**{k:np.asarray(v) for k,v in losses.items()})
        clean={k:[v if math.isfinite(v) else None for v in values] for k,values in losses.items()}
        write(output/"native_losses.json",dict(values=clean,key_map=key_map,nonfinite_indices=invalid,
            convention="unchanged method-native runner values; NOT the common-six score"))
        pst=capture.prestates
        if capture.record_prestates:
            np.savez_compressed(output/"common_prestates.npz",index=np.array([x["index"] for x in pst]),
                **{k:np.stack([x[k] for x in pst]) for k in ("pos","prev_pos","velocity")})
        input_comparison=compare_input(manifest,args.model,args.case,context,body)
        write(output/"input_correspondence.json",input_comparison)
        summary.update(state="rollout_complete",frames=len(pred),pred_shape=list(pred.shape),
            complete_sequence=True,native_loss_nonfinite_indices=invalid,
            prediction_pose_indices=[2,len(pred)+1],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            native_initialization_frames=prefix)
        write(output/"summary.json",summary)
        if manifest.get("cost", {}).get("enabled", True):
            benchmark(manifest,args.model,args.case,runner,cost_sequence,output,args.device)
        else:
            summary["online_cost_scope"] = "Not requested; anomalous common-prestate timing is not deployment cost"
        comparison=normalizer_comparison(before,normalizer_snapshot(runner))
        if comparison["changed"]:raise RuntimeError("Inference modified checkpoint normalizers")
        summary.update(state="complete",normalizer=comparison,finished_at=now(),elapsed_seconds=time.monotonic()-start)
        write(output/"summary.json",summary)
        print(json.dumps({k:summary[k] for k in ("state","model","case","frames","rollout_seconds")}),flush=True)
    except BaseException as exc:
        for handle in handles:handle.remove()
        summary.update(state="failed",error=f"{type(exc).__name__}: {exc}",captured_frames=len(capture.predictions),
            first_nonfinite_prediction_index=capture.first_nonfinite,finished_at=now(),elapsed_seconds=time.monotonic()-start)
        if context is not None and capture.predictions and not (output/"rollout.npz").exists():
            _save_rollout(output,context,capture,np.stack(capture.predictions),np.stack(capture.bodies[:len(capture.predictions)]))
        if runner is not None and before is not None:
            summary["normalizer"]=normalizer_comparison(before,normalizer_snapshot(runner))
        write(output/"summary.json",summary)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--model",required=True);p.add_argument("--case",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--control-revision",required=True)
    infer(p.parse_args())

if __name__=="__main__":main()
