#!/usr/bin/env python3
"""Native physical-objective measurement on saved complete trajectories."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from arcon.control.evaluation_protocol import MATERIAL, SIX, case_dir, finite_mean, now, read, write


def load_complete(manifest,model,case):
    directory=case_dir(manifest["output"],model,case)
    summary=read(directory/"summary.json")
    if not summary.get("complete_sequence"):
        raise ValueError("No complete trajectory; failed case remains in denominator: "+str(directory))
    if not (directory/"input_correspondence.json").exists():
        raise ValueError("Common input correspondence is not established")
    with np.load(directory/"rollout.npz") as z:
        data={key:z[key] for key in z.files}
    expected=manifest["cases"][case]["horizon"]
    if len(data["pred"])!=expected or not np.isfinite(data["pred"]).all():
        raise ValueError("Incomplete/nonfinite full trajectory")
    return directory,data,summary


def describe(values, *, allow_undefined_support=False):
    valid=[v for v in values if v is not None and math.isfinite(v)]
    undefined=sum(v is None for v in values)
    nonfinite=sum(v is not None and not math.isfinite(v) for v in values)
    mean=(math.fsum(valid)/len(valid) if valid and not nonfinite and
          (allow_undefined_support or not undefined) else None)
    return dict(mean=mean,defined_frames=len(valid),undefined_support_frames=undefined,
        nonfinite_frames=nonfinite,total_frames=len(values),
        max=max(valid) if valid else None,p95=float(np.percentile(valid,95)) if valid else None)


def common_physics(args,manifest):
    """Evaluate all methods with the SAME native Pb4U six criteria/time convention.

    No neural forward. Native model-specific losses stay in native_losses.json.
    Prestate is reconstructed from this method's OWN saved trajectory using the
    author collector, initialization, handle handling and material assembly.
    """
    import torch
    from arcon.control.trajectory_evaluation import build
    torch.set_num_threads(4)
    directory,data,summary=load_complete(manifest,args.model,args.case)
    destination=directory/"common_six.json"
    if destination.exists():raise FileExistsError(destination)
    start=time.monotonic()
    runner,sequence,_=build(manifest,manifest["reference_model"],args.case,directory/"common_physics_input",args.device)
    from utils.common import add_field_to_pyg_batch
    sequence=add_field_to_pyg_batch(sequence,"iter",[0],"cloth",reference_key=None)
    series={k:[] for k in SIX};previous=None
    with torch.no_grad():
        sequence=runner.add_cloth_obj(sequence)
        for i,prediction in enumerate(data["pred"]):
            state=runner.collect_sample_wholeseq(sequence,i,previous)
            if i==0:state=runner.collision_solver.solve(state)
            state=runner.model.replace_pinned_verts(state)
            # Register PyG slice/inc metadata too: the original criteria call
            # get_example(), which cannot see a plain newly assigned field.
            endpoint=torch.as_tensor(prediction,device=args.device)
            state=add_field_to_pyg_batch(state,"pred_pos",endpoint,"cloth","pos")
            state=add_field_to_pyg_batch(state,"pred_velocity",endpoint-state["cloth"].pos,"cloth","pos")
            losses=runner.criterion_pass(state)
            for key in SIX:series[key].append(float(losses[key]))
            previous=state
    nonfinite={k:[i for i,x in enumerate(v) if not math.isfinite(x)] for k,v in series.items()}
    nonfinite={k:v for k,v in nonfinite.items() if v}
    np.savez_compressed(directory/"common_six_values.npz",**{k:np.asarray(v) for k,v in series.items()})
    clean={k:[x if math.isfinite(x) else None for x in v] for k,v in series.items()}
    means={k:finite_mean(v) for k,v in clean.items()}
    consistency=None
    if args.model==manifest["reference_model"] or manifest["models"][args.model]["backend"]=="shared_cell":
        native=read(directory/"native_losses.json")["values"]
        consistency={k:dict(common_mean=means[k],native_mean=finite_mean(native[k]),
            max_absolute_difference=(max(abs(a-b) for a,b in zip(series[k],native[k]))
                if all(x is not None for x in native[k]) and not nonfinite else None)) for k in SIX}
    write(destination,dict(state="complete" if not nonfinite else "nonfinite_objective",model=args.model,
        case=args.case,frames=len(data["pred"]),values=clean,means=means,
        total=math.fsum(means.values()) if all(x is not None for x in means.values()) else None,
        nonfinite_indices=nonfinite,source="original Pb4U criteria, eval iter=0, collision weight5000",
        inertia_history="own saved position history; native collector and handle handling, not a teacher-forced state",
        temporal_convention=dict(initial_ts=1/3,regular_ts=1/30),
        native_compatibility=consistency,neural_forward_calls=0,finished_at=now(),elapsed_seconds=time.monotonic()-start))




def collision_vertex_terms(criterion, example):
    """Original Pb4U calc_loss before its final sum; same query and arithmetic."""
    import torch
    from pytorch3d.ops import knn_points
    from utils.common import gather
    body_now=example["obstacle"].pos
    body_next=example["obstacle"].target_pos
    faces=example["obstacle"].faces_batch.T
    current=example["cloth"].pos
    endpoint=example["cloth"].pred_pos
    centers_now=gather(body_now,faces,0,1,1).mean(dim=-2)
    _,indices,_=knn_points(current.unsqueeze(0),centers_now.unsqueeze(0),return_nn=True)
    indices=indices[0]
    centers_next=gather(body_next,faces,0,1,1).mean(dim=-2)
    normals=criterion.f_normals_f(body_next.unsqueeze(0),faces.unsqueeze(0))[0]
    points=gather(centers_next,indices,0,1,1)[:,0]
    directions=gather(normals,indices,0,1,1)[:,0]
    distance=((endpoint-points)*directions).sum(dim=-1)
    penetration=torch.maximum(criterion.eps-distance,torch.FloatTensor([0]).to(endpoint.device))
    return penetration.pow(3)


def common_physics_free_vertices(args, manifest):
    """Replay saved states; change only the collision summation vertex set.

    The original six-term cache and trajectories remain read-only. The other
    five terms are copied exactly, not re-evaluated. The all-vertex replay and
    free/HANDLE partition are checked against the saved original collision.
    """
    import torch
    from arcon.control.trajectory_evaluation import build
    torch.set_num_threads(4)
    directory, data, _ = load_complete(manifest, args.model, args.case)
    original = read(directory / "common_six.json")
    horizon = len(data["pred"])
    if original.get("frames") != horizon:
        raise ValueError("Original physics cache has a different time axis")
    destination = case_dir(args.result_root, args.model, args.case) / "common_six_free_vertices.json"
    if destination.exists():
        raise FileExistsError(destination)
    start = time.monotonic()
    context = Path(args.result_root) / "scoring_context" / args.model / args.case
    runner, sequence, _ = build(manifest, manifest["reference_model"], args.case, context, args.device)
    from utils.common import NodeType, add_field_to_pyg_batch
    candidates = [v for k, v in runner.criterion_dict.items()
                  if k == "collision_penalty" or getattr(v, "name", None) == "collision_penalty"]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one original collision criterion")
    criterion = candidates[0]
    sequence = add_field_to_pyg_batch(sequence, "iter", [0], "cloth", reference_key=None)
    free_values, handle_values, all_values = [], [], []
    counts = None
    native_spot_checks = []
    previous = None
    with torch.no_grad():
        sequence = runner.add_cloth_obj(sequence)
        for i, prediction in enumerate(data["pred"]):
            state = runner.collect_sample_wholeseq(sequence, i, previous)
            if i == 0:
                state = runner.collision_solver.solve(state)
            state = runner.model.replace_pinned_verts(state)
            endpoint = torch.as_tensor(prediction, device=args.device)
            state = add_field_to_pyg_batch(state, "pred_pos", endpoint, "cloth", "pos")
            state = add_field_to_pyg_batch(state, "pred_velocity", endpoint-state["cloth"].pos, "cloth", "pos")
            if state.num_graphs != 1:
                raise ValueError("Complete-trajectory replay expects one graph")
            example = state.get_example(0)
            free = example["cloth"].vertex_type.reshape(-1) != NodeType.HANDLE
            current_counts = dict(free=int(free.sum()), handle=int((~free).sum()), total=len(free))
            if counts is not None and current_counts != counts:
                raise ValueError("Vertex classification changed within the trajectory")
            counts = current_counts
            weight = criterion.get_weight(state["cloth"].iter[0].item())
            terms = collision_vertex_terms(criterion,example)
            all_value = float(terms.sum(-1)*weight)
            partition = [float(terms[mask].sum(-1)*weight) for mask in (free,~free)]
            if i in (0,horizon//2,horizon-1):
                native = float(criterion.calc_loss(example)*weight)
                if not math.isclose(all_value,native,rel_tol=2e-5,abs_tol=1e-6):
                    raise ValueError(f"Per-vertex field differs from original criterion at endpoint {i+1}")
                native_spot_checks.append(dict(endpoint=i+1,native=native,per_vertex_sum=all_value))
            saved = original["values"]["collision_penalty_loss"][i]
            if saved is None or not all(math.isfinite(v) for v in (all_value, *partition)):
                raise ValueError(f"Nonfinite collision at endpoint {i+1}")
            # Floating-point agreement checks, not physical acceptance thresholds.
            if not math.isclose(all_value, saved, rel_tol=2e-5, abs_tol=1e-6):
                raise ValueError(f"Original collision replay differs at endpoint {i+1}: {all_value} vs {saved}")
            if not math.isclose(math.fsum(partition), all_value, rel_tol=2e-5, abs_tol=1e-6):
                raise ValueError(f"Free/HANDLE collision partition differs at endpoint {i+1}")
            free_values.append(partition[0]); handle_values.append(partition[1]); all_values.append(all_value)
            previous = state
    values = {key: original["values"][key] for key in SIX}
    values["collision_penalty_loss"] = free_values
    means = dict(original["means"])
    means["collision_penalty_loss"] = finite_mean(free_values)
    total = math.fsum(means.values()) if all(v is not None and math.isfinite(v) for v in means.values()) else None
    source_values = original["values"]["collision_penalty_loss"]
    write(destination, dict(state="complete" if total is not None else "nonfinite_objective",
        protocol="free_vertex_collision_20261116", collision_vertices="vertex_type != HANDLE",
        model=args.model, case=args.case, frames=horizon, vertex_counts=counts,
        values=values, means=means, total=total, original_means=original["means"], original_total=original["total"],
        original_collision_values=source_values, replayed_all_vertex_collision=all_values,
        handle_collision_values=handle_values,
        replay_max_absolute_difference=max(abs(a-b) for a,b in zip(all_values, source_values)),
        partition_max_absolute_difference=max(abs(a-b-c) for a,b,c in zip(all_values, free_values, handle_values)),
        source_cache=str(directory/"common_six.json"), source_rollout=str(directory/"rollout.npz"),
        temporal_convention=original.get("temporal_convention"),
        unchanged_five_terms=[k for k in SIX if k != "collision_penalty_loss"],
        evaluation="single original-formula vertex field partitioned before summation",
        original_criterion_spot_checks=native_spot_checks,
        neural_forward_calls=0, training_changed=False, geometry_scores_changed=False,
        finished_at=now(), elapsed_seconds=time.monotonic()-start))

def reference_regions(manifest,case,device):
    from arcon.control.surface_measurements import SurfaceDistance
    fixed = manifest.get('reference_regions_root')
    if fixed is not None:
        path = Path(fixed) / case / 'contact_regions.npz'
        with np.load(path) as z:
            return z['near_nodes']
    directory,data,_=load_complete(manifest,manifest["reference_model"],case)
    path=directory/"contact_regions.npz"
    if path.exists():
        with np.load(path) as z:return z["near_nodes"]
    faces=data["cloth_faces"]
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    surface=SurfaceDistance(data["native_initial_obstacle"],data["obstacle_faces"],device)
    masks=[]
    for i in range(len(data["pred"])):
        before=data["native_initial_pos"] if i==0 else data["pred"][i-1]
        body=data["native_initial_obstacle"] if i==0 else data["obstacle"][i-1]
        surface.update(body)
        seeds=surface.query(before)<.002
        near=seeds.copy()
        near[edges[seeds[edges[:,0]],1]]=True
        near[edges[seeds[edges[:,1]],0]]=True
        masks.append(near)
    masks=np.asarray(masks,dtype=bool)
    np.savez_compressed(path,near_nodes=masks,threshold_m=.002,mesh_hops=1)
    return masks


def geometry(args,manifest):
    import torch
    torch.set_num_threads(4)
    import warp as wp
    from arcon.control.surface_measurements import SurfaceDistance
    from arcon.control.quality_summary import weights_and_areas, rest_face_inverse, intrinsic_green, persistence
    wp.init()
    start=time.monotonic()
    directory,data,_=load_complete(manifest,args.model,args.case)
    target=directory/"geometry_quality.json"
    if target.exists():raise FileExistsError(target)
    faces=data["cloth_faces"];rest=data["rest_pos"].astype(np.float64)
    weights,areas=weights_and_areas(rest,faces)
    inverse=rest_face_inverse(rest,faces)
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    lengths=np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1)
    near=reference_regions(manifest,args.case,args.device)
    if near.shape!=data["pred"].shape[:2]:raise ValueError("Full common-neighborhood axis differs")
    surface=SurfaceDistance(data["obstacle"][0],data["obstacle_faces"],args.device)
    series={k:[] for k in ("penetration_fraction","mean_penetration_depth","max_penetration_depth",
        "contact_neighbor_green_strain_rms","contact_neighbor_edge_stretch_rms","green_strain_area_rms",
        "edge_stretch_rms","reference_near_rest_area_fraction")}
    masks=[]
    for i,prediction in enumerate(data["pred"]):
        surface.update(data["obstacle"][i]);depth=np.maximum(-surface.query(prediction),0.)
        active=depth>1e-6;masks.append(active)
        green=intrinsic_green(prediction.astype(np.float64)[None],faces,inverse)[0]
        square=np.sum(green*green,axis=(1,2))
        stretch=np.linalg.norm(prediction[edges[:,1]].astype(np.float64)-prediction[edges[:,0]],axis=1)/lengths-1
        nf=near[i][faces].any(-1);ne=near[i][edges].any(-1)
        values=dict(penetration_fraction=float(active@weights),mean_penetration_depth=float(depth@weights),
            max_penetration_depth=float(depth.max()),green_strain_area_rms=float(np.sqrt(square@areas/areas.sum())),
            edge_stretch_rms=float(np.sqrt(np.mean(stretch**2))),reference_near_rest_area_fraction=float(near[i]@weights),
            contact_neighbor_green_strain_rms=float(np.sqrt(square[nf]@areas[nf]/areas[nf].sum())) if nf.any() else None,
            contact_neighbor_edge_stretch_rms=float(np.sqrt(np.mean(stretch[ne]**2))) if ne.any() else None)
        for key,value in values.items():series[key].append(value)
    dt=np.full(len(data["pred"]),1/30);dt[0]=1/3
    persistence_info,persistent=persistence(np.asarray(masks),weights,dt)
    write(target,dict(state="complete",model=args.model,case=args.case,frames=len(data["pred"]),values=series,
        statistics={k:describe(v,allow_undefined_support=k.startswith("contact_neighbor")) for k,v in series.items()},
        body_persistence=persistence_info,body_persistence_series=persistent,threshold_m=1e-6,
        neighborhood=manifest["reference_regions"],finished_at=now(),elapsed_seconds=time.monotonic()-start))


def self_collision(args,manifest):
    # A separate process/CC environment keeps its native utils namespace intact.
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    source=Path(manifest["library_root"])/"external/ContourCraft-main"
    sys.path.insert(0,str(source))
    from utils.selfcollisions import find_close_faces
    from arcon.control.quality_summary import weights_and_areas, persistence
    start=time.monotonic()
    directory,data,_=load_complete(manifest,args.model,args.case)
    target=directory/"self_quality.json"
    if target.exists():raise FileExistsError(target)
    face_array=data["cloth_faces"].astype(np.int64)
    faces=torch.as_tensor(face_array,device=args.device)
    _,area=weights_and_areas(data["rest_pos"].astype(np.float64),face_array);area/=area.sum()
    affected=[];series={k:[] for k in ("native_face_pair_count","unique_nonadjacent_face_pair_count","affected_rest_area_fraction")}
    for q in data["pred"]:
        vertices=torch.as_tensor(q,device=args.device,dtype=torch.float64)
        pairs=find_close_faces(vertices,faces,threshold=0.).detach().cpu().numpy()
        native_count=len(pairs)
        if len(pairs):
            pairs=np.unique(np.sort(pairs,axis=1),axis=0)
            adjacent=(face_array[pairs[:,0],:,None]==face_array[pairs[:,1],None,:]).any(axis=(1,2))
            pairs=pairs[~adjacent]
        mask=np.zeros(len(face_array),dtype=bool)
        if len(pairs):mask[np.unique(pairs)]=True
        affected.append(mask)
        series["native_face_pair_count"].append(native_count)
        series["unique_nonadjacent_face_pair_count"].append(len(pairs))
        series["affected_rest_area_fraction"].append(float(area[mask].sum()))
    persisted,pseries=persistence(np.asarray(affected),area)
    series.update(pseries)
    write(target,dict(state="complete",model=args.model,case=args.case,frames=len(data["pred"]),values=series,
        statistics={k:describe(v) for k,v in series.items()},persistence=persisted,
        detector=dict(source=str(source/"utils/selfcollisions.py"),threshold=0.,dtype="float64",max_candidates_per_triangle=320),
        finished_at=now(),elapsed_seconds=time.monotonic()-start))


def density(args,manifest):
    from arcon.control.surface_measurements import SurfaceDistance
    from arcon.control.quality_summary import weights_and_areas, weighted_quantiles
    from arcon.control.evaluation_protocol import MOTIONS, GARMENTS
    import warp as wp
    wp.init()
    start=time.monotonic();rows=[]
    for garment in GARMENTS:
        motion=next(iter(MOTIONS))
        lo_case=f"{motion}_{garment}_Lv.1";hi_case=f"{motion}_{garment}_Lv.4"
        # Rest maps depend only on the declared original meshes, not method predictions.
        with np.load(Path(manifest["output"])/"inputs/context"/(lo_case+".npz")) as z:
            lr,lf=z["rest_pos"].astype(np.float64),z["cloth_faces"]
        with np.load(Path(manifest["output"])/"inputs/context"/(hi_case+".npz")) as z:
            hr,hf=z["rest_pos"].astype(np.float64),z["cloth_faces"]
        _,area=weights_and_areas(lr,lf);area/=area.sum()
        mapping=Path(manifest["output"])/"inputs/density_maps"/(garment+".npz")
        if mapping.exists():
            with np.load(mapping) as z:ids,bary,dist=z["face_ids"],z["barycentric"],z["distance"]
        else:
            surface=SurfaceDistance(hr,hf,args.device)
            ids,bary,dist=surface.material_coordinates(lr[lf].mean(1))
            if np.any(ids<0) or not np.isfinite(dist).all():raise ValueError("Rest-surface correspondence incomplete")
            mapping.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(mapping,face_ids=ids,barycentric=bary,distance=dist)
        for motion in MOTIONS:
            lo=f"{motion}_{garment}_Lv.1";hi=f"{motion}_{garment}_Lv.4"
            try:
                _,a,_=load_complete(manifest,args.model,lo);_,b,_=load_complete(manifest,args.model,hi)
            except (FileNotFoundError,ValueError) as exc:
                rows.append(dict(motion=motion,garment=garment,state="unavailable",reason=str(exc)))
                continue
            if a["obstacle"].shape!=b["obstacle"].shape or np.max(np.abs(a["obstacle"]-b["obstacle"]))>1e-6:
                raise ValueError("Cross-grid body/physical time driver differs")
            values=[]
            for l,h in zip(a["pred"],b["pred"]):
                x=l[lf].astype(np.float64).mean(1)
                y=np.einsum("fkc,fk->fc",h[hf[ids]].astype(np.float64),bary)
                values.append(float(np.sqrt(np.sum((x-y)**2,axis=1)@area)))
            rows.append(dict(motion=motion,garment=garment,state="complete",frames=len(values),
                position_consistency_rms_m=values,statistics=describe(values),mapping=str(mapping),
                rest_projection_distance_m=weighted_quantiles(dist,area)))
    complete=sum(r["state"]=="complete" for r in rows)
    write(Path(manifest["output"])/"reports"/(args.model+"_density.json"),dict(
        state="complete" if complete==16 else "partial",model=args.model,expected_pairs=16,complete_pairs=complete,
        rows=rows,scope="Full aligned sequences, fixed original Lv.1-to-Lv.4 rest-surface quadrature; consistency not GT error",
        mean=finite_mean([r.get("statistics",{}).get("mean") for r in rows]),finished_at=now(),elapsed_seconds=time.monotonic()-start))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("physics","physics-free","geometry","self","density"))
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--model",required=True)
    p.add_argument("--case");p.add_argument("--device",default="cuda:0")
    p.add_argument("--result-root",type=Path,help="Separate output tree for physics-free; raw caches are read-only")
    args=p.parse_args();manifest=read(args.manifest)
    if args.mode!="density" and args.case is None:p.error("--case is required")
    if args.mode=="physics-free" and args.result_root is None:p.error("--result-root is required for physics-free")
    functions={"physics":common_physics,"physics-free":common_physics_free_vertices,"geometry":geometry,"self":self_collision,"density":density}
    functions[args.mode](args,manifest)
    print(json.dumps(dict(state="complete",mode=args.mode,model=args.model,case=args.case)),flush=True)

if __name__=="__main__":main()
