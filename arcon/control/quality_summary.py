"""Trajectory-level summaries of the direct material and contact measurements."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import pickle
import time

import numpy as np

ARMS = ('N', 'S', 'R', 'A')
AVAILABLE_ARMS = ARMS + ('RF','RG','W','J','WJ','WJK','WJ_L','WJ_LS','Author')
CASES = [f'{s}_{g}_{r}' for s, g, r in itertools.product(
    ('01_01', '07_02'), ('tank', 'tshirt', 'longsleeve', 'dress'), ('0.5', '2.5'))]
WINDOWS = {'H4': (0, 4), 'H20': (0, 20), 'H40': (0, 40), 'H80': (0, 80),
           'F1_20': (0, 20), 'F21_40': (20, 40), 'F41_60': (40, 60), 'F61_80': (60, 80)}
SCALE_BINS = ((0., .75), (.75, 1.), (1., 1.25), (1.25, float('inf')))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def describe(values):
    a = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if not len(a):
        return {'count': 0, 'mean': None, 'median': None, 'p95': None, 'max': None}
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95)), 'max': float(a.max())}


def windowed(series, windows=WINDOWS):
    return {name: {k: describe(v[a:b]) for k, v in series.items()} for name, (a, b) in windows.items()}


def weights_and_areas(rest, faces):
    area = np.linalg.norm(np.cross(rest[faces[:, 1]] - rest[faces[:, 0]],
                                   rest[faces[:, 2]] - rest[faces[:, 0]]), axis=-1) / 2
    w = np.zeros(len(rest), dtype=np.float64)
    for k in range(3):
        np.add.at(w, faces[:, k], area / 3)
    return w / w.sum(), area


def weighted_quantiles(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    c = np.cumsum(w) / w.sum()
    result = {'min': float(v[0]), 'max': float(v[-1]), 'mean': float(v @ (w / w.sum()))}
    for q in (.05, .25, .5, .75, .95):
        result['p' + str(int(100*q)).zfill(2)] = float(np.interp(q, c, v))
    return result


def rms(vector, weights, mask=None):
    if mask is not None:
        weights = weights * mask
    den = float(weights.sum())
    return float(np.sqrt(np.sum(vector*vector, axis=-1) @ weights / den)) if den else None


def persistence(mask, weights, timesteps=None):
    """Consecutive violating recorded endpoints, with explicit interval durations."""
    current = np.zeros(mask.shape[1], dtype=np.int64)
    longest = current.copy()
    current_duration = np.zeros(mask.shape[1], dtype=np.float64)
    longest_duration = current_duration.copy()
    series = {'persistent3_area_fraction': [], 'persistent10_area_fraction': []}
    for t, active in enumerate(mask):
        current = np.where(active, current + 1, 0)
        longest = np.maximum(longest, current)
        if timesteps is not None:
            current_duration = np.where(active, current_duration + timesteps[t], 0.)
            longest_duration = np.maximum(longest_duration, current_duration)
        for k in (3, 10):
            series[f'persistent{k}_area_fraction'].append(float(weights[current >= k].sum()))
    metrics = {'max_consecutive_frames': int(longest.max()),
               'area_weighted_longest_frames': float(longest @ weights),
               'ever_violating_area_fraction': float(weights[longest > 0].sum())}
    for k in (3, 10, 20):
        metrics[f'ever_consecutive{k}_area_fraction'] = float(weights[longest >= k].sum())
    if timesteps is not None:
        metrics['max_violating_endpoint_interval_seconds'] = float(longest_duration.max())
        metrics['area_weighted_longest_endpoint_interval_seconds'] = float(longest_duration @ weights)
    return metrics, series


def receiver_scales(rest, edges):
    if edges.shape[0] == 2:
        edges = edges.T
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=-1)
    count = np.bincount(edges[:, 1], minlength=len(rest))
    local = np.bincount(edges[:, 1], weights=lengths, minlength=len(rest)) / np.maximum(count, 1)
    if np.any(local <= 0):
        raise ValueError('Positive receiver rest-edge scales are required by the native interface')
    return float(lengths.mean()), local


def saved_stage_compatibility(saved, index, weights, face_area, faces, inverse, near_nodes, surface):
    labels = ('stage5_q_hat','stage5_refined','stage10_q_hat','stage10_refined','q_hat','pred_pos')
    face_mask = near_nodes[faces].any(-1)
    selected_area = face_area * face_mask
    denominator = float(selected_area.sum())
    free = saved['q_t'][index].astype(np.float64) + saved['u_t'][index].astype(np.float64)
    snapshots = {}
    for key in labels:
        q = saved[key][index].astype(np.float64)
        depth = np.maximum(-surface.query(q), 0.)
        green = intrinsic_green(q[None], faces, inverse)[0]
        snapshots[key] = dict(body_area_fraction=float(weights[depth > 1e-6].sum()),
            body_mean_depth_m=float(depth @ weights), body_max_depth_m=float(depth.max()),
            near_green_rms=float(np.sqrt(np.sum(green*green, axis=(-1,-2)) @ selected_area / denominator))
                           if denominator else None,
            free_step_defect_square_m2=float(np.sum((q-free)**2,axis=-1) @ weights))
    links = []
    for before,after in zip(labels[:-1],labels[1:]):
        links.append(dict(before=before,after=after,
            delta={k:snapshots[after][k]-v if v is not None and snapshots[after][k] is not None else None
                   for k,v in snapshots[before].items()}))
    return dict(frame=int(saved['frame'][index]),snapshots=snapshots,links=links,
                region='fixed saved N prestate near nodes, incident rest faces',
                interpretation='rest-relative Green and free-motion defect;not GT or single-writeback causal ablation')


def garment_analysis(args):
    import warp as wp
    from arcon.control.surface_measurements import SurfaceDistance
    wp.init()
    start = time.monotonic()
    rows = []
    for case in args.cases or CASES:
        d = args.root / 'evals' / args.arm / case
        with np.load(d / 'rollout.npz') as z:
            pred, body = z['pred'], z['obstacle']
            rest, faces = z['rest_pos'].astype(np.float64), z['cloth_faces']
            body_faces, edges = z['obstacle_faces'], z['mesh_edges']
        w, areas = weights_and_areas(rest, faces)
        h, local_h = receiver_scales(rest, edges)
        ratio = local_h / h
        cost = read(d / 'summary.json')
        old = read(d / 'screen_metrics.json')['frames']
        with np.load(args.root / 'evals' / 'N' / case / 'fixed_prestate_contact_regions.npz') as z:
            near = z['near_nodes']
        surface = SurfaceDistance(body[0], body_faces, args.device)
        distances = np.empty(pred.shape[:2], dtype=np.float64)
        for t in range(len(pred)):
            surface.update(body[t])
            distances[t] = surface.query(pred[t])
        depth = np.maximum(-distances, 0.)
        active = depth > args.threshold
        dt = np.full(len(pred), args.regular_ts)
        dt[0] = args.initial_ts
        persisted, series = persistence(active, w, dt)
        series.update({'max_penetration_depth_m': depth.max(axis=1).tolist(),
                       'penetrating_rest_area_fraction': (active @ w).tolist(),
                       'mean_penetration_depth_m': (depth @ w).tolist()})
        for k in ['contact_neighbor_green_strain_rms', 'contact_neighbor_edge_stretch_rms',
                  'remote_step_rms_m', 'nontranslation_acceleration_rms',
                  'inertia_loss', 'bending_energy_loss', 'stretching_energy_loss',
                  'collision_penalty_loss', 'friction_energy_loss']:
            series[k] = old[k]
        bins = []
        for lo, hi in SCALE_BINS:
            mask = (ratio >= lo) & (ratio < hi)
            den = float(w[mask].sum())
            bseries = {'penetrating_fraction_of_bin': ((active[:, mask] @ w[mask]) / den).tolist() if den else [None]*len(pred),
                       'mean_depth_m_in_bin': ((depth[:, mask] @ w[mask]) / den).tolist() if den else [None]*len(pred)}
            bins.append({'local_over_global_range': [lo, hi if np.isfinite(hi) else None],
                         'rest_area_fraction': den, 'metrics': windowed(bseries)})
        stages, compatibility = [], []
        compatible = getattr(args, 'physical_stage_compatibility', False) or args.arm in ('RF','RG','R','W','J','WJ','WJK','WJ_L','WJ_LS')
        inverse = rest_face_inverse(rest, faces) if compatible else None
        # Author endpoints were saved without intermediate model fields; do not replay.
        if args.arm != 'Author' and not getattr(args, 'endpoint_only', False):
            with np.load(d / 'response_runtime.npz') as z:
                for j, frame in enumerate(z['frame']):
                    free = z['q_t'][j].astype(np.float64) + z['u_t'][j]
                    if inverse is not None:
                        surface.update(body[int(frame)-1])
                        compatibility.append(saved_stage_compatibility(z,j,w,areas,faces,inverse,near[int(frame)-1],surface))
                    for stage in (5, 10):
                        key = f'stage{stage}_q_hat'
                        if key not in z.files:
                            continue
                        before, after = z[key][j].astype(np.float64), z[f'stage{stage}_refined'][j].astype(np.float64)
                        delta = after - before
                        b0, b1 = before - free, after - free
                        stage_row = {'frame': int(frame), 'stage': stage,
                                     'increment_rms_m': rms(delta, w),
                                     'increment_rms_over_global_h': rms(delta/h, w),
                                     'increment_rms_over_receiver_h': rms(delta/local_h[:, None], w),
                                     'increment_rms_over_A_L0': rms(delta/.1, w),
                                     'free_step_defect_square_change_m2': float((np.sum(b1*b1-b0*b0, axis=-1)) @ w),
                                     'by_scale': []}
                        for lo, hi in SCALE_BINS:
                            mask = (ratio >= lo) & (ratio < hi)
                            den = float(w[mask].sum())
                            stage_row['by_scale'].append({'range': [lo, hi if np.isfinite(hi) else None],
                                'increment_rms_m': rms(delta, w, mask),
                                'free_step_defect_square_change_m2': float(np.sum(b1*b1-b0*b0, axis=-1)[mask] @ w[mask]/den) if den else None})
                        stages.append(stage_row)
        mean_near = np.sum(near*w, axis=0)
        row = {'case': case, 'arm': args.arm, 'vertices': len(rest), 'faces': len(faces),
               'windows': windowed(series), 'frames': series, 'persistence': persisted,
               'scale': {'global_mean_directed_rest_edge_m': h,
                         'receiver_local_h_m': weighted_quantiles(local_h, w),
                         'local_over_global_h': weighted_quantiles(ratio, w),
                         'global_readout_gain_over_local_gain': weighted_quantiles(1/ratio, w),
                         'mean_h_in_reference_near_m': float(local_h @ mean_near / mean_near.sum()) if mean_near.sum() else None,
                         'near_local_over_global_mean': float(ratio @ mean_near / mean_near.sum()) if mean_near.sum() else None,
                         'native_K': cost['native_repeat'],
                         'nominal_K_times_global_h_m': float(cost['native_repeat']*h),
                         'nominal_K_times_local_h_m': weighted_quantiles(local_h*cost['native_repeat'], w),
                         'scale_bins': bins},
               'saved_response_stages': stages, 'stage_compatibility': compatibility,
               'stage_evidence': ('not applicable; native baseline endpoint only' if getattr(args, 'endpoint_only', False) else 'not saved; no replay' if args.arm=='Author' else 'saved intermediate fields'),
               'positive_consistency_check': {
                   'max_frame_depth_difference_from_old_m': float(np.max(np.abs(depth.max(1)-old['max_penetration_depth']))),
                   'max_frame_area_difference_from_old': float(np.max(np.abs(active@w-old['penetration_fraction'])))},
               'cost': {'evaluation_path_seconds': cost['elapsed_seconds'],
                        'evaluation_path_ms_per_frame': cost['elapsed_seconds']/len(pred)*1000,
                        'peak_allocated_gib_in_evaluation': cost['peak_allocated_gib'],
                        'native_K': cost['native_repeat'], 'response_calls_per_frame': 0 if args.arm in ('N','Author') or getattr(args, 'endpoint_only', False) else 3,
                        'global_exchanges_per_frame': getattr(args, 'global_exchanges_per_frame',
                            2 if args.arm in ('U','V','R','A','RF','RG','W','J','WJ','WJK') else 0)}}
        rows.append(row)
        write(args.output / 'garments' / args.arm / f'{case}.json', row)
        print(args.arm, case, 'tail_mm', round(row['windows']['H80']['max_penetration_depth_m']['p95']*1000, 4),
              'longest_frames', persisted['max_consecutive_frames'], 'h_mm', round(h*1000, 3), flush=True)
    result = {'scope': 'Saved raw H80 predictions. Garment motion/strain magnitudes are descriptions, not GT errors.',
              'body_query': 'Existing Warp closest-triangle winding-sign query; original threshold retained',
              'penetration_threshold_m': args.threshold,
              'duration_definition': 'Consecutive violating recorded endpoints; first relaxation interval initial_ts, subsequent intervals regular_ts',
              'initial_ts': args.initial_ts, 'regular_ts': args.regular_ts,
              'cost_scope': ('Native evaluation path including scoring/storage; not pure online inference' if getattr(args, 'endpoint_only', False) else 'Old evaluation path including stage diagnostics/storage; not pure online inference'),
              'rows': rows, 'elapsed_seconds': time.monotonic()-start}
    write(args.output / f'garment_{args.arm}.json', result)


def intrinsic_green(position, faces, rest_inverse):
    ds = np.stack((position[:, faces[:, 1]]-position[:, faces[:, 0]],
                   position[:, faces[:, 2]]-position[:, faces[:, 0]]), axis=-1)
    deformation = np.einsum('tfci,fij->tfcj', ds, rest_inverse)
    return .5*(np.einsum('tfci,tfcj->tfij', deformation, deformation)-np.eye(2))


def rest_face_inverse(rest, faces):
    e1, e2 = rest[faces[:, 1]]-rest[faces[:, 0]], rest[faces[:, 2]]-rest[faces[:, 0]]
    a = np.linalg.norm(e1, axis=-1)
    b = np.sum(e1*e2, axis=-1)/a
    c = np.sqrt(np.maximum(np.sum(e2*e2, axis=-1)-b*b, 0.))
    mat = np.zeros((len(faces), 2, 2))
    mat[:, 0, 0], mat[:, 0, 1], mat[:, 1, 1] = a, b, c
    return np.linalg.inv(mat)


def regional_series(square, mask, weights):
    weighted = mask * weights[None, :]
    den = weighted.sum(1)
    num = (square*weighted).sum(1)
    return [float(np.sqrt(n/d)) if d > 0 else None for n, d in zip(num, den)]


def cavity_analysis(args):
    start = time.monotonic()
    with args.raw_eval.open('rb') as f:
        raw = pickle.load(f)
    free = {name: Path(path) for name, path in (s.split('=', 1) for s in args.rollouts)}
    first_rest = np.asarray(raw[0]['tissue_mesh_positions'][0], dtype=np.float64)
    first_faces = np.asarray(raw[0]['tissue_mesh_triangles'], dtype=np.int64)
    e = np.unique(np.sort(np.concatenate((first_faces[:, [0, 1]], first_faces[:, [1, 2]], first_faces[:, [2, 0]])), axis=1), axis=0)
    fixed_h = float(np.linalg.norm(first_rest[e[:, 0]]-first_rest[e[:, 1]], axis=-1).mean())
    rows = []
    windows = dict(WINDOWS, H100=(0, 100), first20=(0, 20), middle40=(20, 60), last40=(60, 100))
    for i in range(args.start, args.stop):
        record = raw[i]
        rest = np.asarray(record['tissue_mesh_positions'][0], dtype=np.float64)
        gt = np.asarray(record['tissue_mesh_positions'], dtype=np.float64)[1:args.frames+1]
        faces = np.asarray(record['tissue_mesh_triangles'], dtype=np.int64)
        w, area = weights_and_areas(rest, faces)
        L = float(np.linalg.norm(np.ptp(rest, axis=0)))
        all_edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        edges, inv = np.unique(all_edges, axis=0, return_inverse=True)
        ew = np.bincount(inv, weights=np.tile(area/3, 3), minlength=len(edges))
        ew /= ew.sum()
        rest_lengths = np.linalg.norm(rest[edges[:, 0]]-rest[edges[:, 1]], axis=-1)
        gt_lengths = np.linalg.norm(gt[:, edges[:, 0]]-gt[:, edges[:, 1]], axis=-1)
        gt_step = np.diff(np.concatenate((rest[None], gt)), axis=0)
        inverse = rest_face_inverse(rest, faces)
        gt_green = intrinsic_green(gt, faces, inverse)
        with np.load(args.surface_fields/f'surface_fields_{i:03d}.npz') as z:
            sd, names = z['signed_distances'].astype(np.float64), list(z['model_names'])
        ref_sd = sd[names.index('reference')]
        gap = np.abs(ref_sd).min(-1)
        regions = {'near': gap <= 2*fixed_h, 'remote': gap > 4*fixed_h, 'all': np.ones(gt.shape[:2], dtype=bool)}
        row = {'case': i, 'poisson_ratio': float(record['poisson_ratio']), 'rest_bbox_diagonal': L,
               'rest_mean_unique_edge': float(rest_lengths.mean()), 'models': {}}
        for name, path in free.items():
            p = path/f'case_{i:03d}.npz'
            if not p.exists():
                p = path/f'trajectory_{i:03d}.npz'
            with np.load(p) as z:
                pred, target = z['prediction'][:args.frames].astype(np.float64), z['target'][:args.frames].astype(np.float64)
            if pred.shape != gt.shape or np.max(np.abs(target-gt)) > 1e-6:
                raise ValueError(f'Saved GT alignment changed for {name}/{i}')
            step = np.diff(np.concatenate((rest[None], pred)), axis=0)
            green_delta = intrinsic_green(pred, faces, inverse)-gt_green
            gsquare = np.sum(green_delta*green_delta, axis=(-2, -1))
            edge_delta = (np.linalg.norm(pred[:, edges[:, 0]]-pred[:, edges[:, 1]], axis=-1)-gt_lengths)/rest_lengths
            depth = np.maximum(-sd[names.index(name)].min(-1), 0.)
            ref_depth = np.maximum(-ref_sd.min(-1), 0.)
            persistent, series = persistence(depth>1e-6, w)
            series.update({'max_penetration_depth_over_L': (depth.max(1)/L).tolist(),
                           'penetrating_rest_area_fraction': ((depth>1e-6)@w).tolist(),
                           'depth_reference_error_over_L': (np.abs(depth-ref_depth)@w/L).tolist()})
            for region, mask in regions.items():
                incident_edges = mask[:, edges[:, 0]] | mask[:, edges[:, 1]]
                incident_faces = mask[:, faces].any(axis=-1)
                series[region+'_position_vector_rmse_over_L'] = regional_series(np.sum((pred-gt)**2, axis=-1)/L**2, mask, w)
                series[region+'_step_vector_rmse_over_L'] = regional_series(np.sum((step-gt_step)**2, axis=-1)/L**2, mask, w)
                series[region+'_edge_relative_reference_rmse'] = regional_series(edge_delta**2, incident_edges, ew)
                series[region+'_surface_green_reference_rms'] = regional_series(gsquare, incident_faces, area)
            row['models'][name] = {'windows': windowed(series, windows), 'frames': series, 'persistence': persistent}
        rows.append(row)
        if (i+1)%10 == 0 or i == args.start:
            print('Cavity expanded reference', i+1, args.stop, flush=True)
    groups = {'all': rows}
    nu_values = sorted({r['poisson_ratio'] for r in rows})
    for nu in nu_values:
        groups['nu_'+str(nu)] = [r for r in rows if r['poisson_ratio']==nu]
    aggregate = {}
    for group, selected in groups.items():
        aggregate[group] = {name: {window: {metric: describe([r['models'][name]['windows'][window][metric]['mean'] for r in selected])
                                  for metric in selected[0]['models'][name]['windows'][window]}
                              for window in windows} for name in free}
    result = {'scope': 'Saved 120-case native free rollouts, unchanged GT and original surface fields; no model replay',
              'GT_metric': 'Position and frame displacement vector RMS; face tangential Green tensor difference; area/dual-area quadrature',
              'coordinate_units': 'Dataset units, normalized by fixed object rest bbox L; not asserted to be meters',
              'near_band_dataset_units': 2*fixed_h, 'remote_band_dataset_units': 4*fixed_h,
              'region_definition': 'Common GT surface proximity and one fixed physical bandwidth across cases/models',
              'GGNS_role': 'Additional observation reference, retain native input role',
              'rows': rows, 'aggregate': aggregate, 'elapsed_seconds': time.monotonic()-start}
    write(args.output/'cavity_expanded_quality.json', result)


def density_analysis(args):
    import warp as wp
    from arcon.control.surface_measurements import SurfaceDistance
    wp.init()
    start, rows, mappings = time.monotonic(), [], {}
    for seq, garment in itertools.product(('01_01','07_02'), ('tank','tshirt','longsleeve','dress')):
        low_case, high_case = f'{seq}_{garment}_0.5', f'{seq}_{garment}_2.5'
        with np.load(args.root/'evals'/'N'/low_case/'rollout.npz') as z:
            lo_rest, lf = z['rest_pos'].astype(np.float64), z['cloth_faces']
        with np.load(args.root/'evals'/'N'/high_case/'rollout.npz') as z:
            hi_rest, hf = z['rest_pos'].astype(np.float64), z['cloth_faces']
        _, area = weights_and_areas(lo_rest, lf)
        if garment not in mappings:
            samples = lo_rest[lf].mean(1)
            surface = SurfaceDistance(hi_rest, hf, args.device)
            ids, bary, distances = surface.material_coordinates(samples)
            selected = hf[ids]
            reconstructed = np.einsum('fkc,fk->fc', hi_rest[selected], bary)
            lo_norm = np.cross(lo_rest[lf[:,1]]-lo_rest[lf[:,0]],lo_rest[lf[:,2]]-lo_rest[lf[:,0]])
            hi_norm = np.cross(hi_rest[selected[:,1]]-hi_rest[selected[:,0]],hi_rest[selected[:,2]]-hi_rest[selected[:,0]])
            cosine = np.sum(lo_norm*hi_norm,axis=-1)/(np.linalg.norm(lo_norm,axis=-1)*np.linalg.norm(hi_norm,axis=-1))
            mappings[garment] = (ids, bary, {'samples': len(samples),
                'rest_projection_distance_m': weighted_quantiles(distances, area),
                'rest_projection_reconstruction_max_difference_m': float(np.linalg.norm(reconstructed-samples,axis=-1).max()),
                'normal_cosine': weighted_quantiles(cosine, area),
                'opposite_normal_rest_area_fraction': float(area[cosine<0].sum()/area.sum())})
        ids, bary, mapping_info = mappings[garment]
        low_inverse, high_inverse = rest_face_inverse(lo_rest, lf), rest_face_inverse(hi_rest,hf)
        with np.load(args.root/'evals'/'N'/low_case/'fixed_prestate_contact_regions.npz') as z:
            near_samples = z['near_nodes'][:,lf].any(-1)
        for arm in args.arms:
            with np.load(args.root/'evals'/arm/low_case/'rollout.npz') as z:
                low, low_initial, low_body = z['pred'].astype(np.float64),z['initial_pos'].astype(np.float64),z['obstacle']
            with np.load(args.root/'evals'/arm/high_case/'rollout.npz') as z:
                high, high_initial, high_body = z['pred'].astype(np.float64),z['initial_pos'].astype(np.float64),z['obstacle']
            low_summary = read(args.root/'evals'/arm/low_case/'summary.json')
            high_summary = read(args.root/'evals'/arm/high_case/'summary.json')
            if low_summary['material']!=high_summary['material'] or low_body.shape!=high_body.shape or np.max(np.abs(low_body-high_body))>1e-6:
                raise ValueError(f'Coarse/fine material or driver mismatch for {arm}/{seq}/{garment}')
            qlow = low[:,lf].mean(2)
            qhigh = np.einsum('tfkc,fk->tfc',high[:,hf[ids]],bary)
            init_low = low_initial[lf].mean(1)
            init_high = np.einsum('fkc,fk->fc',high_initial[hf[ids]],bary)
            step_low = np.diff(np.concatenate((init_low[None],qlow)),axis=0)
            step_high = np.diff(np.concatenate((init_high[None],qhigh)),axis=0)
            low_green = intrinsic_green(low,lf,low_inverse)
            high_green = intrinsic_green(high,hf,high_inverse)[:,ids]
            eigen_delta = np.linalg.eigvalsh(low_green)-np.linalg.eigvalsh(high_green)
            series = {}
            for region, mask in [('all',np.ones(near_samples.shape,dtype=bool)),('near',near_samples),('remote',~near_samples)]:
                series[region+'_position_consistency_rms_m'] = regional_series(np.sum((qlow-qhigh)**2,axis=-1),mask,area)
                series[region+'_step_consistency_rms_m'] = regional_series(np.sum((step_low-step_high)**2,axis=-1),mask,area)
                series[region+'_displacement_from_initial_consistency_rms_m'] = regional_series(np.sum(((qlow-init_low)-(qhigh-init_high))**2,axis=-1),mask,area)
                series[region+'_principal_green_eigenvalue_consistency_rms'] = regional_series(np.sum(eigen_delta*eigen_delta,axis=-1),mask,area)
            row = {'arm':arm,'pair':f'{seq}_{garment}','windows':windowed(series),'frames':series,
                   'mapping':mapping_info,'initial_sample_difference_rms_m':rms(init_low-init_high,area),
                   'coarse_K':low_summary['native_repeat'],'fine_K':high_summary['native_repeat']}
            rows.append(row)
            print('Density',arm,seq,garment,'H80_mm',round(row['windows']['H80']['all_position_consistency_rms_m']['mean']*1000,3),flush=True)
    write(args.output/'density_consistency.json',{'scope':'Fixed canonical rest-surface correspondence, reused for all frames/arms; consistency, not GT accuracy',
        'sampling':'Coarse rest triangle centroids with coarse rest-area quadrature; frozen fine face/barycentric map',
        'region':'Common coarse-N material sample labels shared across all arms and both densities',
        'mapping_quality':'Rest projection and normal agreement recorded; no dynamic nearest-point reassignment',
        'rows':rows,'elapsed_seconds':time.monotonic()-start})


def summarize(args):
    garments = {arm:read(args.output/f'garment_{arm}.json') for arm in args.arms}
    collisions = {arm:read(args.output/f'self_collision_{arm}.json') for arm in args.arms}
    rows = []
    for arm in args.arms:
        coll = {r['case']:r for r in collisions[arm]['rows']}
        for r in garments[arm]['rows']:
            c = coll[r['case']]
            windows = {k:dict(v) for k,v in r['windows'].items()}
            for k in windows:
                for metric, value in c['windows'][k].items():
                    windows[k]['self_'+metric] = value
            rows.append({'arm':arm,'case':r['case'],'windows':windows,'cost':r['cost'],
                         'persistence':r['persistence'],'scale':r['scale']})
    groups = {'all':rows,'coarse':[r for r in rows if r['case'].endswith('_0.5')],
              'fine':[r for r in rows if r['case'].endswith('_2.5')]}
    aggregate = {}
    for group, selected in groups.items():
        aggregate[group]={}
        for arm in args.arms:
            selected_arm=[r for r in selected if r['arm']==arm]
            aggregate[group][arm]={window:{metric:{stat:describe([r['windows'][window][metric][stat] for r in selected_arm])
                                                  for stat in ('mean','p95','max')}
                                          for metric in selected_arm[0]['windows'][window]}
                                    for window in WINDOWS}
            aggregate[group][arm]['cost']={key:describe([r['cost'][key] for r in selected_arm]) for key in selected_arm[0]['cost']}
            aggregate[group][arm]['persistence']={key:describe([r['persistence'][key] for r in selected_arm]) for key in selected_arm[0]['persistence']}
    paired={}
    for candidate, reference in [('WJ_LS','WJ_L'),('WJ_L','WJ'),('WJ_LS','WJ'),
                                 ('WJ_L','WJK'),('WJ_LS','WJK'),('S','N'),('R','N'),('A','N'),('R','S'),('A','R'),
                                 ('RF','N'),('RF','S'),('RF','R'),('RF','A'),
                                 ('RG','N'),('RG','S'),('RG','R'),('RG','A'),('RG','RF'),
                                 ('W','R'),('J','R'),('WJ','R'),('WJK','R'),
                                 ('WJ','W'),('WJ','J'),('WJK','WJ')]:
        if candidate not in args.arms or reference not in args.arms:
            continue
        pairs={}
        for group, selected in groups.items():
            a={r['case']:r for r in selected if r['arm']==candidate}
            b={r['case']:r for r in selected if r['arm']==reference}
            pairs[group]={}
            for window in WINDOWS:
                metrics={}
                for metric in next(iter(a.values()))['windows'][window]:
                    metrics[metric]={}
                    for stat in ('mean','p95','max'):
                        values=[(a[case]['windows'][window][metric][stat],b[case]['windows'][window][metric][stat]) for case in a]
                        values=[(x,y) for x,y in values if x is not None and y is not None]
                        ma,mb=np.mean([x for x,y in values]),np.mean([y for x,y in values])
                        metrics[metric][stat]={'candidate_mean':float(ma),'reference_mean':float(mb),
                            'relative_mean_change_percent':float((ma/mb-1)*100) if mb else None,
                            'lower_cases':sum(x<y-max(1e-12,1e-6*max(abs(x),abs(y))) for x,y in values),
                            'nonincreasing_cases':sum(x<=y+max(1e-12,1e-6*max(abs(x),abs(y))) for x,y in values),
                            'compared_cases':len(values)}
                pairs[group][window]=metrics
        paired[f'{candidate}_vs_{reference}']=pairs
    write(args.output/'expanded_quality_summary.json',{'scope':f'{len(rows)} saved garment rollouts, unchanged metrics, contact/persistence/self-contact and scale diagnostics',
          'quality_selection':'Compare contact average/tail/self-contact and cost with structure/motion, not only five-term nonincrease',
          'cost_scope':garments[args.arms[0]]['cost_scope'],'rows':rows,'aggregate':aggregate,'paired':paired})
    print('Expanded summary',len(rows),'saved trajectories',flush=True)


def response_analysis(args):
    """Read natural saved states; separate geometry sensitivity and response directions."""
    import torch
    from arcon.control.surface_measurements import SurfaceDistance
    from arcon.control.graph_setup import install
    anchor, _ = install(args.project, args.data)
    hinge_basis = anchor.hinge_basis
    torch.set_num_threads(4)
    started = time.monotonic()
    rows = []
    cases = args.cases or CASES
    labels = ('q_t', 'stage5_q_hat', 'stage5_refined', 'stage10_q_hat',
              'stage10_refined', 'q_hat', 'pred_pos')

    def finite_stats(values):
        finite = np.asarray(values, dtype=np.float64)
        good = np.isfinite(finite)
        result = describe(finite[good])
        result['min'] = float(finite[good].min()) if good.any() else None
        result['nonfinite_count'] = int((~good).sum())
        return result

    for arm in args.arms:
        for case in cases:
            directory = args.root/'evals'/arm/case
            with np.load(directory/'rollout.npz') as saved_rollout:
                rest = saved_rollout['rest_pos']
                faces = saved_rollout['cloth_faces'].astype(np.int64)
                body = saved_rollout['obstacle']
                body_faces = saved_rollout['obstacle_faces'].astype(np.int64)
            if faces.shape[1] != 3: faces = faces.T
            if body_faces.shape[1] != 3: body_faces = body_faces.T
            weights, area = weights_and_areas(rest.astype(np.float64), faces)
            # Each interior rest edge defines the same two triangles as native hinges.
            edge_pairs = np.sort(faces[:, [[0,1],[1,2],[2,0]]], axis=-1).reshape(-1,2)
            face_ids = np.repeat(np.arange(len(faces)), 3)
            shared, inverse, counts = np.unique(edge_pairs, axis=0, return_inverse=True, return_counts=True)
            order = np.argsort(inverse, kind='stable')
            starts = np.r_[0, np.cumsum(counts)[:-1]][counts == 2]
            connectivity = np.stack((face_ids[order[starts]], face_ids[order[starts+1]]), -1)
            shared = shared[counts == 2]
            hinge_length = np.linalg.norm(rest[shared[:,0]]-rest[shared[:,1]], axis=-1)
            tf, tc, te = (torch.as_tensor(x, device=args.device) for x in (faces, connectivity, shared))
            with torch.no_grad():
                _, derivative, _, _ = hinge_basis(torch.as_tensor(rest, device=args.device), tf, tc, te)
                rest_condition = derivative.flatten(1).norm(dim=1).cpu().numpy()*hinge_length
            surface = SurfaceDistance(body[0], body_faces, args.device)
            with np.load(directory/'response_runtime.npz') as runtime:
                for index, frame in enumerate(runtime['frame']):
                    surface.update(body[int(frame)-1])
                    entry = {'arm': arm, 'case': case, 'frame': int(frame), 'geometry': {}, 'responses': {}}
                    for label in labels:
                        q = runtime[label][index]
                        tri = q.astype(np.float64)[faces]
                        current_area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)/2
                        with torch.no_grad():
                            _, derivative, _, _ = hinge_basis(torch.as_tensor(q,device=args.device), tf, tc, te)
                            condition = derivative.flatten(1).norm(dim=1).cpu().numpy()*hinge_length
                        # The model's float32 derivative is used; no altered force or repaired state.
                        entry['geometry'][label] = {
                            'current_area_min_m2': float(current_area.min()),
                            'face_area_ratio': finite_stats(current_area/area),
                            'dimensionless_hinge_condition': finite_stats(condition),
                            'hinge_amplification_from_rest': finite_stats(condition/rest_condition)}
                    for stage, before, after in ((5,'stage5_q_hat','stage5_refined'),
                                                (10,'stage10_q_hat','stage10_refined'),
                                                (15,'q_hat','pred_pos')):
                        q = runtime[before][index].astype(np.float64)
                        total = runtime[after][index].astype(np.float64)-q
                        defect = q-runtime['q_t'][index]-runtime['u_t'][index]
                        distance, _, outward = surface.details(q)
                        penetrating = distance < -args.threshold
                        source = ('stage'+str(stage)+'_') if stage != 15 else ''
                        internal_key, contact_key = source+'internal_increment', source+'contact_increment'
                        response = {'total_increment_norm_m': finite_stats(np.linalg.norm(total,axis=-1)),
                            'area_weighted_total_increment_rms_m': rms(total,weights),
                            'defect_square_change_m2': float((2*np.sum(defect*total,axis=-1)+np.sum(total*total,axis=-1))@weights),
                            'proposal_penetrating_rest_area': float(weights[penetrating].sum()),
                            'component_fields_saved': internal_key in runtime and contact_key in runtime}
                        if response['component_fields_saved']:
                            internal, contact = (runtime[k][index].astype(np.float64) for k in (internal_key,contact_key))
                            normal = np.sum(contact*outward,axis=-1)
                            contact_norm = np.linalg.norm(contact,axis=-1)
                            support = contact_norm > 1e-12
                            pen_contact = penetrating & support
                            area_support, area_pen = float(weights[support].sum()),float(weights[pen_contact].sum())
                            response.update({
                                'internal_increment_norm_m': finite_stats(np.linalg.norm(internal,axis=-1)),
                                'contact_increment_norm_m': finite_stats(contact_norm),
                                'area_weighted_internal_increment_rms_m': rms(internal,weights),
                                'area_weighted_contact_increment_rms_m': rms(contact,weights),
                                'contact_support_rest_area': area_support,
                                'penetrating_contact_support_rest_area': area_pen,
                                'contact_inward_rest_area': float(weights[support & (normal < -1e-9)].sum()),
                                'contact_inward_fraction_on_penetrating_support':
                                    float(weights[pen_contact & (normal < -1e-9)].sum()/area_pen) if area_pen else None,
                                'contact_normal_projection_mean_on_penetrating_support_m':
                                    float((normal*pen_contact)@weights/area_pen) if area_pen else None,
                                'internal_defect_square_change_m2':
                                    float((2*np.sum(defect*internal,axis=-1)+np.sum(internal*internal,axis=-1))@weights),
                                'contact_defect_square_change_m2':
                                    float((2*np.sum(defect*contact,axis=-1)+np.sum(contact*contact,axis=-1))@weights),
                                'component_cross_defect_square_m2': float(2*np.sum(internal*contact,axis=-1)@weights),
                                'component_sum_difference_max_m': float(np.abs(internal+contact-total).max())})
                        entry['responses'][str(stage)] = response
                    rows.append(entry)
            print('Saved response geometry',arm,case,flush=True)
    groups = {'all':rows,'coarse':[r for r in rows if r['case'].endswith('_0.5')],
              'fine':[r for r in rows if r['case'].endswith('_2.5')]}
    aggregate = {}
    metrics = ('area_weighted_total_increment_rms_m','defect_square_change_m2',
               'contact_inward_fraction_on_penetrating_support',
               'contact_normal_projection_mean_on_penetrating_support_m',
               'area_weighted_internal_increment_rms_m','area_weighted_contact_increment_rms_m')
    for group, selected in groups.items():
        aggregate[group] = {arm: {str(stage): {metric: describe([
            r['responses'][str(stage)].get(metric) for r in selected if r['arm']==arm])
            for metric in metrics} for stage in (5,10,15)} for arm in args.arms}
    write(args.output/'response_geometry.json',{
        'state':'complete','scope':'Natural saved S/R/RF/RG proposals, actual response components; no model replay or constructed states',
        'geometry':'Actual native float32 hinge derivative, rest-edge dimensionless norm and per-hinge rest amplification',
        'contact_direction':'Aggregated saved contact displacement projected on true closest-body outward direction, not per-pair force/admissibility or stress',
        'defect':'Rest-area weighted squared displacement from q_t+u_t, units m2, not thermodynamic work',
        'rows':rows,'aggregate':aggregate,'elapsed_seconds':time.monotonic()-started})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='mode',required=True)
    g=sp.add_parser('garment')
    g.add_argument('--root',type=Path,required=True);g.add_argument('--output',type=Path,required=True)
    g.add_argument('--arm',choices=AVAILABLE_ARMS,required=True);g.add_argument('--device',default='cuda:0')
    g.add_argument('--cases',nargs='+');g.add_argument('--threshold',type=float,default=1e-6)
    g.add_argument('--initial-ts',type=float,default=1/3);g.add_argument('--regular-ts',type=float,default=1/30)
    g.set_defaults(func=garment_analysis)
    c=sp.add_parser('cavity')
    c.add_argument('--raw-eval',type=Path,required=True);c.add_argument('--rollouts',action='append',required=True)
    c.add_argument('--surface-fields',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    c.add_argument('--start',type=int,default=0);c.add_argument('--stop',type=int,default=120);c.add_argument('--frames',type=int,default=100)
    c.set_defaults(func=cavity_analysis)
    d=sp.add_parser('density')
    d.add_argument('--root',type=Path,required=True);d.add_argument('--output',type=Path,required=True)
    d.add_argument('--arms',nargs='+',default=list(ARMS));d.add_argument('--device',default='cuda:0')
    d.set_defaults(func=density_analysis)
    r=sp.add_parser('responses')
    r.add_argument('--root',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--project',type=Path,required=True);r.add_argument('--data',type=Path,required=True)
    r.add_argument('--device',default='cuda:0')
    r.add_argument('--arms',nargs='+',choices=('S','R','RF','RG'),default=['S','R','RF','RG'])
    r.add_argument('--cases',nargs='+');r.add_argument('--threshold',type=float,default=1e-6)
    r.set_defaults(func=response_analysis)
    a=sp.add_parser('summarize')
    a.add_argument('--output',type=Path,required=True);a.add_argument('--arms',nargs='+',default=list(ARMS))
    a.set_defaults(func=summarize)
    args=p.parse_args();args.func(args)


if __name__=='__main__':
    main()
