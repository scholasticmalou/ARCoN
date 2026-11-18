"""Exact geometry/persistence primitives used for the paper measurements."""
import numpy as np

def weights_and_areas(rest, faces):
    area = np.linalg.norm(np.cross(rest[faces[:, 1]] - rest[faces[:, 0]],
                                   rest[faces[:, 2]] - rest[faces[:, 0]]), axis=-1) / 2
    w = np.zeros(len(rest), dtype=np.float64)
    for k in range(3):
        np.add.at(w, faces[:, k], area / 3)
    return w / w.sum(), area

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
