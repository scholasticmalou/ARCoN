"""Local material-geometry conditioning for the bending response."""
from __future__ import annotations

import math
import time
import numpy as np
import torch


class MaterialCurvatureStencil:
    radius = .03
    normal_matrix_regularization = 1e-8
    coverage_eigenvalue_ratio = 1e-6

    def __init__(self, sample):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        started = time.perf_counter()
        cloth = sample['cloth']
        rest = cloth.rest_pos.detach().cpu().double().numpy()
        faces = cloth.faces_batch.T.detach().cpu().numpy()
        connectivity = cloth.f_connectivity.detach().cpu().numpy()
        shared = cloth.f_connectivity_edges.detach().cpu().numpy()
        area = cloth.f_area.detach().cpu().double().numpy().reshape(-1)
        count = len(shared)
        tri = rest[faces]
        normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        normal = normals[connectivity].sum(1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        edge = rest[shared[:, 1]]-rest[shared[:, 0]]
        length = np.linalg.norm(edge, axis=1)
        tangent = edge/length[:, None]
        conormal = np.cross(normal, tangent)
        conormal /= np.linalg.norm(conormal, axis=1, keepdims=True)
        second = np.cross(normal, conormal)
        centers = tri.mean(1)
        orientation = np.sign(np.einsum('ij,ij->i', conormal,
                                       centers[connectivity[:, 1]]-centers[connectivity[:, 0]]))
        dual = 2*area[connectivity].sum(1)/(3*length)
        if not (np.all(dual > 0) and np.all(orientation != 0)):
            raise ValueError('Curvature requires positive material duals and oriented manifold hinges')
        midpoint = rest[shared].mean(1)
        incident = [[] for _ in faces]
        for hinge, pair in enumerate(connectivity):
            for face in pair:
                incident[int(face)].append(hinge)
        src, dst = [], []
        for hinges in incident:
            for i, a in enumerate(hinges):
                for b in hinges[i+1:]:
                    src.extend((a, b)); dst.extend((b, a))
        src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
        distance = np.linalg.norm(midpoint[src]-midpoint[dst], axis=1)
        graph = coo_matrix((distance, (src, dst)), shape=(count, count)).tocsr()
        support = .5*length*dual
        all_rows, all_cols, all_coefficients = [], [], []
        eligible = np.zeros(count, dtype=bool)
        reconstruction_error = 0.
        opposite_normal_pairs = 0
        # Batches bound temporary all-target distances, not the neighborhood.
        for begin in range(0, count, 64):
            stop = min(begin+64, count)
            distances = dijkstra(graph, directed=False, indices=np.arange(begin, stop), limit=self.radius)
            for offset, target in enumerate(range(begin, stop)):
                ds = distances[offset]
                neighbors = np.flatnonzero((ds > 0) & (ds < self.radius))
                cosine = normal[neighbors] @ normal[target]
                opposite_normal_pairs += int(np.count_nonzero(cosine <= 0))
                keep = cosine > 0
                neighbors, cosine = neighbors[keep], cosine[keep]
                if len(neighbors) < 3:
                    continue
                axis = np.cross(normal[neighbors], normal[target])
                first_cross = np.cross(axis, conormal[neighbors])
                transported = conormal[neighbors]+first_cross+np.cross(axis, first_cross)/(1+cosine[:, None])
                x, y = transported @ conormal[target], transported @ second[target]
                design = np.stack((x*x, math.sqrt(2)*x*y, y*y), axis=1)
                weight = support[neighbors]*(1-(ds[neighbors]/self.radius)**2)**2
                matrix = design.T @ (weight[:, None]*design)
                eigenvalues = np.linalg.eigvalsh(matrix)
                if eigenvalues[0] <= self.coverage_eigenvalue_ratio*eigenvalues[-1]:
                    continue
                regularized = matrix+self.normal_matrix_regularization*np.trace(matrix)/3*np.eye(3)
                functional = np.linalg.solve(regularized, np.array([1., 0., 0.]))
                coefficients = (design @ functional)*weight
                reconstruction_error = max(reconstruction_error,
                    float(np.max(np.abs(coefficients @ design-np.array([1., 0., 0.])))))
                eligible[target] = True
                all_rows.append(np.full(len(neighbors), target, dtype=np.int64))
                all_cols.append(neighbors)
                all_coefficients.append(coefficients*orientation[neighbors]/dual[neighbors])
        if not eligible.any():
            raise ValueError('No full-direction material curvature neighborhood; C would equal A everywhere')
        device, dtype = cloth.rest_pos.device, cloth.rest_pos.dtype
        self.rows = torch.as_tensor(np.concatenate(all_rows), device=device)
        self.cols = torch.as_tensor(np.concatenate(all_cols), device=device)
        self.coefficients = torch.as_tensor(np.concatenate(all_coefficients), device=device, dtype=dtype)
        self.scale = torch.as_tensor(orientation/dual, device=device, dtype=dtype)
        self.eligible = torch.as_tensor(eligible, device=device)
        self.metadata = dict(radius_m=self.radius, hinges=count, conditioned_hinges=int(eligible.sum()),
            conditioned_fraction=float(eligible.mean()), sparse_pairs=len(self.rows),
            static_tensor_bytes=sum(t.numel()*t.element_size() for t in
                (self.rows, self.cols, self.coefficients, self.scale, self.eligible)),
            preparation_seconds=time.perf_counter()-started,
            tensor_linear_reconstruction_max_error=reconstruction_error,
            excluded_opposed_rest_normal_pairs=opposite_normal_pairs,
            unsupported_rule='g=1, retain A; no fitted-tensor claim at unsupported hinges',
            neighborhood='3cm material midpoint geodesic; self excluded; no neighbor cap',
            tensor='symmetric 2x2 in rest tangent plane; minimal-rotation conormal transport',
            regularization=self.normal_matrix_regularization,
            coverage_eigenvalue_ratio=self.coverage_eigenvalue_ratio)

    def describe(self, theta):
        curvature = self.scale*theta
        prediction = torch.zeros_like(theta).index_add(0, self.rows, self.coefficients*theta[self.cols])
        same_sign = torch.relu(curvature.sign()*prediction)
        gate = ((curvature.abs()-same_sign)/(curvature.abs()+1e-6/self.radius)).clamp(0, 1)
        gate = torch.where(self.eligible, gate, torch.ones_like(gate))
        return gate, curvature, prediction

    def gate(self, theta):
        return self.describe(theta)[0]

    def statistics(self, theta, gate):
        _, curvature, prediction = self.describe(theta.detach())
        return dict(**self.metadata, gate_mean=float(gate.detach().mean()),
            gate_min=float(gate.detach().min()), gate_max=float(gate.detach().max()),
            curvature_abs_mean_per_m=float(curvature.abs().mean()),
            predicted_curvature_abs_mean_per_m=float(prediction.abs().mean()))
