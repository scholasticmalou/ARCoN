"""Current-geometry self-contact queries and sparse spatial graph interactions."""
from __future__ import annotations

import copy
import time

import torch
from torch import nn

from models.core.relation_response import mean_scatter, mlp, vertex_normals
from models.core.geometry_feedback import GeometryRelationResponse, GeometryFeedbackCore, GeometryFeedbackModel

SELF_ARMS = ('WJ_L', 'WJ_LS')
SELF_PARAMETER_NAMES = ('self_encoder', 'self_message', 'self_output')


@torch.no_grad()
def sparse_self_pairs(q, mesh_edges):
    """K11 before exclusion; canonical pairs use the actual vertex count."""
    from pytorch3d.ops import knn_points

    n = len(q)
    pos = q.detach()[None]
    squared_distance, index, _ = knn_points(
        pos, pos, K=11, return_sorted=False, return_nn=False)
    distance = squared_distance[0].sqrt()
    node = torch.arange(n, device=q.device)[:, None].expand_as(index[0])
    selected = (distance > 1e-7) & (distance < 0.02)
    left, right = node[selected], index[0][selected]
    keys = torch.unique(torch.minimum(left, right)*n+torch.maximum(left, right), sorted=True)
    source, receiver = mesh_edges
    mesh = torch.unique(torch.minimum(source, receiver)*n+torch.maximum(source, receiver), sorted=True)
    where = torch.searchsorted(mesh, keys)
    existing = (where < len(mesh)) & (mesh[where.clamp_max(len(mesh)-1)] == keys)
    keys = keys[~existing]
    left, right = keys//n, keys % n
    return torch.stack((torch.cat((left, right)), torch.cat((right, left))))


class MaterialHopTwo:
    """Fixed physical-mesh exclusion; no learned tensors or deformation adjacency."""

    def __init__(self, mesh_edges, n):
        begin = time.perf_counter()
        adjacency = [set() for _ in range(n)]
        for a, b in mesh_edges.detach().cpu().T.tolist():
            if a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)
        keys, sizes = [], []
        for i, one in enumerate(adjacency):
            two = set(one)
            for j in one:
                two.update(adjacency[j])
            two.discard(i)
            sizes.append(len(two))
            keys.extend(i*n+j for j in sorted(two))
        self.keys = torch.tensor(keys, dtype=torch.long, device=mesh_edges.device)
        self.maximum_size = max(sizes, default=0)
        self.vertices = n
        self.preparation_seconds = time.perf_counter()-begin

    def contains(self, keys):
        if not len(self.keys):
            return torch.zeros_like(keys, dtype=torch.bool)
        at = torch.searchsorted(self.keys, keys)
        return (at < len(self.keys)) & (self.keys[at.clamp_max(len(self.keys)-1)] == keys)


@torch.no_grad()
def nonlocal_self_pairs(q, topology, *, profile=False):
    """Filter H2 before nearest eligible10; expand only still-underfilled rows."""
    from pytorch3d.ops import knn_points

    n = len(q)
    if topology.vertices != n:
        raise ValueError('Material topology and query vertex counts disagree')
    initial_k = min(n, 1+topology.maximum_size+10)
    k = initial_k
    rows = torch.arange(n, device=q.device)
    chosen_left, chosen_right, queries, events = [], [], [], []
    selector_begin = time.perf_counter()
    while len(rows):
        start = end = None
        if profile and q.is_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        squared, index, _ = knn_points(q.detach()[rows][None], q.detach()[None],
            K=k, return_sorted=True, return_nn=False)
        if end is not None:
            end.record()
            events.append((start, end))
        distance, index = squared[0].sqrt(), index[0]
        left = rows[:, None].expand_as(index)
        eligible = ((index != left) & (distance > 1e-7) & (distance < .02)
                    & ~topology.contains(left*n+index))
        counts = eligible.sum(-1)
        # A sorted farthest candidate outside the radius closes this query.
        done = (counts >= 10) | (distance[:, -1] >= .02) | (k == n)
        selected = eligible & (eligible.cumsum(-1) <= 10) & done[:, None]
        chosen_left.append(left[selected])
        chosen_right.append(index[selected])
        queries.append(dict(K=k, queried_rows=len(rows), completed_rows=int(done.sum()),
            underfilled_rows=int((~done).sum())))
        rows = rows[~done]
        if len(rows):
            k = min(n, max(k+1, 2*k))
    left, right = torch.cat(chosen_left), torch.cat(chosen_right)
    canonical = torch.unique(torch.minimum(left, right)*n+torch.maximum(left, right), sorted=True)
    low, high = canonical//n, canonical % n
    edges = torch.stack((torch.cat((low, high)), torch.cat((high, low))))
    record = dict(selector='material_h2_then_nearest10_eligible_2cm', initial_K=initial_k,
        candidate_queries=queries, knn_calls=len(queries), selected_directed_choices=len(left),
        directed_edges=edges.shape[1], material_distance_lower_bound=3,
        material_distance_upper_bound=None, maximum_H2_size=topology.maximum_size,
        union_degree_is_not_capped_at_10=True)
    if profile:
        if events:
            events[-1][1].synchronize()
            record['knn_gpu_ms'] = sum(start.elapsed_time(end) for start, end in events)
        record['selector_wall_seconds'] = time.perf_counter()-selector_begin
        record['selected_H2_edges'] = int(topology.contains(edges[1]*n+edges[0]).sum())
        degree = torch.bincount(edges[1], minlength=n)
        record['active_nodes'] = int((degree > 0).sum())
        record['maximum_union_degree'] = int(degree.max())
        if len(left):
            distances = (q[left]-q[right]).norm(dim=-1)
            record['selected_distance_min_m'] = float(distances.min())
            record['selected_distance_max_m'] = float(distances.max())
        if record['selected_H2_edges']:
            raise RuntimeError('Actual nonlocal selector retained material H2 edges')
    return edges, record


class SparseSelfRelationResponse(GeometryRelationResponse):
    """One shared 8->128 edge encoder and receiver-mean latent residual."""

    def __init__(self, edge_sets, mode):
        super().__init__(edge_sets, mode)
        self.self_encoder = mlp(8, 128, norm=True)
        self.self_message = mlp(384, 128, norm=True)
        self.self_output = nn.Linear(128, 128)
        nn.init.zeros_(self.self_output.weight)
        nn.init.zeros_(self.self_output.bias)
        self.self_queries = 0
        self.self_message_calls = 0
        self.self_graph_records = []
        self.self_query_records = []
        self.self_knn_calls = 0
        self.self_neighbor_mode = 'legacy_k11'
        self.material_topologies = {}
        self.profile_self_queries = False

    def reset_frame(self):
        super().reset_frame()
        self.self_queries = 0
        self.self_message_calls = 0
        self.self_graph_records = []
        self.self_query_records = []
        self.self_knn_calls = 0

    def encode_self_graph(self, graph):
        cloth = graph['cloth']
        q = cloth.pred_pos
        mesh_edges = graph['cloth', 'mesh_edge', 'cloth'].edge_index
        if self.self_neighbor_mode == 'nonlocal_h2_eligible10':
            key = (str(graph.garment_name), len(q), mesh_edges.shape[1], str(q.device))
            fresh = key not in self.material_topologies
            if fresh:
                self.material_topologies[key] = MaterialHopTwo(mesh_edges, len(q))
            topology = self.material_topologies[key]
            edge_index, query = nonlocal_self_pairs(q, topology, profile=self.profile_self_queries)
            query['material_preparation_seconds_this_call'] = topology.preparation_seconds if fresh else 0.
            self.self_query_records.append(query)
            self.self_knn_calls += query['knn_calls']
        elif self.self_neighbor_mode == 'legacy_k11':
            edge_index = sparse_self_pairs(q, mesh_edges)
            self.self_knn_calls += 1
        else:
            raise ValueError('Unknown self neighbor semantics: '+self.self_neighbor_mode)
        self.self_queries += 1
        source, receiver = edge_index
        if not len(source):
            if self.record:
                self.self_graph_records.append(dict(directed_edges=0, active_nodes=0,
                    feature_requires_grad=q.requires_grad, feature_width=8, output_increments=[]))
            return edge_index, None, receiver
        h = cloth.average_edge_length.reshape(-1)[0]
        normals = vertex_normals(q, cloth.faces_batch.T)
        relative = q[receiver]-q[source]
        displacement = q-cloth.pos
        relative_displacement = displacement[receiver]-displacement[source]
        features = torch.cat((relative/h, relative.norm(dim=-1, keepdim=True)/h,
                              relative_displacement/h,
                              (normals[source]*normals[receiver]).sum(-1, keepdim=True)), -1)
        encoded = self.self_encoder(features)
        active = torch.unique(receiver, sorted=True)
        if self.record:
            self.self_graph_records.append(dict(directed_edges=len(source), active_nodes=len(active),
                feature_requires_grad=features.requires_grad, feature_width=features.shape[1],
                selection_requires_grad=edge_index.requires_grad, output_increments=[]))
        return edge_index, encoded, active

    def add_self_message(self, graph, edge_index, encoded, active):
        self.self_message_calls += 1
        # Empty receivers never get the projection bias, including after training.
        if encoded is None:
            return graph
        nodes = graph['cloth'].node_features
        source, receiver = edge_index
        messages = self.self_message(torch.cat((nodes[source], encoded, nodes[receiver]), -1))
        aggregated = mean_scatter(messages, receiver, len(nodes))
        increment = self.self_output(aggregated[active])
        graph['cloth'].node_features = nodes.index_add(0, active, increment)
        if self.record:
            self.self_graph_records[-1]['output_increments'].append(dict(
                max_absolute=float(increment.detach().abs().max()),
                rms=float(increment.detach().square().mean().sqrt())))
        return graph

    def process_graph(self, graph, repeat):
        # Query/encode once on this response's q_hat, reuse in both message calls.
        edge_index, encoded, active = self.encode_self_graph(graph)
        graph = self.aggr_processor(graph, repeat)
        graph = self.add_self_message(graph, edge_index, encoded, active)
        graph = self.processor_steps[0](graph)
        graph = self.add_self_message(graph, edge_index, encoded, active)
        return self.processor_steps[1](graph)


class SelfInteractionCore(GeometryFeedbackCore):
    def __init__(self, mcfg):
        arm = mcfg.screen_arm
        if arm not in SELF_ARMS:
            raise ValueError('Self-interaction screen requires WJ_L or WJ_LS')
        # Choose the response BEFORE StageCore installs native recomputation.
        self.response_class = SparseSelfRelationResponse if arm == 'WJ_LS' else GeometryRelationResponse
        internal = copy.deepcopy(mcfg)
        internal.screen_arm = 'WJ'
        super().__init__(internal)
        self.self_arm = arm
        if arm == 'WJ_LS':
            self.response.self_neighbor_mode = getattr(mcfg, 'self_neighbor_mode', 'legacy_k11')
        # Inherited core.arm='R' and coevolution_arm='WJ' retain current contexts.


class SelfInteractionModel(GeometryFeedbackModel):
    def forward(self, sample, repeat=None, is_training=True):
        core, response = self._learned_model, self._learned_model.response
        # Query profiling is a TRAIN record, not an inference-time synchronization.
        if core.self_arm == 'WJ_LS':
            response.profile_self_queries = bool(is_training)
        sample = super().forward(sample, repeat=repeat, is_training=is_training)
        messages = core.self_arm == 'WJ_LS'
        core.execution.update(organization=core.self_arm, base_organization='WJ',
            immediate_context=core.arm, self_relation_queries=response.self_queries if messages else 0,
            self_message_calls=response.self_message_calls if messages else 0,
            response_order='K,self,typed1,self,typed2' if messages else 'K,typed1,typed2',
            self_message_shared_across_responses=messages, self_new_endpoint_head=False)
        if messages:
            core.execution['self_knn_calls'] = response.self_knn_calls
            core.execution['self_neighbor_mode'] = response.self_neighbor_mode
            if response.self_neighbor_mode == 'nonlocal_h2_eligible10':
                core.execution['self_query_records'] = list(response.self_query_records)
        if messages and response.record:
            core.execution['self_graphs'] = list(response.self_graph_records)
        return sample


def EncodeProcessDecode(mcfg):
    return SelfInteractionCore(mcfg)
