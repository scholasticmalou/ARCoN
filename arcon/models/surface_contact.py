"""Body-surface correspondence and contact-plane geometry."""
import torch
import warp as wp


@wp.kernel
def nearest_surface(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), radius: float,
                    faces: wp.array(dtype=wp.int32), closest: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    hit = wp.mesh_query_point_no_sign(mesh, points[i], radius)
    faces[i] = -1
    closest[i] = wp.vec3(0.0, 0.0, 0.0)
    if hit.result:
        faces[i] = hit.face
        closest[i] = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)


class SurfaceGeometry:
    """Prescribed body mesh retained for the frame and checkpoint recomputation."""
    def __init__(self, sample):
        wp.init()
        body = sample['obstacle']
        self.positions = body.target_pos.detach().contiguous()
        self.faces = body.faces_batch.T.long().contiguous()
        self.indices = self.faces.to(torch.int32).reshape(-1).contiguous()
        self.types = body.vertex_type.reshape(-1)
        from utils.common import NodeType
        self.omit = NodeType.OBSTACLE_OMIT
        triangle = self.positions[self.faces]
        raw = torch.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0], dim=-1)
        self.normals = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1.e-12)
        self.stream = wp.stream_from_torch(self.positions.device)
        with wp.ScopedStream(self.stream):
            self.mesh = wp.Mesh(points=wp.from_torch(self.positions, dtype=wp.vec3),
                                indices=wp.from_torch(self.indices, dtype=wp.int32))
        cloth = sample['cloth']
        self.cloth_faces = cloth.faces_batch.T.long()
        self.area = cloth.f_area.detach().reshape(-1)
        self.bary = self.positions.new_tensor([[1/3,1/3,1/3], [.5,.5,0], [0,.5,.5], [.5,0,.5]])

    @torch.no_grad()
    def query(self, points, radius):
        points = points.detach().contiguous()
        ids = torch.empty(len(points), dtype=torch.int32, device=points.device)
        closest = torch.empty_like(points)
        if len(points):
            wp.launch(nearest_surface, dim=len(points), inputs=[self.mesh.id,
                wp.from_torch(points, dtype=wp.vec3), float(radius)],
                outputs=[wp.from_torch(ids, dtype=wp.int32), wp.from_torch(closest, dtype=wp.vec3)],
                stream=wp.stream_from_torch(points.device))
        valid = ids >= 0
        safe = ids.long().clamp_min(0)
        # As in CC, omitted-body filtering uses the nearest vertex of the chosen face.
        vertices = self.faces[safe]
        which = (self.positions[vertices]-points[:,None]).square().sum(-1).argmin(-1)
        allowed = self.types[vertices.gather(1, which[:,None]).squeeze(1)] != self.omit
        return valid & allowed, closest, self.normals[safe]

    def project(self, q_raw, mass, free, inherited, epsilon, radius, record=False):
        valid, points, normals = self.query(q_raw, 1.e6)
        with torch.no_grad():
            near = (q_raw.detach()-points).square().sum(-1) <= radius**2
            near[inherited] = True
            supported = valid & near & free
        gap = ((q_raw-points)*normals).sum(-1)
        vertex_delta = (epsilon-gap).clamp_min(0)[:,None]*normals*supported[:,None]
        q_vertex = q_raw + vertex_delta
        triangles = q_vertex[self.cloth_faces]
        samples = torch.einsum('sc,fcd->fsd', self.bary, triangles).reshape(-1,3)
        valid_face, fp, fn = self.query(samples, radius)
        face_gap = ((samples-fp)*fn).sum(-1)
        deficit = (epsilon-face_gap).clamp_min(0)*valid_face
        inv_mass = free.to(mass.dtype)/mass
        wi = inv_mass[self.cloth_faces]
        denominator = torch.einsum('sc,fc->fs', self.bary.square(), wi).clamp_min(1.e-20)
        factors = self.bary[None,:,:]*wi[:,None,:]/denominator[:,:,None]
        offsets = factors[:,:,:,None]*(deficit.reshape(-1,4)[:,:,None,None]*fn.reshape(-1,4,3)[:,:,None,:])
        # Average the supported sample constraints by rest area; do not sum a
        # larger correction merely because the mesh contains more samples.
        weights = (self.area[:,None,None]/4 * valid_face.reshape(-1,4,1)
                   * (self.bary[None,:,:]>0) * free[self.cloth_faces][:,None,:])
        indices = self.cloth_faces[:,None,:].expand(-1,4,-1).reshape(-1)
        numerator = torch.zeros_like(q_raw).index_add(0, indices, (weights[:,:,:,None]*offsets).reshape(-1,3))
        total = mass.new_zeros(len(mass)).index_add(0, indices, weights.reshape(-1))
        face_delta = numerator / total.clamp_min(1.e-20)[:,None]
        diagnostics = None
        if record:
            diagnostics = dict(mode='surface_face_floor', surface_vertex_support=int(supported.sum()),
                surface_vertex_active=int(((gap<epsilon)&supported).sum()), face_samples=len(samples),
                supported_face_samples=int(valid_face.sum()), active_face_samples=int((deficit>0).sum()),
                vertex_delta_rms_m=float(vertex_delta.detach().square().mean().sqrt()),
                face_delta_rms_m=float(face_delta.detach().square().mean().sqrt()),
                face_sample_gap_min_m=float(face_gap.detach()[valid_face].min()) if bool(valid_face.any()) else None,
                epsilon_m=epsilon, query_radius_m=radius, sweeps=1)
        return vertex_delta+face_delta, diagnostics
