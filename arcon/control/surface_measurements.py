"""Surface-distance and material-deformation measurements on saved geometry."""
import numpy as np
import warp as wp

@wp.kernel
def signed_query(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), result: wp.array(dtype=float)):
    i=wp.tid()
    q=wp.mesh_query_point_sign_winding_number(mesh,points[i],1.0e6)
    result[i]=1.0e20
    if q.result:
        closest=wp.mesh_eval_position(mesh,q.face,q.u,q.v)
        result[i]=q.sign*wp.length(points[i]-closest)

@wp.kernel
def closest_query(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), distance: wp.array(dtype=float), closest: wp.array(dtype=wp.vec3), outward: wp.array(dtype=wp.vec3)):
    i=wp.tid()
    q=wp.mesh_query_point_sign_winding_number(mesh,points[i],1.0e6)
    distance[i]=1.0e20
    if q.result:
        c=wp.mesh_eval_position(mesh,q.face,q.u,q.v)
        length=wp.length(points[i]-c)
        closest[i]=c
        distance[i]=q.sign*length
        if length>1.0e-12:
            outward[i]=q.sign*(points[i]-c)/length
        else:
            a=wp.mesh_eval_position(mesh,q.face,1.0,0.0)
            b=wp.mesh_eval_position(mesh,q.face,0.0,1.0)
            z=wp.mesh_eval_position(mesh,q.face,0.0,0.0)
            outward[i]=wp.normalize(wp.cross(b-a,z-a))

@wp.kernel
def material_query(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), face_ids: wp.array(dtype=int), barycentric: wp.array(dtype=wp.vec3), distance: wp.array(dtype=float)):
    i=wp.tid()
    q=wp.mesh_query_point_sign_winding_number(mesh,points[i],1.0e6)
    face_ids[i]=-1
    distance[i]=1.0e20
    if q.result:
        face_ids[i]=q.face
        barycentric[i]=wp.vec3(q.u,q.v,1.0-q.u-q.v)
        closest=wp.mesh_eval_position(mesh,q.face,q.u,q.v)
        distance[i]=wp.length(points[i]-closest)

class SurfaceDistance:
    def __init__(self, vertices, faces, device='cuda:0'):
        self.device=device
        self.points=wp.array(np.asarray(vertices,dtype=np.float32),dtype=wp.vec3,device=device)
        self.mesh=wp.Mesh(points=self.points,indices=wp.array(np.asarray(faces,dtype=np.int32).ravel(),dtype=int,device=device),support_winding_number=True)
    def update(self, vertices):
        self.points.assign(np.asarray(vertices,dtype=np.float32))
        self.mesh.refit()
    def query(self, points):
        query=wp.array(np.asarray(points,dtype=np.float32),dtype=wp.vec3,device=self.device)
        out=wp.empty(len(points),dtype=float,device=self.device)
        wp.launch(signed_query,dim=len(points),inputs=[self.mesh.id,query,out],device=self.device)
        values=out.numpy().astype(np.float64)
        if not np.isfinite(values).all() or np.any(np.abs(values)>=1e19):
            raise RuntimeError('Closest-triangle query did not return a finite surface distance')
        return values

    def details(self, points):
        query=wp.array(np.asarray(points,dtype=np.float32),dtype=wp.vec3,device=self.device)
        distances=wp.empty(len(points),dtype=float,device=self.device)
        closest=wp.empty(len(points),dtype=wp.vec3,device=self.device)
        outward=wp.empty(len(points),dtype=wp.vec3,device=self.device)
        wp.launch(closest_query,dim=len(points),inputs=[self.mesh.id,query,distances,closest,outward],device=self.device)
        d,c,n=distances.numpy(),closest.numpy(),outward.numpy()
        if not np.isfinite(d).all() or np.any(np.abs(d)>=1e19):
            raise RuntimeError('Closest surface did not return finite geometry')
        return d,c,n

    def material_coordinates(self, points):
        """Freeze a closest-rest-surface face/barycentric map, not a time-varying match."""
        query=wp.array(np.asarray(points,dtype=np.float32),dtype=wp.vec3,device=self.device)
        ids=wp.empty(len(points),dtype=int,device=self.device)
        bary=wp.empty(len(points),dtype=wp.vec3,device=self.device)
        distances=wp.empty(len(points),dtype=float,device=self.device)
        wp.launch(material_query,dim=len(points),inputs=[self.mesh.id,query,ids,bary,distances],device=self.device)
        f,b,d=ids.numpy(),bary.numpy().astype(np.float64),distances.numpy().astype(np.float64)
        if np.any(f<0) or not np.isfinite(d).all():
            raise RuntimeError('Rest-surface mapping did not return finite face coordinates')
        return f,b,d

def rest_weights(vertices,faces):
    v=np.asarray(vertices,dtype=np.float64); f=np.asarray(faces,dtype=np.int64)
    area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=-1)/2
    weights=np.zeros(len(v))
    for k in range(3): np.add.at(weights,f[:,k],area/3)
    return weights/weights.sum(), area

def unique_edges(faces):
    f=np.asarray(faces,dtype=np.int64)
    return np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
