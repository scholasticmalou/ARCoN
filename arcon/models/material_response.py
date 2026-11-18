"""Element-local material coordinates, Jacobians and bending-response assembly.

The reported model uses current-edge bending coordinates with material-geometry
conditioning. Additional branches support the original component comparisons."""
from __future__ import annotations

import torch
from models.core.adaptive_cell import ProposalRelationResponse
from models.core.relation_response import hinge_basis
from models.core.geometry_feedback import unique_hinge_gradient


def bending_coordinate(sample, q, theta, derivative, tri_ids, shared):
    cloth = sample['cloth']
    area = cloth.f_area[cloth.f_connectivity].sum(1).reshape(-1)
    stiffness = cloth.bending_coeff.reshape(-1)[0]
    return bending_elements(q, theta, derivative, tri_ids, shared, area, stiffness)


def bending_elements(q, theta, derivative, tri_ids, shared, area, stiffness):
    edge = q[shared[:, 1]]-q[shared[:, 0]]
    length = edge.norm(dim=-1)
    tangent = edge/length[:, None]
    # l uses the shared edge in the first triangle only; do not duplicate dl.
    signs = ((tri_ids[:, 0] == shared[:, 1, None]).to(q.dtype)
             -(tri_ids[:, 0] == shared[:, 0, None]).to(q.dtype))
    first = signs[:, :, None]*tangent[:, None]
    dl = torch.stack((first, torch.zeros_like(first)), 1)
    scale = torch.sqrt(stiffness/(4*area))
    coordinate = scale*length*theta
    jacobian_slots = scale[:, None, None, None]*(length[:, None, None, None]*derivative
                                                  +theta[:, None, None, None]*dl)
    unique, jacobian = unique_hinge_gradient(jacobian_slots, tri_ids, shared)
    return coordinate, jacobian_slots, unique, jacobian


def membrane_elements(sample, q, face_indices, thickness):
    cloth = sample['cloth']
    faces = cloth.faces_batch.T[face_indices]
    tri = q[faces]
    dm = cloth.Dm_inv[face_indices]
    deformation = torch.stack((tri[:, 0]-tri[:, 2], tri[:, 1]-tri[:, 2]), -1) @ dm
    strain = .5*(deformation.transpose(-1, -2) @ deformation-torch.eye(2, device=q.device, dtype=q.dtype))
    basis = torch.cat((dm, -dm.sum(1, keepdim=True)), 1)
    jac = .5*(torch.einsum('fia,fdb->fabid', basis, deformation)
              +torch.einsum('fib,fda->fabid', basis, deformation))
    area = cloth.f_area[face_indices].reshape(-1)
    mu, lam = (cloth[k].reshape(-1)[0] for k in ('lame_mu', 'lame_lambda'))
    shear = torch.sqrt(2*area*thickness*mu)
    volume = torch.sqrt(area*thickness*lam)
    coordinate = torch.cat((shear[:, None]*strain.reshape(-1, 4),
                            volume[:, None]*strain.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None]), -1)
    jacobian = torch.cat((shear[:, None, None, None]*jac.reshape(-1, 4, 3, 3),
                          volume[:, None, None, None]*(jac[:, 0, 0]+jac[:, 1, 1])[:, None]), 1)
    return coordinate, faces, jacobian


class MaterialReadout(ProposalRelationResponse):
    def __init__(self, arm, thickness):
        super().__init__()
        if arm not in ('A', 'B', 'AB', 'C') or thickness <= 0:
            raise ValueError('Use the approved A/B/AB/C and native positive membrane thickness')
        self.material_arm = arm
        self.elastic_bending = arm in ('A', 'AB', 'C')
        self.elastic_contact = arm in ('B', 'AB')
        self.material_thickness = thickness
        self.material_bending_record = None
        self.material_contact_record = None
        self.curvature_stencils = {}
        self.material_curvature_record = None

    def assemble_hinge(self, sample, q, mass, rest, theta, derivative, tri_ids,
                       shared, hinge_h, hinge_a, material_multiplier=None):
        if not self.elastic_bending:
            return super().assemble_hinge(sample, q, mass, rest, theta, derivative,
                                          tri_ids, shared, hinge_h, hinge_a, material_multiplier)
        if material_multiplier is None:
            raise ValueError('A requires the explicit per-cell multiplier tensor')
        c, slots, unique, jac = bending_coordinate(sample, q, theta, derivative, tri_ids, shared)
        free = sample['cloth'].vertex_type.reshape(-1) != 3
        gram = (jac.square().sum(-1)*free[unique]/mass[unique]).sum(-1)
        dt2 = sample['cloth'].timestep.reshape(-1)[0].square()
        eta = (1+hinge_a)*.5
        curvature_gate = None
        if self.material_arm == 'C':
            from arcon.models.material_geometry import MaterialCurvatureStencil
            key = (str(sample.garment_name), len(q), len(shared), str(q.device))
            if key not in self.curvature_stencils:
                self.curvature_stencils[key] = MaterialCurvatureStencil(sample)
            stencil = self.curvature_stencils[key]
            curvature_gate = stencil.gate(theta)
            eta = eta*curvature_gate
        change = eta*(-c-material_multiplier/dt2)/(gram+1/dt2)
        impulse = slots*change[:, None, None, None]
        updated = material_multiplier+change
        if self.record:
            with torch.no_grad():
                self.material_bending_record = dict(energy=float((.5*c.square().sum()).detach()),
                    multiplier_input_l2=float(material_multiplier.detach().norm()),
                    multiplier_output_l2=float(updated.detach().norm()),
                    multiplier_update_l2=float(change.detach().norm()),
                    relaxation_mean=float(eta.detach().mean()), native_dt=float(dt2.detach().sqrt()),
                    coordinate='sqrt(kappa/(4*Arest_sum))*current_edge_length*signed_theta',
                    full_edge_length_derivative=True, old_hinge_and_J_gain_replaced=True)
                if curvature_gate is not None:
                    self.material_curvature_record = stencil.statistics(theta, curvature_gate)
        return impulse, change, updated

    def material_statistics(self):
        return dict(arm=self.material_arm, bending=self.material_bending_record,
                    contact=self.material_contact_record, curvature=self.material_curvature_record)

    def contact_stencil(self, sample, ci):
        cloth = sample['cloth']
        correspondence = self._contact_backward_correspondence
        if correspondence is not None and 'material_stencil' in correspondence:
            return correspondence['material_stencil']
        with torch.no_grad():
            domain = torch.zeros(len(cloth.pred_pos), dtype=torch.bool, device=ci.device)
            domain[ci] = True
            edges = sample['cloth', 'mesh_edge', 'cloth'].edge_index
            for _ in range(2):
                expanded = domain.clone()
                expanded[edges[1, domain[edges[0]]]] = True
                expanded[edges[0, domain[edges[1]]]] = True
                domain = expanded
            faces = cloth.faces_batch.T
            face_ids = torch.nonzero(domain[faces].any(-1), as_tuple=False).reshape(-1)
            hinge_ids = torch.nonzero(domain[faces[cloth.f_connectivity]].any(-1).any(-1), as_tuple=False).reshape(-1)
            free_domain = domain & (cloth.vertex_type.reshape(-1) != 3)
            result = free_domain, face_ids, hinge_ids
            if correspondence is not None:
                correspondence['material_stencil'] = result
            return result

    def coordinate_contact(self, sample, q, delta, ci, normals):
        if len(ci) == 0:
            return delta
        # If no violated free support exists, the original zero correction is
        # the exact minimum of this quadratic; avoid undefined zero derivatives.
        free = sample['cloth'].vertex_type.reshape(-1) != 3
        delta = delta*free[:, None]
        domain, face_ids, hinge_ids = self.contact_stencil(sample, ci)
        mass = sample['cloth'].v_mass.reshape(-1)
        dt2 = sample['cloth'].timestep.reshape(-1)[0].square()
        _, faces, membrane_jac = membrane_elements(sample, q, face_ids, self.material_thickness)
        cloth = sample['cloth']
        connectivity, shared = cloth.f_connectivity[hinge_ids], cloth.f_connectivity_edges[hinge_ids]
        theta, derivative, tri_ids, _ = hinge_basis(q, cloth.faces_batch.T, connectivity, shared)
        area = cloth.f_area[connectivity].sum(1).reshape(-1)
        _, _, bend_ids, bending_jac = bending_elements(q, theta, derivative, tri_ids, shared,
                                                       area, cloth.bending_coeff.reshape(-1)[0])

        def project(v):
            v = v*domain[:, None]
            at = v[ci]
            return v.index_copy(0, ci, at-(at*normals).sum(-1, keepdim=True)*normals)

        def apply_material(v):
            m = torch.einsum('fric,fic->fr', membrane_jac, v[faces])
            m_back = torch.einsum('fric,fr->fic', membrane_jac, m)
            b = (bending_jac*v[bend_ids]).sum((-1, -2))
            return (torch.zeros_like(v).index_add(0, faces.reshape(-1), m_back.reshape(-1, 3))
                    .index_add(0, bend_ids.reshape(-1), (bending_jac*b[:, None, None]).reshape(-1, 3)))

        diagonal = mass/dt2
        diagonal = diagonal.index_add(0, faces.reshape(-1), membrane_jac.square().sum((1, 3)).reshape(-1)/3)
        diagonal = diagonal.index_add(0, bend_ids.reshape(-1), bending_jac.square().sum(-1).reshape(-1)/3)
        correction = torch.zeros_like(q)
        step_lengths = []
        for _ in range(2):
            gradient = project(mass[:, None]/dt2*correction+apply_material(delta+correction))
            direction = -project(gradient/diagonal[:, None])
            h_direction = mass[:, None]/dt2*direction+apply_material(direction)
            denominator = (direction*h_direction).sum()
            numerator = -(gradient*direction).sum()
            valid = denominator > 0
            step = torch.where(valid, numerator/torch.where(valid, denominator, torch.ones_like(denominator)),
                               torch.zeros_like(denominator))
            correction = project(correction+step*direction)
            if self.record:
                step_lengths.append(float(step.detach()))
        if self.record:
            with torch.no_grad():
                before = .5*(delta*apply_material(delta)).sum()
                combined = delta+correction
                after = .5*(combined*apply_material(combined)).sum()+.5*(mass[:, None]/dt2*correction.square()).sum()
                self.material_contact_record = dict(local_nodes=int(domain.sum()),
                    membrane_faces=len(face_ids), bending_hinges=len(hinge_ids), iterations=2,
                    objective_before=float(before), objective_after=float(after), step_lengths=step_lengths,
                    normal_preservation_max_m=float((correction[ci]*normals).sum(-1).abs().max()),
                    correction_rms_m=float(correction.square().mean().sqrt()),
                    domain='two_material_mesh_hops;free_only', extra_body_queries=0,
                    original_membrane_thickness=self.material_thickness)
        return delta+correction
