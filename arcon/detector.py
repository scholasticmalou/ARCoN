"""ContourCraft native detector adapter; see THIRD_PARTY_NOTICES.md and its MIT license."""
import cccollisions as collisions

def find_close_faces(vertices, faces, threshold=0):
    """
    vertices: Vx3
    faces: Fx3

    return: pairs of (almost) copccraftlliding faces Nx2
    """
    # threshold = 0
    triangles = vertices[faces].unsqueeze(dim=0).contiguous()
    bboxes, tree = collisions.bvh(triangles)
    face_pairs = collisions.find_collisions(bboxes, tree, triangles, threshold, 320)[0]
    face_pairs = face_pairs[face_pairs[:, 0] >= 0, :]

    return face_pairs
