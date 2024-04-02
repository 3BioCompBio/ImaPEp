from typing import *

import torch
from sklearn.decomposition import PCA
from sklearn.metrics import *
from torch import Tensor, tensor


def run_pca(X: Tensor):
    device = X.device
    pca = PCA(2)
    X = X.cpu()
    pca.fit(X)
    X_dimen = pca.transform(X)
    return (
        pca, 
        tensor(X_dimen, device=device, dtype=torch.float32), 
        torch.cross(tensor(pca.components_[0], device=device, 
                           dtype=torch.float32), 
                    tensor(pca.components_[1], device=device,
                           dtype=torch.float32))
    )


def project_point(X: Tensor, plane: Tensor):
    """Project points onto the specified plane.

    Args:
        X (Tensor): Point coordinates with shape [N,3] or [3].
        plane (Tensor): The plane on which the ponits are projected
            on. This should be a tensor of shape [4] containing the
            four coefficients of the plane.

    Returns:
        Tensor: Coordinates of the projected point(s).
    """
    device = X.device
    if device != plane.device:
        raise ValueError("`X` and `plane` are on different devices")
    if X.dim() == 1:
        X = X.unsqueeze(0)  # [1,3]
    intercept = torch.ones(X.size(0), device=device).unsqueeze(-1)  # [N,1]
    X_expand = torch.cat([X, intercept], dim=-1)  # [N,4]
    t = (torch.sum(X_expand * plane, dim=-1) 
         / torch.sum(plane[:3]**2))  # [N]
    X_proj = X - plane[:3].unsqueeze(0) * t.unsqueeze(-1)
    if X.dim() == 1:
        X_proj = X_proj.squeeze(0)
    return X_proj  # [N,3] or [3]


def superpose(X_1: Tensor, X_2: Tensor):
    """Superpose two sets of 3-D points using PCA coordinates systems. 

    Two PCA are respetively run on the two sets of points `X_1` and 
    `X_2` which define two coordinates systems. The points in `X_2` 
    are rotated to make its PCA coordinates system the same as that 
    of `X_1`. 

    Args:
        X_1 (Tensor): Point set 1 with shape `[N_1,3]`.
        X_2 (Tensor): Point set 2 with shape `[N_2,3]`.

    Returns:
        Tuple: The 2-D coordinates of dimensionality, the `PCA` 
        objects and the normals of the two before-superposion PCA 
        planes.  
    """
    device = X_1.device
    if device != X_2.device:
        raise ValueError("`X_1` and `X_2` are on different devices")
    pca1, X_pca_1, normal1 = run_pca(X_1)
    pca2, _, normal2 = run_pca(X_2)
    # Directional vector of intersecting line of the two planes
    intersect_dv = torch.cross(normal1, normal2)
    intersect_dv /= torch.norm(intersect_dv)  # normalize
    # cos(angle) = (a·b)/(|a|*|b|)
    cos = torch.abs(
        (normal1.dot(normal2)
         / (torch.norm(normal1) * torch.norm(normal2)))
    ).clamp(min=0, max=1)
    sin = torch.sqrt(1 - cos ** 2).clamp(min=0, max=1)  # sin=(1-cos**2)**0.5
    centralized_X_2 = tensor(X_2 - pca2.mean_, dtype=torch.float32,
                             device=device)  # [N,3]
    normal2_expand = torch.cat([normal2, tensor([0])])  # [4]
    # [N,3]
    points_proj = project_point(centralized_X_2, normal2_expand)
    # print("points_proj", points_proj.type())
    # print("cos", cos.type())
    # print("intersect_dv", intersect_dv.type())
    rot_X_2 = (cos * points_proj  # [N,3]
               + sin 
               * torch.cross(intersect_dv.unsqueeze(0), points_proj, dim=1)
               + points_proj.matmul(intersect_dv).unsqueeze(-1)  # [N]
               * (1 - cos) * intersect_dv)  # [N,3]
    # Note: use pca1 instead of pca2
    X_pca_2 = pca1.transform(rot_X_2)
    X_pca_2 -= X_pca_2.mean(axis=0)  # note X_pca_2 is numpy array
    X_pca_2 = tensor(X_pca_2, dtype=torch.float32, device=device)
    return X_pca_1, X_pca_2, pca1, pca2, normal1, normal2


def distance_to_plane(X: Tensor, plane: Tensor) -> Tensor:
    """ Calculate distance from a point to a plane.

    Args:
        points (Tensor): [N,3] or [3]-shaped Tensor
        plane (Tensor): [4]-shaped Tensor. 
    """
    device = X.device
    if device != plane.device:
        raise ValueError("`X` and `plane` are on different devices")
    if X.dim == 1:
        X = X.unsqueeze(0)  # [1,3]
    plane_normal = plane[:3]  # [3]
    plane_D_value = plane[3]  # [1]
    return (torch.abs(X.matmul(plane_normal) + plane_D_value)
            / torch.norm(plane_normal)).squeeze()  # [N]


def get_collinear_point(direct: Tensor, ptr: Tensor, dist: float):
    r"""
    OB = OA + AB
    In the returned points, the former one is in front of the central
    point according to the direction of the normal.
    """
    direct = direct / direct.norm()
    ptr_b1 = ptr + dist * direct
    ptr_b2 = ptr - dist * direct
    return ptr_b1, ptr_b2


def rotate_around_axis(X: Tensor, axis: Tensor, angle: float):
    if X.dim == 1:
        X = X.unsqueeze(-1)
    axis /= torch.norm(axis)
    nx, ny, nz = axis[0], axis[1], axis[2]
    sint, cost = torch.sin(angle), torch.cos(angle)
    rot_mat = tensor([
        [nx**2*(1-cost)+cost, nx*ny*(1-cost)-nz*sint, nx*nz*(1-cost)+ny*sint],
        [nx*ny*(1-cost)+nz*sint, ny**2*(1-cost)+cost, ny*nz*(1-cost)-nx*sint],
        [nx*nz*(1-cost)-ny*sint, ny*nz*(1-cost)+nx*sint, nz**2*(1-cost)+cost]
    ])
    Y = torch.matmul(rot_mat, X.t()).t()  # [N,3]
    return Y


# X_a:[N_a,3], X_b:[N_b,3]
def num_adjacent_residues(X_a: Tensor, X_b: Tensor, threshold: float):
    dx = X_a.unsqueeze(1) - X_b.unsqueeze(0)  # [N_a,N_b,3]
    dist_mat = torch.sqrt(torch.sum(dx**2, dim=-1).clamp(min=0))  # [N_a,N_b]
    num_adj_a = torch.count_nonzero((dist_mat<threshold).int(), dim=1)  # [N_a]
    num_adj_b = torch.count_nonzero((dist_mat<threshold).int(), dim=0)  # [N_b]
    return num_adj_a, num_adj_b


class Cylinder(object):
    def __init__(self, bc1: Tensor, bc2: Tensor, r):
        self.params = (bc1, bc2, r)
        self.bc1 = bc1
        self.bc2 = bc2
        self.r = r
        self.h = torch.norm(bc1-bc2)
        self.ctr = (bc1+bc2)/2

    def is_inside(self, q: Tensor):
        c1, c2, r = self.bc1, self.bc2, self.r
        axis = c2 - c1  # [3]
        const = r * torch.norm(axis)
        if q.dim() == 1:
            q = q.unsqueeze(0)  # [1,3]
        relative_pos1 = (torch.matmul(q-c1, axis) > 0).int()  # [N]
        relative_pos2 = (torch.matmul(q-c2, axis) < 0).int()  # [N]
        axis_expand = axis.unsqueeze(0).expand(q.size(0), -1)  # [N,3]
        between = (
            torch.norm(torch.cross(q-c1, axis_expand, dim=1), dim=1) < const
        ).int()  # [N]
        results = relative_pos1 * relative_pos2 * between
        return results


def build_patch_cylinder(X: Tensor):
    device = X.device
    pca, _, normal = run_pca(X)

    # D of the PCA plane
    D = -torch.dot(normal, tensor(pca.mean_, device=device, dtype=torch.float32))  # the intercept, [0]
    X_proj = project_point(X, torch.cat([normal, D.unsqueeze(0)]))  # [N,3]
    
    center = X_proj.mean(dim=0).squeeze()  # p1 of cylinder [3]
    r_cylinder = torch.max(torch.norm(X_proj - center, dim=1))  # [N]
    
    # D_1-D_2 = ±norm * h, D_1 is the larger intercept
    # D of the plane on which the base circle 1 locates
    D_side1 = D + 4 * r_cylinder * torch.norm(normal)  # [1]
    cc_side1 = project_point(center, torch.cat([normal, D_side1.unsqueeze(0)])).squeeze()  # [3]  
    # D of the plane on which the base circle 2 locates
    D_side2 = D - 4 * r_cylinder * torch.norm(normal)  # [1]
    cc_side2 = project_point(center, torch.cat([normal, D_side2.unsqueeze(0)])).squeeze()  # [3]
    
    return Cylinder(center, cc_side1, r_cylinder), Cylinder(center, cc_side2, r_cylinder)


def tensor_to_list(X):
    if isinstance(X, Tensor):
        X_new = X.tolist()
    elif isinstance(X, (list, tuple, set)):
        X_new = []
        for item in X:
            X_new.append(tensor_to_list(item))
        X_new = tuple(X_new)
    elif isinstance(X, dict):
        X_new = {}
        for k, v in X.items():
            X_new[k] = tensor_to_list(v)
    else:
        X_new = X
    return X_new
