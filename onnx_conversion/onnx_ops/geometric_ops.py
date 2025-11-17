# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
ONNX-compatible geometric operations.

Replaces custom CUDA implementations with pure PyTorch operations
that are fully supported by ONNX export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def rotation_6d_to_matrix_onnx(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    ONNX-compatible version using only standard PyTorch ops.

    Based on: "On the Continuity of Rotation Representations in Neural Networks"
    Zhou et al., CVPR 2019

    Args:
        d6: (B, 6) tensor representing rotation as first two columns of rotation matrix

    Returns:
        R: (B, 3, 3) rotation matrices
    """
    # Reshape to (B, 2, 3) - first two columns
    a1 = d6[..., :3]  # (B, 3)
    a2 = d6[..., 3:6]  # (B, 3)

    # Normalize first column
    b1 = F.normalize(a1, dim=-1)  # (B, 3)

    # Gram-Schmidt orthogonalization for second column
    dot = (b1 * a2).sum(dim=-1, keepdim=True)  # (B, 1)
    b2 = a2 - dot * b1  # (B, 3)
    b2 = F.normalize(b2, dim=-1)  # (B, 3)

    # Third column via cross product
    b3 = torch.cross(b1, b2, dim=-1)  # (B, 3)

    # Stack into rotation matrix
    R = torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)

    return R


def so3_exp_map_onnx(log_rot: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from so(3) (axis-angle) to SO(3) (rotation matrix).
    ONNX-compatible version using Rodrigues' formula.

    Args:
        log_rot: (B, 3) axis-angle representation

    Returns:
        R: (B, 3, 3) rotation matrices
    """
    batch_size = log_rot.shape[0]
    device = log_rot.device
    dtype = log_rot.dtype

    # Compute rotation angle
    theta = torch.norm(log_rot, dim=-1, keepdim=True)  # (B, 1)

    # Avoid division by zero
    eps = 1e-6
    theta = torch.clamp(theta, min=eps)

    # Normalize to get rotation axis
    axis = log_rot / theta  # (B, 3)

    # Rodrigues' formula components
    cos_theta = torch.cos(theta)  # (B, 1)
    sin_theta = torch.sin(theta)  # (B, 1)
    one_minus_cos = 1.0 - cos_theta  # (B, 1)

    # Skew-symmetric matrix [axis]_x
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    zeros = torch.zeros_like(x)

    K = torch.cat([
        torch.cat([zeros, -z, y], dim=-1),
        torch.cat([z, zeros, -x], dim=-1),
        torch.cat([-y, x, zeros], dim=-1)
    ], dim=-2).reshape(batch_size, 3, 3)  # (B, 3, 3)

    # Outer product axis ⊗ axis
    axis_outer = axis.unsqueeze(-1) @ axis.unsqueeze(-2)  # (B, 3, 3)

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    # K² = (axis ⊗ axis) - I
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, 3, 3)

    R = I + sin_theta.unsqueeze(-1) * K + one_minus_cos.unsqueeze(-1) * (axis_outer - I)

    return R


def se3_exp_map_onnx(log_transform: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from se(3) to SE(3).

    Args:
        log_transform: (B, 6) where [:3] is translation, [3:] is rotation (axis-angle)

    Returns:
        T: (B, 4, 4) transformation matrices
    """
    batch_size = log_transform.shape[0]
    device = log_transform.device
    dtype = log_transform.dtype

    trans = log_transform[..., :3]  # (B, 3)
    rot = log_transform[..., 3:6]  # (B, 3)

    # Get rotation matrix
    R = so3_exp_map_onnx(rot)  # (B, 3, 3)

    # Build SE(3) matrix
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, 4, 4).clone()
    T[:, :3, :3] = R
    T[:, :3, 3] = trans

    return T


def transform_pts_onnx(pts: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Transform 3D points by SE(3) transformation.
    ONNX-compatible version.

    Args:
        pts: (N, 3) or (B, N, 3) points
        transform: (4, 4) or (B, 4, 4) transformation matrices

    Returns:
        pts_transformed: same shape as pts
    """
    input_2d = pts.ndim == 2
    if input_2d:
        pts = pts.unsqueeze(0)  # (1, N, 3)

    if transform.ndim == 2:
        transform = transform.unsqueeze(0)  # (1, 4, 4)

    # Homogeneous coordinates
    ones = torch.ones((*pts.shape[:-1], 1), device=pts.device, dtype=pts.dtype)
    pts_homo = torch.cat([pts, ones], dim=-1)  # (B, N, 4)

    # Apply transformation
    pts_transformed = (transform.unsqueeze(1) @ pts_homo.unsqueeze(-1)).squeeze(-1)  # (B, N, 4)
    pts_transformed = pts_transformed[..., :3]  # (B, N, 3)

    if input_2d:
        pts_transformed = pts_transformed.squeeze(0)

    return pts_transformed


def barycentric_coords_3d_onnx(triangle: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Calculate barycentric coordinates for points in 3D triangles.
    ONNX-compatible version.

    Args:
        triangle: (B, 3, 3) or (3, 3) - three vertices of triangle
        points: (B, 3) or (N, 3) - query points

    Returns:
        weights: (B, 3) or (N, 3) barycentric coordinates
    """
    if triangle.ndim == 2:
        triangle = triangle.unsqueeze(0)  # (1, 3, 3)
    if points.ndim == 1:
        points = points.unsqueeze(0)  # (1, 3)

    # Expand for batch operations
    batch_size = max(triangle.shape[0], points.shape[0])
    if triangle.shape[0] == 1:
        triangle = triangle.expand(batch_size, -1, -1)
    if points.shape[0] == 1:
        points = points.expand(batch_size, -1)

    # Extract vertices
    v0 = triangle[:, 0, :]  # (B, 3)
    v1 = triangle[:, 1, :]  # (B, 3)
    v2 = triangle[:, 2, :]  # (B, 3)

    # Compute vectors
    v0v1 = v1 - v0  # (B, 3)
    v0v2 = v2 - v0  # (B, 3)
    v0p = points - v0  # (B, 3)

    # Compute dot products
    dot00 = (v0v1 * v0v1).sum(dim=-1)  # (B,)
    dot01 = (v0v1 * v0v2).sum(dim=-1)  # (B,)
    dot02 = (v0v1 * v0p).sum(dim=-1)  # (B,)
    dot11 = (v0v2 * v0v2).sum(dim=-1)  # (B,)
    dot12 = (v0v2 * v0p).sum(dim=-1)  # (B,)

    # Compute barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom  # w1
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom  # w2
    w = 1.0 - u - v  # w0

    weights = torch.stack([w, u, v], dim=-1)  # (B, 3)

    return weights


def barycentric_coords_2d_onnx(triangle: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Calculate barycentric coordinates for points in 2D triangles.
    ONNX-compatible version.

    Args:
        triangle: (B, 3, 2) or (3, 2) - three vertices of triangle in 2D
        points: (B, 2) or (N, 2) - query points

    Returns:
        weights: (B, 3) or (N, 3) barycentric coordinates
    """
    if triangle.ndim == 2:
        triangle = triangle.unsqueeze(0)  # (1, 3, 2)
    if points.ndim == 1:
        points = points.unsqueeze(0)  # (1, 2)

    # Expand for batch operations
    batch_size = max(triangle.shape[0], points.shape[0])
    if triangle.shape[0] == 1:
        triangle = triangle.expand(batch_size, -1, -1)
    if points.shape[0] == 1:
        points = points.expand(batch_size, -1)

    # Extract vertices
    v0 = triangle[:, 0, :]  # (B, 2)
    v1 = triangle[:, 1, :]  # (B, 2)
    v2 = triangle[:, 2, :]  # (B, 2)

    # Compute vectors
    CA = v0 - v2
    CP = points - v2
    AB = v1 - v0
    AC = -CA
    AP = points - v0

    # Compute determinants (2D cross products)
    def cross_2d(a, b):
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    numerator1 = cross_2d(CA, CP)
    numerator2 = cross_2d(AB, AP)
    denominator = cross_2d(AB, AC)

    # Avoid division by zero
    denominator = denominator + 1e-10

    w1 = numerator1 / denominator
    w2 = numerator2 / denominator
    w0 = 1.0 - w1 - w2

    weights = torch.stack([w0, w1, w2], dim=-1)  # (B, 3)

    return weights


def matrix_to_axis_angle_onnx(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to axis-angle representation.
    ONNX-compatible version.

    Args:
        R: (B, 3, 3) rotation matrices

    Returns:
        axis_angle: (B, 3) axis-angle representation
    """
    batch_size = R.shape[0]
    device = R.device
    dtype = R.dtype

    # Compute rotation angle
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # (B,)
    theta = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6))  # (B,)

    # Handle small angles (use first-order approximation)
    small_angle_mask = theta.abs() < 1e-4

    # For non-small angles, extract axis from skew-symmetric part
    axis = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=-1)  # (B, 3)

    # Normalize axis
    axis = axis / (2.0 * torch.sin(theta.unsqueeze(-1)) + 1e-10)

    # For small angles, use linear approximation
    axis_small = axis / (axis.norm(dim=-1, keepdim=True) + 1e-10)
    theta_small = axis.norm(dim=-1) / 2.0

    # Select based on angle magnitude
    theta_final = torch.where(small_angle_mask, theta_small, theta)
    axis_final = torch.where(small_angle_mask.unsqueeze(-1), axis_small, axis)

    # Combine into axis-angle
    axis_angle = axis_final * theta_final.unsqueeze(-1)

    return axis_angle


def quaternion_to_matrix_onnx(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    ONNX-compatible version.

    Args:
        quat: (B, 4) quaternions (w, x, y, z)

    Returns:
        R: (B, 3, 3) rotation matrices
    """
    # Normalize quaternion
    quat = F.normalize(quat, dim=-1)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Compute rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = torch.stack([
        torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1),
        torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1),
        torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)
    ], dim=-2)

    return R


class GeometricOpsONNX(nn.Module):
    """
    Module containing all geometric operations for ONNX export.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Placeholder - not meant to be called directly
        return x

    @staticmethod
    def rotation_6d_to_matrix(d6):
        return rotation_6d_to_matrix_onnx(d6)

    @staticmethod
    def so3_exp_map(log_rot):
        return so3_exp_map_onnx(log_rot)

    @staticmethod
    def transform_pts(pts, transform):
        return transform_pts_onnx(pts, transform)

    @staticmethod
    def barycentric_coords_3d(triangle, points):
        return barycentric_coords_3d_onnx(triangle, points)

    @staticmethod
    def barycentric_coords_2d(triangle, points):
        return barycentric_coords_2d_onnx(triangle, points)
