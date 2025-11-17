# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
ONNX-compatible pose operations.

Replaces C++ pose clustering implementation with pure PyTorch operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def geodesic_distance_onnx(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance between rotation matrices.
    ONNX-compatible version.

    Geodesic distance on SO(3): θ = arccos((trace(R1^T R2) - 1) / 2)

    Args:
        R1: (B, 3, 3) or (3, 3) rotation matrices
        R2: (B, 3, 3) or (3, 3) rotation matrices

    Returns:
        distance: (B,) or scalar - geodesic distance in radians
    """
    if R1.ndim == 2:
        R1 = R1.unsqueeze(0)
    if R2.ndim == 2:
        R2 = R2.unsqueeze(0)

    # Ensure same batch size
    if R1.shape[0] != R2.shape[0]:
        if R1.shape[0] == 1:
            R1 = R1.expand(R2.shape[0], -1, -1)
        elif R2.shape[0] == 1:
            R2 = R2.expand(R1.shape[0], -1, -1)

    # Compute R1^T R2
    R_diff = R1.transpose(-2, -1) @ R2  # (B, 3, 3)

    # Trace
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]  # (B,)

    # Geodesic distance: arccos((trace - 1) / 2)
    # Clamp to avoid numerical issues with arccos
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    distance = torch.acos(cos_angle)  # (B,)

    return distance


def translation_distance_onnx(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance between translations.
    ONNX-compatible version.

    Args:
        t1: (B, 3) or (3,) translations
        t2: (B, 3) or (3,) translations

    Returns:
        distance: (B,) or scalar - Euclidean distance
    """
    if t1.ndim == 1:
        t1 = t1.unsqueeze(0)
    if t2.ndim == 1:
        t2 = t2.unsqueeze(0)

    # Euclidean distance
    distance = torch.norm(t1 - t2, dim=-1)  # (B,)

    return distance


def pose_distance_onnx(
    T1: torch.Tensor,
    T2: torch.Tensor,
    weight_rot: float = 1.0,
    weight_trans: float = 1.0
) -> torch.Tensor:
    """
    Compute combined pose distance (rotation + translation).
    ONNX-compatible version.

    Args:
        T1: (B, 4, 4) or (4, 4) SE(3) transformations
        T2: (B, 4, 4) or (4, 4) SE(3) transformations
        weight_rot: weight for rotation distance
        weight_trans: weight for translation distance

    Returns:
        distance: (B,) or scalar - combined distance
    """
    if T1.ndim == 2:
        T1 = T1.unsqueeze(0)
    if T2.ndim == 2:
        T2 = T2.unsqueeze(0)

    # Extract rotation and translation
    R1 = T1[:, :3, :3]  # (B, 3, 3)
    t1 = T1[:, :3, 3]  # (B, 3)
    R2 = T2[:, :3, :3]  # (B, 3, 3)
    t2 = T2[:, :3, 3]  # (B, 3)

    # Compute distances
    rot_dist = geodesic_distance_onnx(R1, R2)  # (B,)
    trans_dist = translation_distance_onnx(t1, t2)  # (B,)

    # Combined distance
    distance = weight_rot * rot_dist + weight_trans * trans_dist

    return distance


def cluster_poses_onnx(
    poses: torch.Tensor,
    angle_diff: float = 15.0,
    dist_diff: float = 0.01,
    symmetry_tfs: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cluster poses based on geodesic distance for rotation and Euclidean for translation.
    ONNX-compatible version of cluster_poses() from mycpp/Utils.cpp.

    Uses agglomerative clustering with distance thresholds.

    Args:
        poses: (N, 4, 4) SE(3) transformations
        angle_diff: angular threshold in degrees
        dist_diff: translation distance threshold in meters
        symmetry_tfs: (M, 4, 4) symmetry transformations (optional)

    Returns:
        cluster_centers: (K, 4, 4) representative poses for each cluster
        cluster_labels: (N,) cluster assignment for each input pose
    """
    N = poses.shape[0]
    device = poses.device
    dtype = poses.dtype

    # Convert angle threshold to radians
    angle_threshold = math.radians(angle_diff)

    # If symmetry transformations provided, augment poses
    if symmetry_tfs is not None:
        M = symmetry_tfs.shape[0]
        # Create augmented poses: poses @ symmetry_tfs
        poses_aug = poses.unsqueeze(1) @ symmetry_tfs.unsqueeze(0)  # (N, M, 4, 4)
        poses_aug = poses_aug.reshape(-1, 4, 4)  # (N*M, 4, 4)
    else:
        poses_aug = poses
        M = 1

    N_aug = poses_aug.shape[0]

    # Compute pairwise distances
    # Extract rotations and translations
    R_aug = poses_aug[:, :3, :3]  # (N_aug, 3, 3)
    t_aug = poses_aug[:, :3, 3]  # (N_aug, 3)

    # Pairwise geodesic distances (rotation)
    R1 = R_aug.unsqueeze(1)  # (N_aug, 1, 3, 3)
    R2 = R_aug.unsqueeze(0)  # (1, N_aug, 3, 3)
    R_diff = R1.transpose(-2, -1) @ R2  # (N_aug, N_aug, 3, 3)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]  # (N_aug, N_aug)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    rot_dist = torch.acos(cos_angle)  # (N_aug, N_aug)

    # Pairwise Euclidean distances (translation)
    t1 = t_aug.unsqueeze(1)  # (N_aug, 1, 3)
    t2 = t_aug.unsqueeze(0)  # (1, N_aug, 3)
    trans_dist = torch.norm(t1 - t2, dim=-1)  # (N_aug, N_aug)

    # Combined distance matrix
    # Two poses are similar if BOTH rotation and translation are within thresholds
    is_similar = (rot_dist < angle_threshold) & (trans_dist < dist_diff)  # (N_aug, N_aug)

    # Greedy clustering
    cluster_labels = -torch.ones(N_aug, dtype=torch.long, device=device)
    cluster_centers_list = []
    current_cluster = 0

    for i in range(N_aug):
        if cluster_labels[i] >= 0:
            continue  # Already assigned

        # Find all poses similar to this one
        similar_mask = is_similar[i]  # (N_aug,)

        # Assign cluster label
        cluster_labels[similar_mask] = current_cluster

        # Compute cluster center (mean pose - approximate)
        cluster_poses = poses_aug[similar_mask]  # (K, 4, 4)

        # For simplicity, use the first pose as representative
        # (More sophisticated: compute mean on manifold)
        cluster_center = cluster_poses[0]

        cluster_centers_list.append(cluster_center)
        current_cluster += 1

    # Stack cluster centers
    if len(cluster_centers_list) > 0:
        cluster_centers = torch.stack(cluster_centers_list, dim=0)  # (K, 4, 4)
    else:
        cluster_centers = torch.empty((0, 4, 4), device=device, dtype=dtype)

    # Map back to original poses (without symmetry augmentation)
    if symmetry_tfs is not None:
        # Take the minimum cluster label across symmetries
        cluster_labels_orig = cluster_labels.reshape(N, M)  # (N, M)
        cluster_labels_orig = cluster_labels_orig.min(dim=1)[0]  # (N,)
    else:
        cluster_labels_orig = cluster_labels

    return cluster_centers, cluster_labels_orig


def average_poses_onnx(poses: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Compute weighted average of SE(3) poses.
    ONNX-compatible version using quaternion averaging for rotations.

    Args:
        poses: (N, 4, 4) SE(3) transformations
        weights: (N,) weights for averaging (optional, default: uniform)

    Returns:
        avg_pose: (4, 4) average pose
    """
    N = poses.shape[0]
    device = poses.device
    dtype = poses.dtype

    if weights is None:
        weights = torch.ones(N, device=device, dtype=dtype) / N
    else:
        weights = weights / weights.sum()

    # Extract rotations and translations
    R = poses[:, :3, :3]  # (N, 3, 3)
    t = poses[:, :3, 3]  # (N, 3)

    # Average translation (weighted)
    t_avg = (weights.unsqueeze(-1) * t).sum(dim=0)  # (3,)

    # Average rotation: convert to quaternions, average, convert back
    # This is an approximation; exact Frechet mean on SO(3) is more complex
    from .geometric_ops import matrix_to_axis_angle_onnx, so3_exp_map_onnx

    # Convert to axis-angle
    axis_angles = matrix_to_axis_angle_onnx(R)  # (N, 3)

    # Weighted average of axis-angles (approximation)
    axis_angle_avg = (weights.unsqueeze(-1) * axis_angles).sum(dim=0)  # (3,)

    # Convert back to rotation matrix
    R_avg = so3_exp_map_onnx(axis_angle_avg.unsqueeze(0)).squeeze(0)  # (3, 3)

    # Construct average pose
    avg_pose = torch.eye(4, device=device, dtype=dtype)
    avg_pose[:3, :3] = R_avg
    avg_pose[:3, 3] = t_avg

    return avg_pose


def sample_rotation_grid_onnx(
    n_rotations_inplane: int = 60,
    n_views: int = 40
) -> torch.Tensor:
    """
    Sample rotations on a grid (for pose hypothesis generation).
    ONNX-compatible version.

    Samples viewpoints on icosphere, then in-plane rotations.

    Args:
        n_rotations_inplane: number of in-plane rotations per viewpoint
        n_views: number of viewpoints (approximate)

    Returns:
        rotations: (n_views * n_rotations_inplane, 3, 3) rotation matrices
    """
    device = torch.device('cpu')  # Use CPU for initialization
    dtype = torch.float32

    # Generate icosphere vertices for viewpoints
    # Approximate with Fibonacci sphere
    indices = torch.arange(n_views, dtype=dtype, device=device)
    phi = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))  # Golden angle

    y = 1 - (indices / (n_views - 1)) * 2  # Range [-1, 1]
    radius = torch.sqrt(1 - y * y)

    theta = phi * indices

    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    viewpoints = torch.stack([x, y, z], dim=-1)  # (n_views, 3)

    # For each viewpoint, create camera-looking-at-origin rotation
    rotations_list = []

    up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    for i in range(n_views):
        # Camera position
        cam_pos = viewpoints[i]  # (3,)

        # Z-axis: from camera to origin (viewing direction)
        z_axis = -F.normalize(cam_pos.unsqueeze(0), dim=-1).squeeze(0)  # (3,)

        # X-axis: perpendicular to up and z
        x_axis = torch.cross(up, z_axis)
        x_axis_norm = x_axis.norm()

        if x_axis_norm < 1e-6:  # Degenerate case (camera looking straight up/down)
            x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        else:
            x_axis = x_axis / x_axis_norm

        # Y-axis: perpendicular to x and z
        y_axis = torch.cross(z_axis, x_axis)

        # Base rotation matrix
        R_base = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (3, 3)

        # In-plane rotations
        for j in range(n_rotations_inplane):
            angle = 2.0 * math.pi * j / n_rotations_inplane

            # Rotation around z-axis (in-plane)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            R_inplane = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=dtype, device=device)

            # Combined rotation
            R = R_base @ R_inplane  # (3, 3)

            rotations_list.append(R)

    rotations = torch.stack(rotations_list, dim=0)  # (n_views * n_rotations_inplane, 3, 3)

    return rotations


class PoseOpsONNX(nn.Module):
    """
    Module containing all pose operations for ONNX export.
    """

    def __init__(
        self,
        angle_diff: float = 15.0,
        dist_diff: float = 0.01
    ):
        super().__init__()
        self.angle_diff = angle_diff
        self.dist_diff = dist_diff

    def geodesic_distance(self, R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        return geodesic_distance_onnx(R1, R2)

    def translation_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return translation_distance_onnx(t1, t2)

    def pose_distance(
        self,
        T1: torch.Tensor,
        T2: torch.Tensor,
        weight_rot: float = 1.0,
        weight_trans: float = 1.0
    ) -> torch.Tensor:
        return pose_distance_onnx(T1, T2, weight_rot, weight_trans)

    def cluster_poses(
        self,
        poses: torch.Tensor,
        symmetry_tfs: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cluster_poses_onnx(
            poses,
            self.angle_diff,
            self.dist_diff,
            symmetry_tfs
        )

    def average_poses(
        self,
        poses: torch.Tensor,
        weights: torch.Tensor = None
    ) -> torch.Tensor:
        return average_poses_onnx(poses, weights)

    def forward(self, poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Default forward: cluster poses."""
        return self.cluster_poses(poses)
