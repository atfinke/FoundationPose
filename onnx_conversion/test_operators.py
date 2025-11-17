#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
Quick test script for ONNX operators.

Validates that all re-implemented operators produce correct outputs.
"""

import torch
import numpy as np
from onnx_ops.geometric_ops import *
from onnx_ops.depth_ops import *
from onnx_ops.ray_tracing import *
from onnx_ops.pose_ops import *


def test_geometric_ops():
    """Test geometric operations."""
    print("Testing geometric operations...")

    # Test rotation_6d_to_matrix
    d6 = torch.randn(5, 6)
    R = rotation_6d_to_matrix_onnx(d6)
    assert R.shape == (5, 3, 3), f"Expected (5,3,3), got {R.shape}"

    # Verify orthogonality: R @ R.T = I
    identity = torch.eye(3).unsqueeze(0).expand(5, 3, 3)
    orthogonality_error = (R @ R.transpose(-2, -1) - identity).abs().max()
    assert orthogonality_error < 1e-5, f"Not orthogonal: error={orthogonality_error}"

    # Test so3_exp_map
    axis_angle = torch.randn(5, 3) * 0.1
    R = so3_exp_map_onnx(axis_angle)
    assert R.shape == (5, 3, 3), f"Expected (5,3,3), got {R.shape}"

    # Test transform_pts
    pts = torch.randn(100, 3)
    T = torch.eye(4).unsqueeze(0).expand(5, 4, 4)
    pts_transformed = transform_pts_onnx(pts, T)
    assert pts_transformed.shape == (5, 100, 3), f"Expected (5,100,3), got {pts_transformed.shape}"

    # Test barycentric_coords_3d
    triangle = torch.rand(5, 3, 3)
    points = torch.rand(5, 3)
    weights = barycentric_coords_3d_onnx(triangle, points)
    assert weights.shape == (5, 3), f"Expected (5,3), got {weights.shape}"
    # Weights should sum to 1
    weight_sum = weights.sum(dim=-1)
    assert (weight_sum - 1.0).abs().max() < 1e-5, "Barycentric weights don't sum to 1"

    print("  All geometric operations passed")


def test_depth_ops():
    """Test depth operations."""
    print("Testing depth operations...")

    # Test depth_erosion
    depth = torch.rand(480, 640) * 5.0
    eroded = depth_erosion_onnx(depth, radius=2)
    assert eroded.shape == (480, 640), f"Expected (480,640), got {eroded.shape}"
    assert eroded.min() >= 0, "Negative depths after erosion"

    # Test bilateral_filter
    filtered = bilateral_filter_onnx(depth, radius=2)
    assert filtered.shape == (480, 640), f"Expected (480,640), got {filtered.shape}"

    # Test with batch
    depth_batch = torch.rand(2, 480, 640) * 5.0
    eroded_batch = depth_erosion_onnx(depth_batch, radius=2)
    assert eroded_batch.shape == (2, 480, 640), f"Expected (2,480,640), got {eroded_batch.shape}"

    print("  All depth operations passed")


def test_pose_ops():
    """Test pose operations."""
    print("Testing pose operations...")

    # Test geodesic_distance
    R1 = torch.eye(3).unsqueeze(0).expand(5, 3, 3)
    R2 = torch.eye(3).unsqueeze(0).expand(5, 3, 3)
    dist = geodesic_distance_onnx(R1, R2)
    assert dist.shape == (5,), f"Expected (5,), got {dist.shape}"
    assert dist.abs().max() < 1e-5, "Identity matrices should have zero distance"

    # Test pose_distance
    T1 = torch.eye(4).unsqueeze(0).expand(5, 4, 4)
    T2 = torch.eye(4).unsqueeze(0).expand(5, 4, 4)
    dist = pose_distance_onnx(T1, T2)
    assert dist.shape == (5,), f"Expected (5,), got {dist.shape}"

    # Test cluster_poses
    poses = torch.eye(4).unsqueeze(0).expand(10, 4, 4).clone()
    # Add small perturbations
    poses[:, :3, 3] = torch.randn(10, 3) * 0.01
    centers, labels = cluster_poses_onnx(poses, angle_diff=15.0, dist_diff=0.05)
    assert centers.shape[1:] == (4, 4), f"Expected (K,4,4), got {centers.shape}"
    assert labels.shape == (10,), f"Expected (10,), got {labels.shape}"

    print("  All pose operations passed")


def test_ray_tracing_ops():
    """Test ray tracing operations."""
    print("Testing ray tracing operations...")

    # Test ray_sphere_intersection
    ray_origins = torch.zeros(10, 3)
    ray_directions = torch.randn(10, 3)
    ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)
    sphere_center = torch.tensor([0., 0., 5.])
    sphere_radius = 1.0

    t_near, t_far = ray_sphere_intersection_onnx(ray_origins, ray_directions, sphere_center, sphere_radius)
    assert t_near.shape == (10,), f"Expected (10,), got {t_near.shape}"
    assert t_far.shape == (10,), f"Expected (10,), got {t_far.shape}"

    # Test ray_aabb_intersection
    aabb_min = torch.tensor([-1., -1., -1.])
    aabb_max = torch.tensor([1., 1., 1.])
    t_near, t_far = ray_aabb_intersection_onnx(ray_origins, ray_directions, aabb_min, aabb_max)
    assert t_near.shape == (10,), f"Expected (10,), got {t_near.shape}"

    print("  All ray tracing operations passed")


def main():
    """Run all tests."""
    print("="*60)
    print("ONNX Operator Tests")
    print("="*60 + "\n")

    test_geometric_ops()
    test_depth_ops()
    test_pose_ops()
    test_ray_tracing_ops()

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)


if __name__ == '__main__':
    main()
