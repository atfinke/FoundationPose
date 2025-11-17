# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
ONNX-compatible operator implementations for FoundationPose.

This module provides PyTorch implementations of custom CUDA operators
that are compatible with ONNX export.
"""

from .rendering import OnnxRenderer, nvdiffrast_render_onnx
from .depth_ops import depth_erosion_onnx, bilateral_filter_onnx
from .geometric_ops import (
    rotation_6d_to_matrix_onnx,
    so3_exp_map_onnx,
    transform_pts_onnx,
    barycentric_coords_3d_onnx,
    barycentric_coords_2d_onnx
)
from .pose_ops import cluster_poses_onnx, geodesic_distance_onnx
from .ray_tracing import (
    sample_rays_uniform_occupied_voxels_onnx,
    postprocess_octree_ray_tracing_onnx,
    ray_color_to_texture_image_onnx
)
from .grid_encoder import GridEncoderONNX, HashGridEncoder

__all__ = [
    # Rendering
    'OnnxRenderer',
    'nvdiffrast_render_onnx',

    # Depth operations
    'depth_erosion_onnx',
    'bilateral_filter_onnx',

    # Geometric operations
    'rotation_6d_to_matrix_onnx',
    'so3_exp_map_onnx',
    'transform_pts_onnx',
    'barycentric_coords_3d_onnx',
    'barycentric_coords_2d_onnx',

    # Pose operations
    'cluster_poses_onnx',
    'geodesic_distance_onnx',

    # Ray tracing
    'sample_rays_uniform_occupied_voxels_onnx',
    'postprocess_octree_ray_tracing_onnx',
    'ray_color_to_texture_image_onnx',

    # Grid encoder
    'GridEncoderONNX',
    'HashGridEncoder',
]
