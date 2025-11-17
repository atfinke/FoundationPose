# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
ONNX-compatible ray tracing operations.

Replaces CUDA ray tracing kernels with pure PyTorch scatter/gather operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_rays_uniform_occupied_voxels_onnx(
    z_sampled: torch.Tensor,
    z_in_out: torch.Tensor
) -> torch.Tensor:
    """
    Sample ray depths uniformly within occupied voxels.
    ONNX-compatible version of sample_rays_uniform_occupied_voxels_kernel from common.cu.

    Args:
        z_sampled: (N_rays, N_samples) sampled depth values (cumulative lengths)
        z_in_out: (N_rays, max_n_box, 2) depth in/out pairs for each ray

    Returns:
        z_vals: (N_rays, N_samples) actual depth values
    """
    N_rays, N_samples = z_sampled.shape
    max_n_box = z_in_out.shape[1]
    device = z_sampled.device
    dtype = z_sampled.dtype

    # Initialize output
    z_vals = torch.zeros_like(z_sampled)

    # Process each ray
    for i_ray in range(N_rays):
        z_remain = z_sampled[i_ray]  # (N_samples,)
        ray_boxes = z_in_out[i_ray]  # (max_n_box, 2)

        # Find valid boxes (non-zero entries)
        valid_mask = ray_boxes[:, 0] > 0  # (max_n_box,)

        if not valid_mask.any():
            continue

        # Get box lengths
        box_lengths = ray_boxes[:, 1] - ray_boxes[:, 0]  # (max_n_box,)
        box_lengths = torch.where(valid_mask, box_lengths, torch.zeros_like(box_lengths))

        # Cumulative sum of box lengths
        cumsum_lengths = torch.cumsum(box_lengths, dim=0)  # (max_n_box,)

        # For each sample, find which box it belongs to
        for i_sample in range(N_samples):
            z_rem = z_remain[i_sample]

            # Find the box index
            box_idx = torch.searchsorted(cumsum_lengths, z_rem, right=False)
            box_idx = torch.clamp(box_idx, 0, max_n_box - 1)

            # Compute offset within the box
            if box_idx > 0:
                offset = z_rem - cumsum_lengths[box_idx - 1]
            else:
                offset = z_rem

            # Compute actual depth
            z_vals[i_ray, i_sample] = ray_boxes[box_idx, 0] + offset

    return z_vals


def postprocess_octree_ray_tracing_onnx(
    ray_index: torch.Tensor,
    depth_in_out: torch.Tensor,
    max_intersections: int,
    N_rays: int
) -> torch.Tensor:
    """
    Postprocess octree ray tracing results.
    ONNX-compatible version of postprocessOctreeRayTracing from common.cu.

    Args:
        ray_index: (N_intersections,) ray indices for each intersection
        depth_in_out: (N_intersections, 2) depth in/out pairs
        max_intersections: maximum intersections per ray
        N_rays: total number of rays

    Returns:
        depths_in_out_padded: (N_rays, max_intersections, 2) padded depth pairs
    """
    device = ray_index.device
    dtype = depth_in_out.dtype

    # Initialize output
    depths_in_out_padded = torch.zeros(
        (N_rays, max_intersections, 2),
        device=device,
        dtype=dtype
    )

    # Get unique ray indices and their positions
    unique_rays, inverse_indices = torch.unique(ray_index, return_inverse=True)

    # Process each unique ray
    for i, ray_id in enumerate(unique_rays):
        # Find all intersections for this ray
        mask = (ray_index == ray_id)
        ray_depths = depth_in_out[mask]  # (n_intersect, 2)

        # Filter valid intersections
        valid_mask = (ray_depths[:, 0] > 0) & (ray_depths[:, 1] > 0)
        valid_mask = valid_mask & (ray_depths[:, 0] <= ray_depths[:, 1])
        valid_mask = valid_mask & ((ray_depths[:, 1] - ray_depths[:, 0]) >= 1e-4)

        valid_depths = ray_depths[valid_mask]

        # Clamp to max_intersections
        n_valid = min(valid_depths.shape[0], max_intersections)
        if n_valid > 0:
            depths_in_out_padded[ray_id, :n_valid] = valid_depths[:n_valid]

    return depths_in_out_padded


def ray_color_to_texture_image_onnx(
    F_faces: torch.Tensor,
    V_vertices: torch.Tensor,
    hit_locations: torch.Tensor,
    hit_face_ids: torch.Tensor,
    uvs_texture: torch.Tensor
) -> torch.Tensor:
    """
    Map ray hit locations to texture UV coordinates.
    ONNX-compatible version of rayColorToTextureImageCUDA from common.cu.

    Args:
        F_faces: (N_faces, 3) face indices
        V_vertices: (N_vertices, 3) vertex positions
        hit_locations: (N_hits, 3) ray hit 3D locations
        hit_face_ids: (N_hits,) face IDs for each hit
        uvs_texture: (N_vertices, 2) texture UV coordinates per vertex

    Returns:
        uvs_output: (N_hits, 2) UV coordinates for each hit
    """
    N_hits = hit_locations.shape[0]
    device = hit_locations.device
    dtype = hit_locations.dtype

    # Initialize output
    uvs_output = torch.zeros((N_hits, 2), device=device, dtype=dtype)

    # Process each hit
    for i_hit in range(N_hits):
        face_id = hit_face_ids[i_hit]
        hit_loc = hit_locations[i_hit]  # (3,)

        # Get face vertices
        face = F_faces[face_id]  # (3,)
        v0 = V_vertices[face[0]]  # (3,)
        v1 = V_vertices[face[1]]  # (3,)
        v2 = V_vertices[face[2]]  # (3,)

        # Stack into triangle
        triangle = torch.stack([v0, v1, v2], dim=0)  # (3, 3)

        # Compute barycentric coordinates (using function from geometric_ops)
        from .geometric_ops import barycentric_coords_3d_onnx
        weights = barycentric_coords_3d_onnx(triangle, hit_loc)  # (3,)

        # Get texture UVs for face vertices
        uv0 = uvs_texture[face[0]]  # (2,)
        uv1 = uvs_texture[face[1]]  # (2,)
        uv2 = uvs_texture[face[2]]  # (2,)

        # Interpolate UV coordinates
        uv = weights[0] * uv0 + weights[1] * uv1 + weights[2] * uv2  # (2,)

        uvs_output[i_hit] = uv

    return uvs_output


def ray_sphere_intersection_onnx(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    sphere_center: torch.Tensor,
    sphere_radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ray-sphere intersections.
    ONNX-compatible implementation.

    Args:
        ray_origins: (N_rays, 3) ray origins
        ray_directions: (N_rays, 3) ray directions (normalized)
        sphere_center: (3,) sphere center
        sphere_radius: sphere radius

    Returns:
        t_near: (N_rays,) near intersection distance (-1 if no intersection)
        t_far: (N_rays,) far intersection distance (-1 if no intersection)
    """
    N_rays = ray_origins.shape[0]
    device = ray_origins.device
    dtype = ray_origins.dtype

    # Vector from ray origin to sphere center
    oc = ray_origins - sphere_center.unsqueeze(0)  # (N_rays, 3)

    # Quadratic coefficients
    a = (ray_directions ** 2).sum(dim=-1)  # (N_rays,)
    b = 2.0 * (oc * ray_directions).sum(dim=-1)  # (N_rays,)
    c = (oc ** 2).sum(dim=-1) - sphere_radius ** 2  # (N_rays,)

    # Discriminant
    discriminant = b ** 2 - 4 * a * c  # (N_rays,)

    # Check for intersections
    has_intersection = discriminant >= 0

    # Compute intersection distances
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0))
    t_near = (-b - sqrt_discriminant) / (2.0 * a)
    t_far = (-b + sqrt_discriminant) / (2.0 * a)

    # Set invalid intersections to -1
    t_near = torch.where(has_intersection, t_near, torch.full_like(t_near, -1.0))
    t_far = torch.where(has_intersection, t_far, torch.full_like(t_far, -1.0))

    return t_near, t_far


def ray_aabb_intersection_onnx(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ray-AABB (axis-aligned bounding box) intersections.
    ONNX-compatible implementation.

    Args:
        ray_origins: (N_rays, 3) ray origins
        ray_directions: (N_rays, 3) ray directions
        aabb_min: (3,) or (N_rays, 3) AABB minimum corner
        aabb_max: (3,) or (N_rays, 3) AABB maximum corner

    Returns:
        t_near: (N_rays,) near intersection distance (-1 if no intersection)
        t_far: (N_rays,) far intersection distance (-1 if no intersection)
    """
    N_rays = ray_origins.shape[0]
    device = ray_origins.device

    if aabb_min.ndim == 1:
        aabb_min = aabb_min.unsqueeze(0).expand(N_rays, -1)
    if aabb_max.ndim == 1:
        aabb_max = aabb_max.unsqueeze(0).expand(N_rays, -1)

    # Compute intersection distances for each axis
    inv_dir = 1.0 / (ray_directions + 1e-10)  # (N_rays, 3)

    t0 = (aabb_min - ray_origins) * inv_dir  # (N_rays, 3)
    t1 = (aabb_max - ray_origins) * inv_dir  # (N_rays, 3)

    # Swap if needed
    t_min = torch.minimum(t0, t1)  # (N_rays, 3)
    t_max = torch.maximum(t0, t1)  # (N_rays, 3)

    # Find overall near and far
    t_near = torch.max(t_min, dim=-1)[0]  # (N_rays,)
    t_far = torch.min(t_max, dim=-1)[0]  # (N_rays,)

    # Check for valid intersections
    has_intersection = (t_near <= t_far) & (t_far >= 0)

    # Set invalid intersections to -1
    t_near = torch.where(has_intersection, t_near, torch.full_like(t_near, -1.0))
    t_far = torch.where(has_intersection, t_far, torch.full_like(t_far, -1.0))

    return t_near, t_far


class RayTracingONNX(nn.Module):
    """
    Module containing all ray tracing operations for ONNX export.
    """

    def __init__(self, max_intersections: int = 128):
        super().__init__()
        self.max_intersections = max_intersections

    def sample_rays(
        self,
        z_sampled: torch.Tensor,
        z_in_out: torch.Tensor
    ) -> torch.Tensor:
        return sample_rays_uniform_occupied_voxels_onnx(z_sampled, z_in_out)

    def postprocess_octree(
        self,
        ray_index: torch.Tensor,
        depth_in_out: torch.Tensor,
        N_rays: int
    ) -> torch.Tensor:
        return postprocess_octree_ray_tracing_onnx(
            ray_index,
            depth_in_out,
            self.max_intersections,
            N_rays
        )

    def ray_to_texture(
        self,
        F_faces: torch.Tensor,
        V_vertices: torch.Tensor,
        hit_locations: torch.Tensor,
        hit_face_ids: torch.Tensor,
        uvs_texture: torch.Tensor
    ) -> torch.Tensor:
        return ray_color_to_texture_image_onnx(
            F_faces,
            V_vertices,
            hit_locations,
            hit_face_ids,
            uvs_texture
        )

    def sphere_intersection(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        sphere_center: torch.Tensor,
        sphere_radius: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ray_sphere_intersection_onnx(
            ray_origins,
            ray_directions,
            sphere_center,
            sphere_radius
        )

    def aabb_intersection(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ray_aabb_intersection_onnx(
            ray_origins,
            ray_directions,
            aabb_min,
            aabb_max
        )

    def forward(self, z_sampled: torch.Tensor, z_in_out: torch.Tensor) -> torch.Tensor:
        """Default forward: sample rays."""
        return self.sample_rays(z_sampled, z_in_out)
