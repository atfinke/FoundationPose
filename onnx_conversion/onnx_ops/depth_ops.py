# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
ONNX-compatible depth processing operations.

Replaces Warp/CUDA depth operations with pure PyTorch convolution-based implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def depth_erosion_onnx(
    depth: torch.Tensor,
    radius: int = 2,
    depth_diff_thres: float = 0.001,
    ratio_thres: float = 0.8,
    zfar: float = 100.0
) -> torch.Tensor:
    """
    Morphological erosion for depth maps using pure PyTorch operations.
    ONNX-compatible version of erode_depth() from Utils.py.

    Erodes pixels if too many neighbors are invalid or have significantly different depth.

    Args:
        depth: (H, W) or (B, H, W) depth map
        radius: erosion kernel radius
        depth_diff_thres: threshold for depth difference
        ratio_thres: ratio of bad neighbors to trigger erosion
        zfar: far clipping plane

    Returns:
        eroded_depth: eroded depth map (same shape as input)
    """
    input_2d = depth.ndim == 2
    if input_2d:
        depth = depth.unsqueeze(0)  # (1, H, W)

    batch_size, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    # Add channel dimension for convolution
    depth = depth.unsqueeze(1)  # (B, 1, H, W)

    # Create valid mask (depth in valid range)
    valid_mask = (depth >= 0.001) & (depth < zfar)  # (B, 1, H, W)

    # Create neighborhood kernel
    kernel_size = 2 * radius + 1
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype)
    center_weight = kernel_size * kernel_size

    # Count total neighbors (excluding center)
    total_neighbors = F.conv2d(
        torch.ones_like(depth),
        kernel,
        padding=radius
    ) - 1.0  # Subtract center pixel

    # Count invalid neighbors
    invalid_neighbors = F.conv2d(
        (~valid_mask).float(),
        kernel,
        padding=radius
    ) - 1.0 + (~valid_mask).float()  # Exclude center from count, then add back if center invalid

    # Compute depth differences from each pixel
    # Replicate depth values across neighborhood
    depth_padded = F.pad(depth, (radius, radius, radius, radius), mode='replicate')

    # Create difference mask by comparing with neighbors
    bad_depth_count = torch.zeros_like(depth)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue  # Skip center

            # Extract shifted depth
            y_start = radius + dy
            x_start = radius + dx
            neighbor_depth = depth_padded[:, :, y_start:y_start+H, x_start:x_start+W]

            # Check if neighbor is invalid or differs significantly
            neighbor_invalid = (neighbor_depth < 0.001) | (neighbor_depth >= zfar)
            depth_diff_large = torch.abs(neighbor_depth - depth) > depth_diff_thres

            bad_neighbor = neighbor_invalid | depth_diff_large
            bad_depth_count = bad_depth_count + bad_neighbor.float()

    # Compute ratio of bad neighbors
    bad_ratio = bad_depth_count / (total_neighbors + 1e-10)

    # Erode if bad ratio exceeds threshold or if center is invalid
    should_erode = (bad_ratio > ratio_thres) | (~valid_mask)

    # Apply erosion
    eroded_depth = torch.where(should_erode, torch.zeros_like(depth), depth)

    # Remove channel dimension
    eroded_depth = eroded_depth.squeeze(1)  # (B, H, W)

    if input_2d:
        eroded_depth = eroded_depth.squeeze(0)  # (H, W)

    return eroded_depth


def bilateral_filter_onnx(
    depth: torch.Tensor,
    radius: int = 2,
    zfar: float = 100.0,
    sigma_spatial: float = 2.0,
    sigma_range: float = 100000.0
) -> torch.Tensor:
    """
    Bilateral filtering for depth maps using pure PyTorch operations.
    ONNX-compatible version of bilateral_filter_depth() from Utils.py.

    Smooths depth while preserving edges.

    Args:
        depth: (H, W) or (B, H, W) depth map
        radius: filter radius
        zfar: far clipping plane
        sigma_spatial: spatial Gaussian sigma
        sigma_range: range (depth) Gaussian sigma

    Returns:
        filtered_depth: filtered depth map (same shape as input)
    """
    input_2d = depth.ndim == 2
    if input_2d:
        depth = depth.unsqueeze(0)  # (1, H, W)

    batch_size, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    # Add channel dimension
    depth = depth.unsqueeze(1)  # (B, 1, H, W)

    # Create valid mask
    valid_mask = (depth >= 0.001) & (depth < zfar)

    # Compute mean depth in neighborhood for outlier rejection
    kernel_size = 2 * radius + 1
    uniform_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype)
    uniform_kernel = uniform_kernel / (kernel_size * kernel_size)

    valid_depth = torch.where(valid_mask, depth, torch.zeros_like(depth))
    depth_sum = F.conv2d(valid_depth, uniform_kernel, padding=radius)
    valid_count = F.conv2d(valid_mask.float(), uniform_kernel, padding=radius)
    mean_depth = depth_sum / (valid_count + 1e-10)

    # Create spatial Gaussian kernel
    y_grid, x_grid = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        indexing='ij'
    )
    spatial_dist = x_grid**2 + y_grid**2
    spatial_kernel = torch.exp(-spatial_dist / (2.0 * sigma_spatial**2))
    spatial_kernel = spatial_kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    # Pad depth for neighborhood access
    depth_padded = F.pad(depth, (radius, radius, radius, radius), mode='replicate')
    valid_padded = F.pad(valid_mask.float(), (radius, radius, radius, radius), mode='replicate')
    mean_padded = F.pad(mean_depth, (radius, radius, radius, radius), mode='replicate')

    # Initialize output
    filtered_depth = torch.zeros_like(depth)
    total_weight = torch.zeros_like(depth)

    # Bilateral filtering loop
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Extract shifted depth
            y_start = radius + dy
            x_start = radius + dx
            neighbor_depth = depth_padded[:, :, y_start:y_start+H, x_start:x_start+W]
            neighbor_valid = valid_padded[:, :, y_start:y_start+H, x_start:x_start+W]
            neighbor_mean = mean_padded[:, :, y_start:y_start+H, x_start:x_start+W]

            # Check if neighbor is valid and close to mean (outlier rejection)
            is_valid = (neighbor_valid > 0.5) & (torch.abs(neighbor_depth - neighbor_mean) < 0.01)

            # Compute range (depth) distance
            range_dist = (depth - neighbor_depth)**2
            range_weight = torch.exp(-range_dist / (2.0 * sigma_range**2))

            # Get spatial weight for this offset
            ky = dy + radius
            kx = dx + radius
            spatial_weight = spatial_kernel[0, 0, ky, kx]

            # Combined weight
            weight = spatial_weight * range_weight * is_valid.float()

            # Accumulate
            filtered_depth = filtered_depth + weight * neighbor_depth
            total_weight = total_weight + weight

    # Normalize
    filtered_depth = filtered_depth / (total_weight + 1e-10)

    # Keep only valid pixels
    filtered_depth = torch.where(valid_mask & (total_weight > 0), filtered_depth, torch.zeros_like(depth))

    # Remove channel dimension
    filtered_depth = filtered_depth.squeeze(1)  # (B, H, W)

    if input_2d:
        filtered_depth = filtered_depth.squeeze(0)  # (H, W)

    return filtered_depth


def depth_to_xyzmap_onnx(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Convert depth map to XYZ map (3D point cloud in camera coordinates).
    ONNX-compatible version.

    Args:
        depth: (B, H, W) or (H, W) depth maps
        K: (3, 3) or (B, 3, 3) camera intrinsics

    Returns:
        xyz_map: (B, H, W, 3) or (H, W, 3) 3D coordinates
    """
    input_2d = depth.ndim == 2
    if input_2d:
        depth = depth.unsqueeze(0)  # (1, H, W)

    if K.ndim == 2:
        K = K.unsqueeze(0)  # (1, 3, 3)

    batch_size, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    # Create pixel grid
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )  # (H, W)

    # Expand for batch
    u_grid = u_grid.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H, W)
    v_grid = v_grid.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H, W)

    # Extract intrinsics
    fx = K[:, 0:1, 0:1]  # (B, 1, 1)
    fy = K[:, 1:2, 1:2]  # (B, 1, 1)
    cx = K[:, 0:1, 2:3]  # (B, 1, 1)
    cy = K[:, 1:2, 2:3]  # (B, 1, 1)

    # Unproject to 3D
    z = depth  # (B, H, W)
    x = (u_grid - cx) * z / fx  # (B, H, W)
    y = (v_grid - cy) * z / fy  # (B, H, W)

    # Stack into XYZ map
    xyz_map = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)

    # Mask invalid depths
    invalid_mask = (depth < 0.001).unsqueeze(-1)  # (B, H, W, 1)
    xyz_map = torch.where(invalid_mask, torch.zeros_like(xyz_map), xyz_map)

    if input_2d:
        xyz_map = xyz_map.squeeze(0)  # (H, W, 3)

    return xyz_map


class DepthOpsONNX(nn.Module):
    """
    Module containing all depth operations for ONNX export.
    """

    def __init__(
        self,
        erosion_radius: int = 2,
        erosion_depth_diff: float = 0.001,
        erosion_ratio_thres: float = 0.8,
        bilateral_radius: int = 2,
        bilateral_sigma_spatial: float = 2.0,
        bilateral_sigma_range: float = 100000.0,
        zfar: float = 100.0
    ):
        super().__init__()
        self.erosion_radius = erosion_radius
        self.erosion_depth_diff = erosion_depth_diff
        self.erosion_ratio_thres = erosion_ratio_thres
        self.bilateral_radius = bilateral_radius
        self.bilateral_sigma_spatial = bilateral_sigma_spatial
        self.bilateral_sigma_range = bilateral_sigma_range
        self.zfar = zfar

    def erode(self, depth: torch.Tensor) -> torch.Tensor:
        return depth_erosion_onnx(
            depth,
            radius=self.erosion_radius,
            depth_diff_thres=self.erosion_depth_diff,
            ratio_thres=self.erosion_ratio_thres,
            zfar=self.zfar
        )

    def bilateral_filter(self, depth: torch.Tensor) -> torch.Tensor:
        return bilateral_filter_onnx(
            depth,
            radius=self.bilateral_radius,
            zfar=self.zfar,
            sigma_spatial=self.bilateral_sigma_spatial,
            sigma_range=self.bilateral_sigma_range
        )

    def to_xyzmap(self, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        return depth_to_xyzmap_onnx(depth, K)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply erosion followed by bilateral filtering."""
        depth = self.erode(depth)
        depth = self.bilateral_filter(depth)
        return depth
