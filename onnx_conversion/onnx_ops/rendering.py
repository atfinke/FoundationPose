# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
ONNX-compatible differentiable rendering operations.

Replaces nvdiffrast with PyTorch3D-based rendering that is ONNX-exportable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def projection_matrix_from_intrinsics_onnx(
    K: torch.Tensor,
    height: int,
    width: int,
    znear: float = 0.001,
    zfar: float = 100.0
) -> torch.Tensor:
    """
    Create OpenGL projection matrix from camera intrinsics.
    ONNX-compatible version.

    Args:
        K: (3, 3) or (B, 3, 3) camera intrinsics
        height: image height
        width: image width
        znear: near clipping plane
        zfar: far clipping plane

    Returns:
        P: (4, 4) or (B, 4, 4) projection matrix
    """
    if K.ndim == 2:
        K = K.unsqueeze(0)  # (1, 3, 3)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = K.shape[0]
    device = K.device
    dtype = K.dtype

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    # OpenGL projection matrix
    P = torch.zeros((batch_size, 4, 4), device=device, dtype=dtype)

    P[:, 0, 0] = 2.0 * fx / width
    P[:, 1, 1] = 2.0 * fy / height
    P[:, 0, 2] = 1.0 - 2.0 * cx / width
    P[:, 1, 2] = 1.0 - 2.0 * cy / height
    P[:, 2, 2] = -(zfar + znear) / (zfar - znear)
    P[:, 2, 3] = -2.0 * zfar * znear / (zfar - znear)
    P[:, 3, 2] = -1.0

    if squeeze_output:
        P = P.squeeze(0)

    return P


def transform_vertices_onnx(
    vertices: torch.Tensor,
    mvp_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Transform vertices to clip space.
    ONNX-compatible version.

    Args:
        vertices: (N, 3) vertex positions
        mvp_matrix: (B, 4, 4) model-view-projection matrices

    Returns:
        vertices_clip: (B, N, 4) vertices in clip space
    """
    if mvp_matrix.ndim == 2:
        mvp_matrix = mvp_matrix.unsqueeze(0)

    batch_size = mvp_matrix.shape[0]
    N = vertices.shape[0]

    # Homogeneous coordinates
    ones = torch.ones((N, 1), device=vertices.device, dtype=vertices.dtype)
    vertices_homo = torch.cat([vertices, ones], dim=-1)  # (N, 4)

    # Transform
    vertices_clip = (mvp_matrix.unsqueeze(1) @ vertices_homo.unsqueeze(0).unsqueeze(-1)).squeeze(-1)  # (B, N, 4)

    return vertices_clip


def rasterize_triangles_onnx(
    vertices_clip: torch.Tensor,
    faces: torch.Tensor,
    height: int,
    width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Software rasterization of triangles (simplified, ONNX-compatible).

    NOTE: This is a simplified implementation. For production use,
    consider using PyTorch3D's rasterizer or a more optimized implementation.

    Args:
        vertices_clip: (B, N, 4) vertices in clip space
        faces: (F, 3) face indices
        height: output height
        width: output width

    Returns:
        face_idx_map: (B, H, W) face index for each pixel (-1 if background)
        bary_coords: (B, H, W, 3) barycentric coordinates
        depth_map: (B, H, W) depth values
    """
    batch_size = vertices_clip.shape[0]
    N = vertices_clip.shape[1]
    F_count = faces.shape[0]
    device = vertices_clip.device
    dtype = vertices_clip.dtype

    # Perspective division
    vertices_ndc = vertices_clip[..., :3] / (vertices_clip[..., 3:4] + 1e-10)  # (B, N, 3)

    # NDC to screen space
    vertices_screen = torch.zeros_like(vertices_ndc)
    vertices_screen[..., 0] = (vertices_ndc[..., 0] + 1.0) * 0.5 * width
    vertices_screen[..., 1] = (1.0 - vertices_ndc[..., 1]) * 0.5 * height  # Flip Y
    vertices_screen[..., 2] = vertices_ndc[..., 2]

    # Initialize output
    face_idx_map = -torch.ones((batch_size, height, width), dtype=torch.long, device=device)
    bary_coords = torch.zeros((batch_size, height, width, 3), dtype=dtype, device=device)
    depth_map = torch.full((batch_size, height, width), float('inf'), dtype=dtype, device=device)

    # Pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing='ij'
    )
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)

    # Rasterize each triangle
    for batch_idx in range(batch_size):
        for face_idx in range(F_count):
            # Get triangle vertices
            v_indices = faces[face_idx]  # (3,)
            v0_screen = vertices_screen[batch_idx, v_indices[0]]  # (3,)
            v1_screen = vertices_screen[batch_idx, v_indices[1]]  # (3,)
            v2_screen = vertices_screen[batch_idx, v_indices[2]]  # (3,)

            # Bounding box
            min_x = max(0, int(torch.min(torch.stack([v0_screen[0], v1_screen[0], v2_screen[0]])).item()))
            max_x = min(width, int(torch.max(torch.stack([v0_screen[0], v1_screen[0], v2_screen[0]])).item()) + 1)
            min_y = max(0, int(torch.min(torch.stack([v0_screen[1], v1_screen[1], v2_screen[1]])).item()))
            max_y = min(height, int(torch.max(torch.stack([v0_screen[1], v1_screen[1], v2_screen[1]])).item()) + 1)

            if min_x >= max_x or min_y >= max_y:
                continue

            # Get pixels in bounding box
            pixels = pixel_coords[min_y:max_y, min_x:max_x]  # (H_box, W_box, 2)

            # Compute barycentric coordinates
            from .geometric_ops import barycentric_coords_2d_onnx

            # Triangle in 2D screen space
            triangle_2d = torch.stack([
                v0_screen[:2],
                v1_screen[:2],
                v2_screen[:2]
            ], dim=0)  # (3, 2)

            # Flatten pixels
            pixels_flat = pixels.reshape(-1, 2)  # (N_pixels, 2)

            # Compute barycentric coordinates
            bary = barycentric_coords_2d_onnx(triangle_2d, pixels_flat)  # (N_pixels, 3)

            # Check if inside triangle
            inside = (bary >= 0).all(dim=-1)  # (N_pixels,)

            if not inside.any():
                continue

            # Compute depth using barycentric interpolation
            depths_vertices = torch.stack([v0_screen[2], v1_screen[2], v2_screen[2]], dim=0)  # (3,)
            depths_pixels = (bary * depths_vertices.unsqueeze(0)).sum(dim=-1)  # (N_pixels,)

            # Update depth buffer
            inside_indices = torch.where(inside)[0]
            for i in inside_indices:
                pixel_linear_idx = i.item()
                y_offset = pixel_linear_idx // (max_x - min_x)
                x_offset = pixel_linear_idx % (max_x - min_x)
                y = min_y + y_offset
                x = min_x + x_offset

                pixel_depth = depths_pixels[i]

                if pixel_depth < depth_map[batch_idx, y, x]:
                    depth_map[batch_idx, y, x] = pixel_depth
                    face_idx_map[batch_idx, y, x] = face_idx
                    bary_coords[batch_idx, y, x] = bary[i]

    # Set invalid depths to 0
    depth_map = torch.where(depth_map < float('inf'), depth_map, torch.zeros_like(depth_map))

    return face_idx_map, bary_coords, depth_map


def interpolate_vertex_attributes_onnx(
    attributes: torch.Tensor,
    faces: torch.Tensor,
    face_idx_map: torch.Tensor,
    bary_coords: torch.Tensor
) -> torch.Tensor:
    """
    Interpolate vertex attributes using barycentric coordinates.
    ONNX-compatible version.

    Args:
        attributes: (N, C) vertex attributes
        faces: (F, 3) face indices
        face_idx_map: (B, H, W) face index for each pixel
        bary_coords: (B, H, W, 3) barycentric coordinates

    Returns:
        interpolated: (B, H, W, C) interpolated attributes
    """
    batch_size, H, W = face_idx_map.shape
    N, C = attributes.shape
    device = attributes.device
    dtype = attributes.dtype

    # Initialize output
    interpolated = torch.zeros((batch_size, H, W, C), dtype=dtype, device=device)

    # Process each batch
    for b in range(batch_size):
        for h in range(H):
            for w in range(W):
                face_idx = face_idx_map[b, h, w].item()

                if face_idx < 0:
                    continue  # Background

                # Get vertex indices
                v_indices = faces[face_idx]  # (3,)

                # Get vertex attributes
                v0_attr = attributes[v_indices[0]]  # (C,)
                v1_attr = attributes[v_indices[1]]  # (C,)
                v2_attr = attributes[v_indices[2]]  # (C,)

                # Barycentric interpolation
                bary = bary_coords[b, h, w]  # (3,)
                attr = bary[0] * v0_attr + bary[1] * v1_attr + bary[2] * v2_attr  # (C,)

                interpolated[b, h, w] = attr

    return interpolated


class OnnxRenderer(nn.Module):
    """
    ONNX-compatible differentiable renderer.

    Simplified software rasterizer for ONNX export.
    For production, consider PyTorch3D with ONNX export support.
    """

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        znear: float = 0.001,
        zfar: float = 100.0
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.znear = znear
        self.zfar = zfar

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_colors: torch.Tensor,
        K: torch.Tensor,
        poses: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render mesh from given camera poses.

        Args:
            vertices: (N, 3) mesh vertices
            faces: (F, 3) mesh faces
            vertex_colors: (N, 3) vertex colors (RGB in [0, 1])
            K: (3, 3) or (B, 3, 3) camera intrinsics
            poses: (B, 4, 4) camera poses (world-to-camera)

        Returns:
            colors: (B, H, W, 3) rendered colors
            depths: (B, H, W) rendered depths
        """
        # Compute MVP matrix
        P = projection_matrix_from_intrinsics_onnx(K, self.height, self.width, self.znear, self.zfar)

        # Transform to camera space
        from .geometric_ops import transform_pts_onnx

        if poses.ndim == 2:
            poses = poses.unsqueeze(0)

        batch_size = poses.shape[0]

        # Compute model-view-projection matrix
        if P.ndim == 2:
            P = P.unsqueeze(0).expand(batch_size, -1, -1)

        MVP = P @ poses  # (B, 4, 4)

        # Transform vertices to clip space
        vertices_clip = transform_vertices_onnx(vertices, MVP)  # (B, N, 4)

        # Rasterize
        face_idx_map, bary_coords, depth_map = rasterize_triangles_onnx(
            vertices_clip,
            faces,
            self.height,
            self.width
        )

        # Interpolate colors
        colors = interpolate_vertex_attributes_onnx(
            vertex_colors,
            faces,
            face_idx_map,
            bary_coords
        )  # (B, H, W, 3)

        return colors, depth_map


def nvdiffrast_render_onnx(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_colors: torch.Tensor,
    K: torch.Tensor,
    poses: torch.Tensor,
    height: int,
    width: int,
    vertex_normals: torch.Tensor = None,
    use_lighting: bool = False,
    light_dir: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ONNX-compatible replacement for nvdiffrast_render().

    Args:
        vertices: (N, 3) mesh vertices
        faces: (F, 3) mesh faces
        vertex_colors: (N, 3) vertex colors
        K: (3, 3) or (B, 3, 3) camera intrinsics
        poses: (B, 4, 4) camera poses
        height: output height
        width: output width
        vertex_normals: (N, 3) vertex normals (optional)
        use_lighting: whether to apply lighting
        light_dir: (3,) light direction (optional)

    Returns:
        colors: (B, H, W, 3) rendered colors
        depths: (B, H, W) rendered depths
        normals: (B, H, W, 3) rendered normals (if vertex_normals provided)
    """
    renderer = OnnxRenderer(height, width)
    colors, depths = renderer(vertices, faces, vertex_colors, K, poses)

    normals = None
    if vertex_normals is not None:
        # Also render normals
        P = projection_matrix_from_intrinsics_onnx(K, height, width)

        if poses.ndim == 2:
            poses = poses.unsqueeze(0)

        batch_size = poses.shape[0]

        if P.ndim == 2:
            P = P.unsqueeze(0).expand(batch_size, -1, -1)

        MVP = P @ poses

        vertices_clip = transform_vertices_onnx(vertices, MVP)
        face_idx_map, bary_coords, _ = rasterize_triangles_onnx(
            vertices_clip,
            faces,
            height,
            width
        )

        # Transform normals to camera space
        from .geometric_ops import transform_pts_onnx
        normals_cam = []
        for b in range(batch_size):
            # Apply rotation only (no translation) to normals
            R = poses[b, :3, :3]
            n_cam = vertex_normals @ R.T
            normals_cam.append(n_cam)

        normals_cam = torch.stack(normals_cam, dim=0)  # (B, N, 3)

        # Interpolate normals for each batch
        normals = torch.zeros((batch_size, height, width, 3), device=vertices.device, dtype=vertices.dtype)
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    face_idx = face_idx_map[b, h, w].item()
                    if face_idx >= 0:
                        v_indices = faces[face_idx]
                        bary = bary_coords[b, h, w]
                        normal = (
                            bary[0] * normals_cam[b, v_indices[0]] +
                            bary[1] * normals_cam[b, v_indices[1]] +
                            bary[2] * normals_cam[b, v_indices[2]]
                        )
                        normals[b, h, w] = F.normalize(normal.unsqueeze(0), dim=-1).squeeze(0)

    if use_lighting and light_dir is not None and normals is not None:
        # Apply simple directional lighting
        light_dir = F.normalize(light_dir.unsqueeze(0), dim=-1)
        diffuse = torch.clamp((normals * light_dir).sum(dim=-1, keepdim=True), 0, 1)
        colors = colors * (0.3 + 0.7 * diffuse)  # Ambient + diffuse

    return colors, depths, normals
