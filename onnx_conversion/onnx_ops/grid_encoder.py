# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
ONNX-compatible grid encoder implementation.

Replaces Instant-NGP CUDA grid encoder with pure PyTorch multi-resolution hash encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HashGridEncoder(nn.Module):
    """
    Multi-resolution hash grid encoding for neural fields.
    ONNX-compatible PyTorch implementation of Instant-NGP hash encoding.

    Based on: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
    Müller et al., SIGGRAPH 2022
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_levels: int = 16,
        level_dim: int = 2,
        per_level_scale: float = 2.0,
        base_resolution: int = 16,
        log2_hashmap_size: int = 19,
        finest_resolution: int = 512
    ):
        """
        Args:
            input_dim: input coordinate dimension (typically 3 for 3D)
            num_levels: number of resolution levels
            level_dim: feature dimension per level
            per_level_scale: scale factor between levels
            base_resolution: resolution of coarsest level
            log2_hashmap_size: log2 of hash table size
            finest_resolution: resolution of finest level (optional)
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size

        # Compute resolution for each level
        self.resolutions = []
        for i in range(num_levels):
            resolution = int(base_resolution * (per_level_scale ** i))
            if finest_resolution is not None:
                resolution = min(resolution, finest_resolution)
            self.resolutions.append(resolution)

        # Hash table for each level
        self.hash_tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, level_dim)
            for _ in range(num_levels)
        ])

        # Initialize embeddings
        for table in self.hash_tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)

        # Prime numbers for hashing (for 3D)
        self.register_buffer('primes', torch.tensor([1, 2654435761, 805459861], dtype=torch.long))

    def hash_3d(self, coords: torch.Tensor) -> torch.Tensor:
        """
        3D spatial hash function.

        Args:
            coords: (B, 3) integer coordinates

        Returns:
            hash_values: (B,) hash values in range [0, hashmap_size)
        """
        # Hash: (x * p1 XOR y * p2 XOR z * p3) mod hashmap_size
        hash_val = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)

        for i in range(3):
            hash_val = hash_val ^ (coords[:, i].long() * self.primes[i])

        hash_val = hash_val % self.hashmap_size

        return hash_val

    def hash_2d(self, coords: torch.Tensor) -> torch.Tensor:
        """
        2D spatial hash function.

        Args:
            coords: (B, 2) integer coordinates

        Returns:
            hash_values: (B,) hash values
        """
        hash_val = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)

        for i in range(2):
            hash_val = hash_val ^ (coords[:, i].long() * self.primes[i])

        hash_val = hash_val % self.hashmap_size

        return hash_val

    def interpolate_level(
        self,
        coords: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """
        Interpolate features at a single resolution level.

        Args:
            coords: (B, input_dim) coordinates in [0, 1]
            level: resolution level index

        Returns:
            features: (B, level_dim) interpolated features
        """
        resolution = self.resolutions[level]
        hash_table = self.hash_tables[level]

        # Scale coordinates to grid resolution
        scaled_coords = coords * resolution  # (B, input_dim)

        # Get grid cell corners
        coords_floor = torch.floor(scaled_coords)  # (B, input_dim)
        coords_frac = scaled_coords - coords_floor  # (B, input_dim)

        if self.input_dim == 3:
            # 3D trilinear interpolation
            # Get 8 corners of cube
            corners = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        offset = torch.tensor([i, j, k], dtype=coords.dtype, device=coords.device)
                        corner = coords_floor + offset  # (B, 3)
                        corners.append(corner)

            # Hash each corner
            corner_features = []
            for corner in corners:
                hash_idx = self.hash_3d(corner)  # (B,)
                features = hash_table(hash_idx)  # (B, level_dim)
                corner_features.append(features)

            # Trilinear interpolation weights
            x, y, z = coords_frac[:, 0], coords_frac[:, 1], coords_frac[:, 2]
            weights = [
                (1-x) * (1-y) * (1-z),  # 000
                (1-x) * (1-y) * z,      # 001
                (1-x) * y * (1-z),      # 010
                (1-x) * y * z,          # 011
                x * (1-y) * (1-z),      # 100
                x * (1-y) * z,          # 101
                x * y * (1-z),          # 110
                x * y * z               # 111
            ]

            # Weighted sum
            result = torch.zeros((coords.shape[0], self.level_dim), dtype=coords.dtype, device=coords.device)
            for w, f in zip(weights, corner_features):
                result = result + w.unsqueeze(-1) * f

        elif self.input_dim == 2:
            # 2D bilinear interpolation
            corners = []
            for i in range(2):
                for j in range(2):
                    offset = torch.tensor([i, j], dtype=coords.dtype, device=coords.device)
                    corner = coords_floor + offset  # (B, 2)
                    corners.append(corner)

            # Hash each corner
            corner_features = []
            for corner in corners:
                hash_idx = self.hash_2d(corner)  # (B,)
                features = hash_table(hash_idx)  # (B, level_dim)
                corner_features.append(features)

            # Bilinear interpolation weights
            x, y = coords_frac[:, 0], coords_frac[:, 1]
            weights = [
                (1-x) * (1-y),  # 00
                (1-x) * y,      # 01
                x * (1-y),      # 10
                x * y           # 11
            ]

            # Weighted sum
            result = torch.zeros((coords.shape[0], self.level_dim), dtype=coords.dtype, device=coords.device)
            for w, f in zip(weights, corner_features):
                result = result + w.unsqueeze(-1) * f

        return result

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates with multi-resolution hash grid.

        Args:
            coords: (B, input_dim) coordinates in [0, 1]

        Returns:
            encodings: (B, num_levels * level_dim) encoded features
        """
        # Clamp coordinates to [0, 1]
        coords = torch.clamp(coords, 0.0, 1.0)

        # Encode at each level
        level_features = []
        for level in range(self.num_levels):
            features = self.interpolate_level(coords, level)  # (B, level_dim)
            level_features.append(features)

        # Concatenate all levels
        encodings = torch.cat(level_features, dim=-1)  # (B, num_levels * level_dim)

        return encodings

    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.num_levels * self.level_dim


class GridEncoderONNX(nn.Module):
    """
    Complete grid encoder with positional encoding fallback.
    Compatible with ONNX export.
    """

    def __init__(
        self,
        input_dim: int = 3,
        encoding_type: str = 'hash',
        num_levels: int = 16,
        level_dim: int = 2,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        log2_hashmap_size: int = 19,
        positional_encoding_multires: int = 10
    ):
        """
        Args:
            input_dim: input dimension
            encoding_type: 'hash' or 'positional'
            num_levels: number of levels for hash encoding
            level_dim: features per level for hash encoding
            base_resolution: base resolution for hash encoding
            finest_resolution: finest resolution for hash encoding
            log2_hashmap_size: hash table size (log2)
            positional_encoding_multires: frequencies for positional encoding
        """
        super().__init__()

        self.input_dim = input_dim
        self.encoding_type = encoding_type

        if encoding_type == 'hash':
            self.encoder = HashGridEncoder(
                input_dim=input_dim,
                num_levels=num_levels,
                level_dim=level_dim,
                base_resolution=base_resolution,
                finest_resolution=finest_resolution,
                log2_hashmap_size=log2_hashmap_size
            )
            self.output_dim = self.encoder.get_output_dim()

        elif encoding_type == 'positional':
            self.multires = positional_encoding_multires
            self.output_dim = input_dim * (2 * positional_encoding_multires + 1)

        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

    def positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional encoding.

        Args:
            coords: (B, input_dim) coordinates

        Returns:
            encodings: (B, input_dim * (2 * multires + 1)) encoded features
        """
        encodings = [coords]

        for i in range(self.multires):
            freq = 2.0 ** i
            encodings.append(torch.sin(freq * math.pi * coords))
            encodings.append(torch.cos(freq * math.pi * coords))

        return torch.cat(encodings, dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode input coordinates.

        Args:
            coords: (B, input_dim) coordinates

        Returns:
            encodings: (B, output_dim) encoded features
        """
        if self.encoding_type == 'hash':
            return self.encoder(coords)
        elif self.encoding_type == 'positional':
            return self.positional_encoding(coords)

    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim


# Backward compatibility
class InstantNGPEncoder(HashGridEncoder):
    """Alias for HashGridEncoder."""
    pass
