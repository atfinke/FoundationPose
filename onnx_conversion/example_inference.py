#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
Example inference with ONNX models.

Demonstrates how to use exported ONNX models for pose refinement and scoring.
"""

import onnxruntime as ort
import numpy as np


def example_refine_net():
    """Example RefineNet inference."""
    print("="*60)
    print("RefineNet ONNX Inference Example")
    print("="*60)

    # Load ONNX model
    model_path = '../models_onnx/refine_net.onnx'
    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path)

    # Print model info
    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    print("\nModel outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Create dummy inputs (batch_size=2, channels=4, height=160, width=160)
    batch_size = 2
    rendered_image = np.random.randn(batch_size, 4, 160, 160).astype(np.float32)
    observed_image = np.random.randn(batch_size, 4, 160, 160).astype(np.float32)

    print(f"\nRunning inference with batch_size={batch_size}...")

    # Run inference
    outputs = session.run(None, {
        'rendered_image': rendered_image,
        'observed_image': observed_image
    })

    # Parse outputs
    translation_delta = outputs[0]  # (batch_size, 3)
    rotation_delta = outputs[1]      # (batch_size, 3) or (batch_size, 6)

    print(f"\nOutputs:")
    print(f"  Translation delta: shape={translation_delta.shape}")
    print(f"    Sample values: {translation_delta[0]}")
    print(f"  Rotation delta: shape={rotation_delta.shape}")
    print(f"    Sample values: {rotation_delta[0]}")

    print("\n✅ RefineNet inference completed successfully")


def example_score_net():
    """Example ScoreNet inference."""
    print("\n" + "="*60)
    print("ScoreNet ONNX Inference Example")
    print("="*60)

    # Load ONNX model
    model_path = '../models_onnx/score_net.onnx'
    print(f"Loading model: {model_path}")

    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Note: ScoreNet model not found or not exported yet")
        print(f"  Run: python models/export_score_net.py ...")
        return

    # Print model info
    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    print("\nModel outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Create dummy inputs
    # For ScoreNet, we need batch_size * num_pairs images
    batch_size = 2
    num_pairs = 8
    total_images = batch_size * num_pairs

    rendered_images = np.random.randn(total_images, 4, 160, 160).astype(np.float32)
    observed_images = np.random.randn(total_images, 4, 160, 160).astype(np.float32)

    print(f"\nRunning inference with batch_size={batch_size}, num_pairs={num_pairs}...")

    # Run inference
    outputs = session.run(None, {
        'rendered_images': rendered_images,
        'observed_images': observed_images
    })

    # Parse outputs
    score_logits = outputs[0]  # (batch_size, num_pairs)

    print(f"\nOutputs:")
    print(f"  Score logits: shape={score_logits.shape}")
    print(f"    Sample values: {score_logits[0]}")
    print(f"    Best pose index: {np.argmax(score_logits[0])}")

    print("\n✅ ScoreNet inference completed successfully")


def example_with_onnx_ops():
    """Example using ONNX operators directly."""
    print("\n" + "="*60)
    print("Direct ONNX Operator Usage Example")
    print("="*60)

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import torch
    from onnx_ops.geometric_ops import (
        rotation_6d_to_matrix_onnx,
        so3_exp_map_onnx,
        transform_pts_onnx
    )
    from onnx_ops.depth_ops import depth_erosion_onnx, bilateral_filter_onnx

    print("\n1. Rotation operations:")

    # 6D rotation to matrix
    rotation_6d = torch.randn(2, 6)
    rotation_matrix = rotation_6d_to_matrix_onnx(rotation_6d)
    print(f"  rotation_6d_to_matrix: (2,6) → (2,3,3)")
    print(f"    Input: {rotation_6d[0, :3]}")
    print(f"    Output shape: {rotation_matrix.shape}")

    # Axis-angle to matrix
    axis_angle = torch.randn(2, 3) * 0.1  # Small rotations
    rotation_matrix = so3_exp_map_onnx(axis_angle)
    print(f"  so3_exp_map (Rodrigues): (2,3) → (2,3,3)")
    print(f"    Input: {axis_angle[0]}")
    print(f"    Output shape: {rotation_matrix.shape}")

    # Transform points
    points = torch.randn(100, 3)
    transform = torch.eye(4).unsqueeze(0).expand(2, 4, 4)
    transform[:, :3, 3] = torch.randn(2, 3)  # Random translation
    transformed = transform_pts_onnx(points, transform)
    print(f"  transform_pts: (100,3) with (2,4,4) → (2,100,3)")
    print(f"    Output shape: {transformed.shape}")

    print("\n2. Depth operations:")

    # Depth erosion
    depth = torch.rand(480, 640) * 5.0  # Random depth map
    eroded = depth_erosion_onnx(depth, radius=2)
    print(f"  depth_erosion: (480,640) → (480,640)")
    print(f"    Input range: [{depth.min():.2f}, {depth.max():.2f}]")
    print(f"    Output range: [{eroded.min():.2f}, {eroded.max():.2f}]")

    # Bilateral filtering
    filtered = bilateral_filter_onnx(depth, radius=2)
    print(f"  bilateral_filter: (480,640) → (480,640)")
    print(f"    Output range: [{filtered.min():.2f}, {filtered.max():.2f}]")

    print("\n✅ ONNX operators working correctly")


def main():
    """Run all examples."""
    print("\nFoundationPose ONNX Inference Examples\n")

    # Example 1: RefineNet
    try:
        example_refine_net()
    except FileNotFoundError:
        print("\nNote: RefineNet ONNX model not found.")
        print("  Run: python models/export_refine_net.py ...")

    # Example 2: ScoreNet
    try:
        example_score_net()
    except FileNotFoundError:
        pass  # Already handled in function

    # Example 3: Direct operator usage
    example_with_onnx_ops()

    print("\n" + "="*60)
    print("All examples completed")
    print("="*60)


if __name__ == '__main__':
    main()
