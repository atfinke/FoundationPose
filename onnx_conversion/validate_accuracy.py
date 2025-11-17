#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
Accuracy validation script for ONNX models.

Compares PyTorch and ONNX model outputs to verify numerical equivalence.
"""

import os
import sys
import torch
import onnxruntime as ort
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.models.refine_network import RefineNet
from learning.models.score_network import ScoreNetMultiPair


def load_pytorch_model(checkpoint_path: str, model_type: str = 'refine'):
    """Load PyTorch model from checkpoint."""
    print(f"Loading PyTorch {model_type} model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract config
    if 'cfg' in checkpoint:
        cfg = checkpoint['cfg']
    else:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            'use_BN': True,
            'rot_rep': 'axis_angle',
            'trans_rep': 'tracknet'
        })

    # Create model
    if model_type == 'refine':
        model = RefineNet(cfg=cfg, c_in=4, n_view=1)
    elif model_type == 'score':
        model = ScoreNetMultiPair(cfg=cfg, c_in=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, cfg


def validate_refine_net(pytorch_model, onnx_session, num_samples: int = 100, tolerance: float = 1e-4):
    """Validate RefineNet accuracy."""
    print(f"\nValidating RefineNet with {num_samples} samples...")

    trans_errors = []
    rot_errors = []

    for i in range(num_samples):
        # Random inputs
        A = torch.randn(1, 4, 160, 160)
        B = torch.randn(1, 4, 160, 160)

        # PyTorch inference
        with torch.no_grad():
            pytorch_out = pytorch_model(A, B)
            pytorch_trans = pytorch_out['trans'].numpy()
            pytorch_rot = pytorch_out['rot'].numpy()

        # ONNX inference
        onnx_out = onnx_session.run(None, {
            'rendered_image': A.numpy(),
            'observed_image': B.numpy()
        })
        onnx_trans = onnx_out[0]
        onnx_rot = onnx_out[1]

        # Compute errors
        trans_error = np.abs(pytorch_trans - onnx_trans).max()
        rot_error = np.abs(pytorch_rot - onnx_rot).max()

        trans_errors.append(trans_error)
        rot_errors.append(rot_error)

    # Statistics
    trans_max = max(trans_errors)
    trans_mean = np.mean(trans_errors)
    rot_max = max(rot_errors)
    rot_mean = np.mean(rot_errors)

    print(f"  Translation: max_error={trans_max:.2e}, mean_error={trans_mean:.2e}")
    print(f"  Rotation:    max_error={rot_max:.2e}, mean_error={rot_mean:.2e}")

    # Check tolerance
    passed = (trans_max < tolerance) and (rot_max < tolerance)

    if passed:
        print(f"PASS: All errors < {tolerance}")
    else:
        print(f"FAIL: Errors exceed {tolerance}")
        print(f"  Translation max error: {trans_max}")
        print(f"  Rotation max error: {rot_max}")

    return passed, {
        'trans_max': trans_max,
        'trans_mean': trans_mean,
        'rot_max': rot_max,
        'rot_mean': rot_mean
    }


def validate_score_net(pytorch_model, onnx_session, num_samples: int = 100, num_pairs: int = 8, tolerance: float = 1e-4):
    """Validate ScoreNet accuracy."""
    print(f"\nValidating ScoreNet with {num_samples} samples...")

    score_errors = []

    for i in range(num_samples):
        # Random inputs
        total_batch = num_pairs
        A = torch.randn(total_batch, 4, 160, 160)
        B = torch.randn(total_batch, 4, 160, 160)

        # PyTorch inference
        with torch.no_grad():
            pytorch_out = pytorch_model(A, B, num_pairs)
            pytorch_scores = pytorch_out['score_logit'].numpy()

        # ONNX inference
        onnx_out = onnx_session.run(None, {
            'rendered_images': A.numpy(),
            'observed_images': B.numpy()
        })
        onnx_scores = onnx_out[0]

        # Compute error
        score_error = np.abs(pytorch_scores - onnx_scores).max()
        score_errors.append(score_error)

    # Statistics
    score_max = max(score_errors)
    score_mean = np.mean(score_errors)

    print(f"  Scores: max_error={score_max:.2e}, mean_error={score_mean:.2e}")

    # Check tolerance
    passed = score_max < tolerance

    if passed:
        print(f"PASS: All errors < {tolerance}")
    else:
        print(f"FAIL: Errors exceed {tolerance}")
        print(f"  Score max error: {score_max}")

    return passed, {
        'score_max': score_max,
        'score_mean': score_mean
    }


def main():
    parser = argparse.ArgumentParser(description='Validate ONNX model accuracy')
    parser.add_argument('--pytorch-checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--onnx-model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--model-type', type=str, default='refine',
                        choices=['refine', 'score'],
                        help='Model type')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Error tolerance threshold')
    parser.add_argument('--num-pairs', type=int, default=8,
                        help='Number of pairs for ScoreNet')

    args = parser.parse_args()

    # Verify files exist
    if not os.path.exists(args.pytorch_checkpoint):
        print(f"Error: PyTorch checkpoint not found: {args.pytorch_checkpoint}")
        return 1

    if not os.path.exists(args.onnx_model):
        print(f"Error: ONNX model not found: {args.onnx_model}")
        return 1

    print("="*60)
    print("ONNX Model Accuracy Validation")
    print("="*60)

    # Load models
    pytorch_model, cfg = load_pytorch_model(args.pytorch_checkpoint, args.model_type)

    print(f"Loading ONNX model from {args.onnx_model}...")
    onnx_session = ort.InferenceSession(args.onnx_model)

    # Validate
    if args.model_type == 'refine':
        passed, stats = validate_refine_net(
            pytorch_model,
            onnx_session,
            args.num_samples,
            args.tolerance
        )
    elif args.model_type == 'score':
        passed, stats = validate_score_net(
            pytorch_model,
            onnx_session,
            args.num_samples,
            args.num_pairs,
            args.tolerance
        )

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for key, value in stats.items():
        print(f"{key:15s}: {value:.2e}")
    print("="*60)

    if passed:
        print("Result: VALIDATION PASSED")
        return 0
    else:
        print("Result: VALIDATION FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
