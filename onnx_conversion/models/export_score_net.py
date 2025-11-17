# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
Export ScoreNet model to ONNX format.

This script exports the pose scoring network to ONNX with ONNX-compatible operators.
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import logging
from pathlib import Path

# Add parent directory to path
code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, code_dir)

from learning.models.score_network import ScoreNetMultiPair
from learning.models.network_modules import ConvBNReLU, ResnetBasicBlock, PositionalEmbedding


class ScoreNetONNX(nn.Module):
    """
    ONNX-compatible version of ScoreNetMultiPair.

    Manually implements multi-head attention to avoid fused operations.
    """

    def __init__(self, original_model: ScoreNetMultiPair):
        super().__init__()

        # Copy configuration
        self.cfg = original_model.cfg

        # Copy all network layers from original model
        self.encoderA = original_model.encoderA
        self.encoderAB = original_model.encoderAB
        self.att = original_model.att
        self.att_cross = original_model.att_cross
        self.pos_embed = original_model.pos_embed
        self.linear = original_model.linear

        # Set to eval mode
        self.eval()

    def _manual_multihead_attention(self, q_input, k_input, v_input, mha_layer):
        """
        Manual implementation of multi-head attention to avoid fused operation.
        """
        batch_size, seq_len, embed_dim = q_input.shape
        num_heads = mha_layer.num_heads
        head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        if q_input is k_input and k_input is v_input:
            # Self-attention: use combined QKV projection
            qkv = nn.functional.linear(q_input, mha_layer.in_proj_weight, mha_layer.in_proj_bias)
            qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # Cross-attention: separate projections
            w_q, w_k, w_v = mha_layer.in_proj_weight.split(embed_dim)
            b_q, b_k, b_v = mha_layer.in_proj_bias.split(embed_dim)
            q = nn.functional.linear(q_input, w_q, b_q).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = nn.functional.linear(k_input, w_k, b_k).reshape(batch_size, k_input.shape[1], num_heads, head_dim).transpose(1, 2)
            v = nn.functional.linear(v_input, w_v, b_v).reshape(batch_size, v_input.shape[1], num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, embed_dim)

        # Output projection
        out = mha_layer.out_proj(out)

        return out

    def forward(self, A: torch.Tensor, B: torch.Tensor, L: int) -> dict:
        """
        Forward pass with ONNX-compatible operations.

        Args:
            A: (B*L, C, H, W) rendered images for L pose pairs
            B: (B*L, C, H, W) observed images for L pose pairs
            L: number of pose pairs

        Returns:
            output: dict with keys:
                - 'score_logit': (B, L) scores for each pose pair
        """
        output = {}
        bs = A.shape[0] // L

        # Extract features
        x = torch.cat([A, B], dim=0)
        x = self.encoderA(x)
        a = x[:bs*L]
        b = x[bs*L:]

        ab = torch.cat((a, b), dim=1)
        ab = self.encoderAB(ab)

        ab = self.pos_embed(ab.reshape(bs*L, ab.shape[1], -1).permute(0, 2, 1))

        # Manual self-attention instead of fused operation
        ab = self._manual_multihead_attention(ab, ab, ab, self.att)

        feats = ab.mean(dim=1).reshape(bs*L, -1)

        # Cross-attention between pairs
        x = feats.reshape(bs, L, -1)
        x = self._manual_multihead_attention(x, x, x, self.att_cross)

        # Score prediction
        output['score_logit'] = self.linear(x).reshape(bs, L)

        return output


def export_score_net(
    checkpoint_path: str,
    output_path: str,
    input_size: tuple = (160, 160),
    num_pairs: int = 8,
    batch_size: int = 1,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    dynamic_pairs: bool = False
):
    """
    Export ScoreNet to ONNX format.

    Args:
        checkpoint_path: path to PyTorch checkpoint
        output_path: path to save ONNX model
        input_size: (H, W) input image size
        num_pairs: number of pose pairs
        batch_size: batch size for export
        opset_version: ONNX opset version
        dynamic_batch: whether to use dynamic batch dimension
        dynamic_pairs: whether to use dynamic num_pairs dimension
    """
    logging.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load configuration
    if 'cfg' in checkpoint:
        cfg = checkpoint['cfg']
    else:
        # Try to load config.yml from checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, 'config.yml')
        if os.path.exists(config_path):
            logging.info(f"Loading config from {config_path}")
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(config_path)
        else:
            # Default configuration
            from omegaconf import OmegaConf
            cfg = OmegaConf.create({
                'use_BN': True,
                'c_in': 4,
                'n_view': 1
            })

    logging.info(f"Model configuration: {cfg}")

    # Get c_in from config or use default
    c_in = cfg.get('c_in', 4) if isinstance(cfg, dict) else getattr(cfg, 'c_in', 4)
    n_view = cfg.get('n_view', 1) if isinstance(cfg, dict) else getattr(cfg, 'n_view', 1)

    logging.info(f"Using c_in={c_in}, n_view={n_view}")

    # Create original model
    original_model = ScoreNetMultiPair(cfg=cfg, c_in=c_in)

    # Load weights
    original_model.load_state_dict(state_dict, strict=False)
    original_model.eval()

    # Create ONNX-compatible model
    model = ScoreNetONNX(original_model)
    model.eval()

    # Create dummy inputs
    H, W = input_size
    total_batch = batch_size * num_pairs
    dummy_A = torch.randn(total_batch, c_in, H, W)
    dummy_B = torch.randn(total_batch, c_in, H, W)
    dummy_L = torch.tensor(num_pairs, dtype=torch.long)

    # NOTE: ONNX export doesn't support integer scalar inputs well
    # We'll use a constant L value in the model

    class ScoreNetONNXConstL(nn.Module):
        def __init__(self, model, L):
            super().__init__()
            self.model = model
            self.L = L

        def forward(self, A, B):
            return self.model(A, B, self.L)

    model_const_L = ScoreNetONNXConstL(model, num_pairs)

    # Define input names
    input_names = ['rendered_images', 'observed_images']

    # Define output names
    output_names = ['score_logits']

    # Define dynamic axes
    dynamic_axes = {}
    if dynamic_batch and dynamic_pairs:
        # Both batch and pairs are dynamic
        dynamic_axes = {
            'rendered_images': {0: 'batch_times_pairs'},
            'observed_images': {0: 'batch_times_pairs'},
            'score_logits': {0: 'batch', 1: 'pairs'}
        }
    elif dynamic_batch:
        # Only batch is dynamic (pairs is fixed)
        dynamic_axes = {
            'rendered_images': {0: 'batch_times_pairs'},
            'observed_images': {0: 'batch_times_pairs'},
            'score_logits': {0: 'batch'}
        }
    else:
        dynamic_axes = None

    # Export to ONNX
    logging.info(f"Exporting to ONNX (opset {opset_version})...")

    with torch.no_grad():
        torch.onnx.export(
            model_const_L,
            (dummy_A, dummy_B),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            dynamo=False  # Use legacy exporter for compatibility
        )

    logging.info(f"ONNX model saved to {output_path}")

    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logging.info("ONNX model verification passed!")

        # Print model info
        logging.info(f"Model inputs: {[i.name for i in onnx_model.graph.input]}")
        logging.info(f"Model outputs: {[o.name for o in onnx_model.graph.output]}")
        logging.info(f"Note: Number of pairs (L) is fixed at {num_pairs}")

    except ImportError:
        logging.warning("onnx package not found, skipping verification")
    except Exception as e:
        logging.error(f"ONNX verification failed: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export ScoreNet to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save ONNX model')
    parser.add_argument('--input-size', type=int, nargs=2, default=[160, 160],
                        help='Input image size (H W)')
    parser.add_argument('--num-pairs', type=int, default=8,
                        help='Number of pose pairs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for export')
    parser.add_argument('--opset-version', type=int, default=17,
                        help='ONNX opset version')
    parser.add_argument('--no-dynamic-batch', action='store_true',
                        help='Disable dynamic batch dimension')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Export model
    export_score_net(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_size=tuple(args.input_size),
        num_pairs=args.num_pairs,
        batch_size=args.batch_size,
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch
    )


if __name__ == '__main__':
    main()
