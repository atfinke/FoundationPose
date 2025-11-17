# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
Export RefineNet model to ONNX format.

This script exports the pose refinement network to ONNX with ONNX-compatible operators.
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

from learning.models.refine_network import RefineNet
from learning.models.network_modules import ConvBNReLU, ResnetBasicBlock, PositionalEmbedding
from onnx_conversion.onnx_ops.geometric_ops import rotation_6d_to_matrix_onnx


class RefineNetONNX(nn.Module):
    """
    ONNX-compatible version of RefineNet.

    Replaces pytorch3d operations with pure PyTorch equivalents.
    Also manually implements TransformerEncoderLayer to avoid fused operations.
    """

    def __init__(self, original_model: RefineNet):
        super().__init__()

        # Copy configuration
        self.cfg = original_model.cfg

        # Copy all network layers from original model
        self.encodeA = original_model.encodeA
        self.encodeAB = original_model.encodeAB
        self.pos_embed = original_model.pos_embed

        # Extract transformer components for manual forward
        self.trans_transformer = original_model.trans_head[0]
        self.trans_linear = original_model.trans_head[1]

        self.rot_transformer = original_model.rot_head[0]
        self.rot_linear = original_model.rot_head[1]

        # Set to eval mode
        self.eval()

    def _manual_multihead_attention(self, x, mha_layer):
        """
        Manual implementation of multi-head attention to avoid fused operation.
        """
        batch_size, seq_len, embed_dim = x.shape
        num_heads = mha_layer.num_heads
        head_dim = embed_dim // num_heads

        # Linear projections
        qkv = nn.functional.linear(x, mha_layer.in_proj_weight, mha_layer.in_proj_bias)
        qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, seq_len, seq_len)
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        out = attn @ v  # (B, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous()  # (B, seq_len, num_heads, head_dim)
        out = out.reshape(batch_size, seq_len, embed_dim)

        # Output projection
        out = mha_layer.out_proj(out)

        return out

    def _transformer_forward(self, x, transformer_layer):
        """
        Manual implementation of TransformerEncoderLayer forward to avoid fused operation.
        """
        # Self-attention (manual implementation)
        attn_output = self._manual_multihead_attention(x, transformer_layer.self_attn)
        x = transformer_layer.norm1(x + transformer_layer.dropout1(attn_output))

        # Feedforward
        ff_output = transformer_layer.linear2(transformer_layer.dropout(transformer_layer.activation(transformer_layer.linear1(x))))
        x = transformer_layer.norm2(x + transformer_layer.dropout2(ff_output))

        return x

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> dict:
        """
        Forward pass with ONNX-compatible operations.

        Args:
            A: (B, C, H, W) rendered image
            B: (B, C, H, W) observed image

        Returns:
            output: dict with keys:
                - 'trans': (B, 3) translation delta
                - 'rot': (B, 3) or (B, 6) rotation delta (axis-angle or 6D)
                - 'rot_matrix': (B, 3, 3) rotation matrix (if rot_rep is 6d)
        """
        bs = len(A)
        output = {}

        # Encoder A
        x = torch.cat([A, B], dim=0)
        x = self.encodeA(x)
        a = x[:bs]
        b = x[bs:]

        # Encoder AB
        ab = torch.cat((a, b), 1).contiguous()
        ab = self.encodeAB(ab)  # (B, C, H, W)

        # Positional embedding
        ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0, 2, 1))

        # Translation head - manual transformer forward
        trans_features = self._transformer_forward(ab, self.trans_transformer)
        output['trans'] = self.trans_linear(trans_features).mean(dim=1)

        # Rotation head - manual transformer forward
        rot_features = self._transformer_forward(ab, self.rot_transformer)
        rot_output = self.rot_linear(rot_features).mean(dim=1)
        output['rot'] = rot_output

        # Convert 6D rotation to matrix if needed (for downstream use)
        if self.cfg['rot_rep'] == '6d':
            output['rot_matrix'] = rotation_6d_to_matrix_onnx(rot_output)

        return output


def export_refine_net(
    checkpoint_path: str,
    output_path: str,
    input_size: tuple = (160, 160),
    batch_size: int = 1,
    opset_version: int = 17,
    dynamic_batch: bool = True
):
    """
    Export RefineNet to ONNX format.

    Args:
        checkpoint_path: path to PyTorch checkpoint
        output_path: path to save ONNX model
        input_size: (H, W) input image size
        batch_size: batch size for export
        opset_version: ONNX opset version
        dynamic_batch: whether to use dynamic batch dimension
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
                'rot_rep': 'axis_angle',  # or '6d'
                'trans_rep': 'tracknet',
                'c_in': 4,
                'n_view': 1
            })

    logging.info(f"Model configuration: {cfg}")

    # Get c_in from config or use default
    c_in = cfg.get('c_in', 4) if isinstance(cfg, dict) else getattr(cfg, 'c_in', 4)
    n_view = cfg.get('n_view', 1) if isinstance(cfg, dict) else getattr(cfg, 'n_view', 1)

    logging.info(f"Using c_in={c_in}, n_view={n_view}")

    # Create original model
    original_model = RefineNet(cfg=cfg, c_in=c_in, n_view=n_view)

    # Load weights
    original_model.load_state_dict(state_dict, strict=False)
    original_model.eval()

    # Create ONNX-compatible model
    model = RefineNetONNX(original_model)
    model.eval()

    # Create dummy inputs
    H, W = input_size
    dummy_A = torch.randn(batch_size, c_in, H, W)
    dummy_B = torch.randn(batch_size, c_in, H, W)

    # Define input names
    input_names = ['rendered_image', 'observed_image']

    # Define output names
    if cfg['rot_rep'] == '6d':
        output_names = ['translation_delta', 'rotation_delta_6d', 'rotation_matrix']
    else:
        output_names = ['translation_delta', 'rotation_delta_axis_angle']

    # Define dynamic axes
    if dynamic_batch:
        dynamic_axes = {
            'rendered_image': {0: 'batch'},
            'observed_image': {0: 'batch'},
            'translation_delta': {0: 'batch'},
        }
        if cfg['rot_rep'] == '6d':
            dynamic_axes['rotation_delta_6d'] = {0: 'batch'}
            dynamic_axes['rotation_matrix'] = {0: 'batch'}
        else:
            dynamic_axes['rotation_delta_axis_angle'] = {0: 'batch'}
    else:
        dynamic_axes = None

    # Export to ONNX
    logging.info(f"Exporting to ONNX (opset {opset_version})...")

    with torch.no_grad():
        torch.onnx.export(
            model,
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

    except ImportError:
        logging.warning("onnx package not found, skipping verification")
    except Exception as e:
        logging.error(f"ONNX verification failed: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export RefineNet to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save ONNX model')
    parser.add_argument('--input-size', type=int, nargs=2, default=[160, 160],
                        help='Input image size (H W)')
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
    export_refine_net(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_size=tuple(args.input_size),
        batch_size=args.batch_size,
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch
    )


if __name__ == '__main__':
    main()
