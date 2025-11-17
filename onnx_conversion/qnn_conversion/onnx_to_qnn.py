# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""
Convert ONNX models to QNN format for Qualcomm Hexagon NPU.

This script uses the Qualcomm Neural Network SDK to convert ONNX models
to QNN format with optimizations for Hexagon DSP.
"""

import os
import sys
import argparse
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional


class ONNXtoQNNConverter:
    """
    Converter from ONNX to QNN format.

    Uses QNN SDK tools to convert and optimize models for Hexagon NPU.
    """

    def __init__(
        self,
        qnn_sdk_path: Optional[str] = None,
        backend: str = 'hexagon',
        custom_ops_path: Optional[str] = None
    ):
        """
        Args:
            qnn_sdk_path: path to QNN SDK (default: $QNN_SDK_ROOT)
            backend: QNN backend ('hexagon', 'cpu', 'gpu')
            custom_ops_path: path to custom op library (optional)
        """
        self.qnn_sdk_path = qnn_sdk_path or os.environ.get('QNN_SDK_ROOT')

        if not self.qnn_sdk_path:
            raise ValueError("QNN SDK path not specified. Set QNN_SDK_ROOT or pass qnn_sdk_path")

        self.backend = backend
        self.custom_ops_path = custom_ops_path

        # QNN tools
        self.qnn_converter = os.path.join(self.qnn_sdk_path, 'bin', 'x86_64-linux-clang', 'qnn-onnx-converter')
        self.qnn_model_lib_generator = os.path.join(self.qnn_sdk_path, 'bin', 'x86_64-linux-clang', 'qnn-model-lib-generator')
        self.qnn_context_binary_generator = os.path.join(self.qnn_sdk_path, 'bin', 'x86_64-linux-clang', 'qnn-context-binary-generator')

        # Verify tools exist
        for tool in [self.qnn_converter, self.qnn_model_lib_generator]:
            if not os.path.exists(tool):
                raise FileNotFoundError(f"QNN tool not found: {tool}")

    def convert_onnx_to_qnn(
        self,
        onnx_model_path: str,
        output_dir: str,
        model_name: str = 'model',
        input_list: Optional[List[str]] = None,
        quantization: str = 'int8',
        use_per_channel_quantization: bool = True,
        keep_weights_quantized: bool = True,
        extra_args: Optional[List[str]] = None
    ) -> str:
        """
        Convert ONNX model to QNN format.

        Args:
            onnx_model_path: path to ONNX model
            output_dir: output directory
            model_name: name for QNN model
            input_list: list of input names (format: "name dim1,dim2,...")
            quantization: quantization type ('int8', 'fp16', 'none')
            use_per_channel_quantization: use per-channel quantization
            keep_weights_quantized: keep weights in quantized format
            extra_args: additional arguments for qnn-onnx-converter

        Returns:
            qnn_model_cpp: path to generated QNN model C++ file
        """
        os.makedirs(output_dir, exist_ok=True)

        # Build command
        cmd = [
            self.qnn_converter,
            '--input_network', onnx_model_path,
            '--output_path', os.path.join(output_dir, model_name + '.cpp'),
        ]

        # Input dimensions
        if input_list:
            for input_spec in input_list:
                cmd.extend(['--input_dim', input_spec])

        # Quantization
        if quantization == 'int8':
            cmd.append('--quantization_overrides')
            cmd.append('use_per_channel_quantization={}'.format('true' if use_per_channel_quantization else 'false'))
            if keep_weights_quantized:
                cmd.append('--keep_quant_nodes')

        elif quantization == 'fp16':
            cmd.append('--float_bw')
            cmd.append('16')

        # Custom ops
        if self.custom_ops_path:
            cmd.extend(['--op_package_lib', self.custom_ops_path])

        # Extra arguments
        if extra_args:
            cmd.extend(extra_args)

        # Run converter
        logging.info(f"Running QNN ONNX converter: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info(result.stdout)

            qnn_model_cpp = os.path.join(output_dir, model_name + '.cpp')
            logging.info(f"QNN model generated: {qnn_model_cpp}")

            return qnn_model_cpp

        except subprocess.CalledProcessError as e:
            logging.error(f"QNN conversion failed: {e.stderr}")
            raise

    def generate_model_library(
        self,
        qnn_model_cpp: str,
        output_dir: str,
        model_name: str = 'model'
    ) -> str:
        """
        Generate QNN model library from C++ file.

        Args:
            qnn_model_cpp: path to QNN model C++ file
            output_dir: output directory
            model_name: model name

        Returns:
            model_lib_path: path to generated model library (.so)
        """
        # Build command
        cmd = [
            self.qnn_model_lib_generator,
            '-c', qnn_model_cpp,
            '-o', output_dir,
            '-l', model_name
        ]

        # Run generator
        logging.info(f"Running QNN model library generator: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info(result.stdout)

            # Find generated library
            model_lib_path = None
            for f in os.listdir(output_dir):
                if f.startswith('lib' + model_name) and f.endswith('.so'):
                    model_lib_path = os.path.join(output_dir, f)
                    break

            if not model_lib_path:
                raise FileNotFoundError("Generated model library not found")

            logging.info(f"QNN model library generated: {model_lib_path}")

            return model_lib_path

        except subprocess.CalledProcessError as e:
            logging.error(f"Model library generation failed: {e.stderr}")
            raise

    def generate_context_binary(
        self,
        model_lib_path: str,
        output_path: str,
        backend: Optional[str] = None
    ) -> str:
        """
        Generate QNN context binary for on-device execution.

        Args:
            model_lib_path: path to model library
            output_path: output path for context binary
            backend: backend ('hexagon', 'cpu', 'gpu')

        Returns:
            context_binary_path: path to generated context binary
        """
        backend = backend or self.backend

        # Build command
        cmd = [
            self.qnn_context_binary_generator,
            '--model', model_lib_path,
            '--backend', f'lib{backend}BackendV2.so',
            '--output_dir', os.path.dirname(output_path)
        ]

        # Run generator
        logging.info(f"Running QNN context binary generator: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info(result.stdout)

            # Context binary is generated in output_dir
            context_binary_path = output_path
            logging.info(f"QNN context binary generated: {context_binary_path}")

            return context_binary_path

        except subprocess.CalledProcessError as e:
            logging.error(f"Context binary generation failed: {e.stderr}")
            raise

    def convert_full_pipeline(
        self,
        onnx_model_path: str,
        output_dir: str,
        model_name: str = 'model',
        quantization: str = 'int8',
        generate_binary: bool = True,
        input_dims: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, str]:
        """
        Full conversion pipeline: ONNX -> QNN C++ -> Library -> Context Binary.

        Args:
            onnx_model_path: path to ONNX model
            output_dir: output directory
            model_name: model name
            quantization: quantization type
            generate_binary: whether to generate context binary
            input_dims: dict of input name -> dimensions

        Returns:
            paths: dict with paths to generated files
        """
        logging.info(f"Starting full QNN conversion pipeline for {onnx_model_path}")

        paths = {}

        # Step 1: Convert ONNX to QNN C++
        input_list = None
        if input_dims:
            input_list = [f"{name} {','.join(map(str, dims))}" for name, dims in input_dims.items()]

        qnn_model_cpp = self.convert_onnx_to_qnn(
            onnx_model_path,
            output_dir,
            model_name,
            input_list,
            quantization
        )
        paths['qnn_cpp'] = qnn_model_cpp

        # Step 2: Generate model library
        model_lib_path = self.generate_model_library(
            qnn_model_cpp,
            output_dir,
            model_name
        )
        paths['model_lib'] = model_lib_path

        # Step 3: Generate context binary (optional)
        if generate_binary:
            context_binary_path = os.path.join(output_dir, model_name + '_' + self.backend + '.bin')
            context_binary = self.generate_context_binary(
                model_lib_path,
                context_binary_path
            )
            paths['context_binary'] = context_binary

        logging.info(f"QNN conversion complete! Output directory: {output_dir}")

        return paths


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX models to QNN format')
    parser.add_argument('--onnx-model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for QNN files')
    parser.add_argument('--model-name', type=str, default='model',
                        help='Model name')
    parser.add_argument('--qnn-sdk-path', type=str, default=None,
                        help='Path to QNN SDK (default: $QNN_SDK_ROOT)')
    parser.add_argument('--backend', type=str, default='hexagon',
                        choices=['hexagon', 'cpu', 'gpu'],
                        help='QNN backend')
    parser.add_argument('--quantization', type=str, default='int8',
                        choices=['int8', 'fp16', 'none'],
                        help='Quantization type')
    parser.add_argument('--custom-ops-lib', type=str, default=None,
                        help='Path to custom op library')
    parser.add_argument('--input-dims', type=str, nargs='+', default=None,
                        help='Input dimensions (format: "name:dim1,dim2,...")')
    parser.add_argument('--no-context-binary', action='store_true',
                        help='Skip context binary generation')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Parse input dimensions
    input_dims = None
    if args.input_dims:
        input_dims = {}
        for spec in args.input_dims:
            name, dims_str = spec.split(':')
            dims = [int(d) for d in dims_str.split(',')]
            input_dims[name] = dims

    # Create converter
    converter = ONNXtoQNNConverter(
        qnn_sdk_path=args.qnn_sdk_path,
        backend=args.backend,
        custom_ops_path=args.custom_ops_lib
    )

    # Run conversion
    paths = converter.convert_full_pipeline(
        onnx_model_path=args.onnx_model,
        output_dir=args.output_dir,
        model_name=args.model_name,
        quantization=args.quantization,
        generate_binary=not args.no_context_binary,
        input_dims=input_dims
    )

    # Print summary
    print("\n" + "=" * 60)
    print("QNN Conversion Summary")
    print("=" * 60)
    for key, path in paths.items():
        print(f"{key}: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
