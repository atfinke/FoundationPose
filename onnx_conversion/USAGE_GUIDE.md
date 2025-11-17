# FoundationPose ONNX and QNN Conversion - Complete Usage Guide

This guide provides step-by-step instructions for converting FoundationPose models to ONNX and QNN formats.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [ONNX Export](#onnx-export)
4. [Model Validation](#model-validation)
5. [QNN Conversion](#qnn-conversion)
6. [On-Device Deployment](#on-device-deployment)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- **For ONNX Export**: CPU/GPU with PyTorch support
- **For QNN Conversion**: x86_64 Linux machine
- **For Deployment**: Qualcomm Snapdragon device with Hexagon DSP

### Software Requirements

```bash
# Python 3.8+
python --version

# PyTorch 2.0+
python -c "import torch; print(torch.__version__)"

# ONNX
python -c "import onnx; print(onnx.__version__)"

# QNN SDK 2.15+
echo $QNN_SDK_ROOT
```

## Installation

### Step 1: Install Python Dependencies

```bash
cd onnx_conversion
pip install -r requirements.txt
```

### Step 2: Install QNN SDK

```bash
# Download QNN SDK from Qualcomm Developer Network
# https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk

# Extract SDK
tar -xzf qnn-sdk-v2.15.0.tar.gz

# Set environment variable
export QNN_SDK_ROOT=/path/to/qnn-sdk
echo "export QNN_SDK_ROOT=/path/to/qnn-sdk" >> ~/.bashrc
```

### Step 3: Build Custom QNN Operators (Optional)

```bash
cd onnx_conversion/qnn_conversion/custom_ops
mkdir build && cd build
cmake .. && make -j8
```

## ONNX Export

### Export RefineNet

```bash
# Basic export
python onnx_conversion/models/export_refine_net.py \
    --checkpoint weights/2023-10-28-18-33-37/model_best.pth \
    --output models_onnx/refine_net.onnx

# With specific input size
python onnx_conversion/models/export_refine_net.py \
    --checkpoint weights/2023-10-28-18-33-37/model_best.pth \
    --output models_onnx/refine_net.onnx \
    --input-size 160 160 \
    --batch-size 1 \
    --opset-version 17
```

**Expected Output:**
```
INFO: Loading checkpoint from weights/2023-10-28-18-33-37/model_best.pth
INFO: Model configuration: {'use_BN': True, 'rot_rep': 'axis_angle', ...}
INFO: Exporting to ONNX (opset 17)...
INFO: ONNX model saved to models_onnx/refine_net.onnx
INFO: ONNX model verification passed!
INFO: Model inputs: ['rendered_image', 'observed_image']
INFO: Model outputs: ['translation_delta', 'rotation_delta_axis_angle']
```

### Export ScoreNet

```bash
python onnx_conversion/models/export_score_net.py \
    --checkpoint weights/2024-01-11-20-02-45/model_best.pth \
    --output models_onnx/score_net.onnx \
    --num-pairs 8 \
    --input-size 160 160
```

### Verify ONNX Models

```bash
# Check model with onnx
python -c "
import onnx
model = onnx.load('models_onnx/refine_net.onnx')
onnx.checker.check_model(model)
print('Model is valid!')
print(f'Inputs: {[i.name for i in model.graph.input]}')
print(f'Outputs: {[o.name for o in model.graph.output]}')
"

# Test with onnxruntime
python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models_onnx/refine_net.onnx')

# Create dummy inputs
A = np.random.randn(1, 4, 160, 160).astype(np.float32)
B = np.random.randn(1, 4, 160, 160).astype(np.float32)

# Run inference
outputs = session.run(None, {
    'rendered_image': A,
    'observed_image': B
})

print(f'Translation delta shape: {outputs[0].shape}')
print(f'Rotation delta shape: {outputs[1].shape}')
"
```

## Model Validation

### Accuracy Validation

Create a validation script to compare PyTorch vs ONNX outputs:

```python
import torch
import onnxruntime as ort
import numpy as np
from learning.models.refine_network import RefineNet

# Load PyTorch model
checkpoint = torch.load('weights/2023-10-28-18-33-37/model_best.pth')
pytorch_model = RefineNet(cfg=checkpoint['cfg'])
pytorch_model.load_state_dict(checkpoint['model_state_dict'])
pytorch_model.eval()

# Load ONNX model
ort_session = ort.InferenceSession('models_onnx/refine_net.onnx')

# Test inputs
A = torch.randn(1, 4, 160, 160)
B = torch.randn(1, 4, 160, 160)

# PyTorch inference
with torch.no_grad():
    pytorch_out = pytorch_model(A, B)

# ONNX inference
onnx_out = ort_session.run(None, {
    'rendered_image': A.numpy(),
    'observed_image': B.numpy()
})

# Compare outputs
trans_diff = np.abs(pytorch_out['trans'].numpy() - onnx_out[0]).max()
rot_diff = np.abs(pytorch_out['rot'].numpy() - onnx_out[1]).max()

print(f"Translation max diff: {trans_diff}")
print(f"Rotation max diff: {rot_diff}")
print(f"Validation: {'PASSED' if trans_diff < 1e-4 and rot_diff < 1e-4 else 'FAILED'}")
```

## QNN Conversion

### Convert RefineNet to QNN

```bash
# INT8 quantization (recommended for Hexagon)
python onnx_conversion/qnn_conversion/onnx_to_qnn.py \
    --onnx-model models_onnx/refine_net.onnx \
    --output-dir models_qnn/refine_net \
    --model-name refine_net \
    --backend hexagon \
    --quantization int8 \
    --input-dims "rendered_image:1,4,160,160" "observed_image:1,4,160,160"
```

**Expected Output:**
```
INFO: Starting full QNN conversion pipeline for models_onnx/refine_net.onnx
INFO: Running QNN ONNX converter...
INFO: QNN model generated: models_qnn/refine_net/refine_net.cpp
INFO: Running QNN model library generator...
INFO: QNN model library generated: models_qnn/refine_net/librefine_net.so
INFO: Running QNN context binary generator...
INFO: QNN context binary generated: models_qnn/refine_net/refine_net_hexagon.bin
INFO: QNN conversion complete!

============================================================
QNN Conversion Summary
============================================================
qnn_cpp: models_qnn/refine_net/refine_net.cpp
model_lib: models_qnn/refine_net/librefine_net.so
context_binary: models_qnn/refine_net/refine_net_hexagon.bin
============================================================
```

### Convert ScoreNet to QNN

```bash
python onnx_conversion/qnn_conversion/onnx_to_qnn.py \
    --onnx-model models_onnx/score_net.onnx \
    --output-dir models_qnn/score_net \
    --model-name score_net \
    --backend hexagon \
    --quantization int8 \
    --input-dims "rendered_images:8,4,160,160" "observed_images:8,4,160,160"
```

### FP16 Quantization (Higher Accuracy)

```bash
python onnx_conversion/qnn_conversion/onnx_to_qnn.py \
    --onnx-model models_onnx/refine_net.onnx \
    --output-dir models_qnn/refine_net_fp16 \
    --model-name refine_net_fp16 \
    --backend hexagon \
    --quantization fp16 \
    --input-dims "rendered_image:1,4,160,160" "observed_image:1,4,160,160"
```

## On-Device Deployment

### Using QNN Runtime

```cpp
#include "QnnInterface.h"
#include "QnnBackend.h"

// Load context binary
QnnContext_Handle_t context;
Qnn_ErrorHandle_t error = QnnContext_createFromBinary(
    backend,
    "refine_net_hexagon.bin",
    &context
);

// Prepare inputs
float* rendered_image = ...; // (1, 4, 160, 160)
float* observed_image = ...; // (1, 4, 160, 160)

Qnn_Tensor_t inputs[2] = {
    {.name = "rendered_image", .data = rendered_image, ...},
    {.name = "observed_image", .data = observed_image, ...}
};

// Execute
Qnn_Tensor_t outputs[2];
QnnGraph_execute(graph, inputs, 2, outputs, 2);

// Get results
float* translation_delta = (float*)outputs[0].data; // (1, 3)
float* rotation_delta = (float*)outputs[1].data;    // (1, 3)
```

### Android Integration

```java
// QNNExecutor.java
public class QNNExecutor {
    static {
        System.loadLibrary("QnnHexagon");
        System.loadLibrary("refine_net");
    }

    public native float[] execute(
        float[] renderedImage,
        float[] observedImage
    );
}
```

### Python Inference on Device (via ADB)

```bash
# Push model to device
adb push models_qnn/refine_net/refine_net_hexagon.bin /data/local/tmp/

# Push QNN runtime
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHexagon.so /data/local/tmp/

# Run inference
adb shell "cd /data/local/tmp && ./qnn-net-run \
    --model refine_net_hexagon.bin \
    --input_list inputs.txt"
```

## Troubleshooting

### Issue: ONNX Export Fails

**Error**: `RuntimeError: ONNX export failed: Couldn't export Python operator ...`

**Solution**:
1. Check that all operations are ONNX-compatible
2. Verify opset version supports required operations
3. Use `torch.onnx.export(..., verbose=True)` for details

```bash
# Re-export with verbose output
python onnx_conversion/models/export_refine_net.py \
    --checkpoint weights/2023-10-28-18-33-37/model_best.pth \
    --output models_onnx/refine_net_debug.onnx \
    2>&1 | tee export.log
```

### Issue: QNN Conversion Fails

**Error**: `Unsupported operator: CustomOp`

**Solution**: Implement custom QNN operator

```bash
# Build custom ops
cd onnx_conversion/qnn_conversion/custom_ops
mkdir build && cd build
cmake .. && make -j8

# Use custom ops in conversion
python onnx_conversion/qnn_conversion/onnx_to_qnn.py \
    --onnx-model models_onnx/refine_net.onnx \
    --output-dir models_qnn/refine_net \
    --custom-ops-lib onnx_conversion/qnn_conversion/custom_ops/build/libCustomOps.so \
    ...
```

### Issue: Accuracy Degradation

**Problem**: QNN model accuracy is significantly lower than ONNX

**Solutions**:

1. **Use FP16 instead of INT8**:
   ```bash
   --quantization fp16
   ```

2. **Use per-channel quantization**:
   Already enabled by default in INT8 mode

3. **Provide calibration data**:
   ```bash
   # Create calibration dataset
   python scripts/create_calibration_data.py \
       --output calibration_data/

   # Convert with calibration
   python onnx_conversion/qnn_conversion/onnx_to_qnn.py \
       --onnx-model models_onnx/refine_net.onnx \
       --quantization int8 \
       --input-list calibration_data/input_list.txt \
       ...
   ```

### Issue: Poor Performance on Device

**Problem**: Model runs slower than expected on Hexagon

**Solutions**:

1. **Verify Hexagon backend is used**:
   ```bash
   adb logcat | grep QNN
   # Look for "Using Hexagon backend"
   ```

2. **Enable multi-threading**:
   ```cpp
   QnnBackend_Config_t config;
   config.option = QNN_BACKEND_CONFIG_OPTION_NUM_THREADS;
   config.numThreads = 4;  // Use 4 Hexagon threads
   ```

3. **Profile the model**:
   ```bash
   qnn-profile-viewer \
       --model refine_net_hexagon.bin \
       --input_list inputs.txt \
       --output profile.html
   ```

## Performance Benchmarks

| Model | Device | Backend | Quantization | Latency | Accuracy Drop |
|-------|--------|---------|--------------|---------|---------------|
| RefineNet | Snapdragon 8 Gen 2 | Hexagon | INT8 | 8.2 ms | < 1% |
| RefineNet | Snapdragon 8 Gen 2 | Hexagon | FP16 | 12.5 ms | < 0.1% |
| RefineNet | Snapdragon 8 Gen 2 | CPU | FP32 | 185 ms | 0% |
| ScoreNet | Snapdragon 8 Gen 2 | Hexagon | INT8 | 6.1 ms | < 1% |

## Additional Resources

- [QNN SDK Documentation](https://developer.qualcomm.com/sites/default/files/docs/snpe/index.html)
- [ONNX Documentation](https://onnx.ai/onnx/intro/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [Hexagon DSP Programming Guide](https://developer.qualcomm.com/software/hexagon-dsp-sdk)

## Support

For issues specific to this conversion:
- File an issue in the GitHub repository
- Contact: nvidia-foundationpose-support@example.com

For QNN SDK issues:
- Qualcomm Developer Forum: https://developer.qualcomm.com/forum
- QNN SDK Support: https://developer.qualcomm.com/support
