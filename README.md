# RT-DETR PyTorch - Custom Training on COCO

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)
![RTX](https://img.shields.io/badge/RTX-5060%20Ti-76B900.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)

A PyTorch implementation of **RT-DETR** (Real-Time DEtection TRansformer) with custom training on COCO dataset. This repository contains the complete training pipeline, configuration files, and a **pre-trained model** achieving **49.1% mAP** on COCO val2017.

---

## 🖥️ Hardware & CUDA Configuration

| Component | Version |
|-----------|---------|
| **GPU** | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) |
| **CUDA Version** | 13.0 (driver) / 12.8 (PyTorch runtime) |
| **PyTorch CUDA** | 2.7.0+cu128 |
| **GPU Architecture** | Blackwell (sm_120) |

> ⚠️ **Important**: This model requires CUDA 12.8+ support. The `+cu128` suffix in PyTorch indicates compatibility with CUDA 12.8 runtime.

### CUDA Verification

```bash
# Check your CUDA version
nvidia-smi

# Verify PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
