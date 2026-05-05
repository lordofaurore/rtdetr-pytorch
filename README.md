# RT-DETR PyTorch — Custom Training on COCO

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.7.0+cu128-red.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/CUDA-12.8-green.svg" alt="CUDA"/>
  <img src="https://img.shields.io/badge/GPU-RTX%205060%20Ti%2016GB-76B900.svg" alt="GPU"/>
  <img src="https://img.shields.io/badge/mAP-49.1%25-orange.svg" alt="mAP"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License"/>
</p>

<p align="center">
  A PyTorch implementation of <strong>RT-DETR</strong> (Real-Time DEtection TRansformer) trained on COCO.<br/>
  This repository includes the complete training pipeline, configuration files, and a <strong>pre-trained model achieving 49.1% mAP</strong> on COCO val2017.
</p>


## 🖥️ Hardware & CUDA Configuration

| Component | Version |
|-----------|---------|
| **GPU** | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) |
| **CUDA Driver** | 13.0 |
| **PyTorch CUDA runtime** | 12.8 (`+cu128`) |
| **GPU Architecture** | Blackwell (sm_120) |

> ⚠️ **Important**: This project requires CUDA 12.8+ and PyTorch 2.7+ for native Blackwell architecture support (RTX 50xx).

### CUDA Verification

```bash
nvidia-smi

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True
Version: 12.8
GPU: NVIDIA GeForce RTX 5060 Ti
```


## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **mAP** (IoU=0.50:0.95) | **49.1%** |
| AP50 (IoU=0.50) | 66.9% |
| AP75 (IoU=0.75) | 53.0% |
| AP small objects | 30.8% |
| AP medium objects | 53.6% |
| AP large objects | 67.2% |
| AR @maxDets=100 | 69.3% |

<details>
<summary>View detailed COCO metrics</summary>

```
IoU metric: bbox
Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.491
Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.669
Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.530
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.308
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.536
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.672
Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.693
```
</details>
