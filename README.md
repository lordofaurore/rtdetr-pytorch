# RT-DETR PyTorch - Custom Training on COCO

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)

A PyTorch implementation of **RT-DETR** (Real-Time DEtection TRansformer) with custom training on COCO dataset. This repository contains the complete training pipeline, configuration files, and a **pre-trained model** achieving **49.1% mAP** on COCO val2017.

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **mAP (IoU=0.50:0.95)** | **49.1%** |
| AP50 (IoU=0.50) | 66.9% |
| AP75 (IoU=0.75) | 53.0% |
| AP (small objects) | 30.8% |
| AP (medium objects) | 53.6% |
| AP (large objects) | 67.2% |

### Performance Details
