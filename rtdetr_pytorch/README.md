# TODO

---

## Installation

<details>
<summary><strong>Prerequisites</strong></summary>

- Python 3.11+
- NVIDIA GPU with CUDA 12.8+ (Blackwell RTX 50xx architecture recommended)
- 16GB+ RAM (32GB recommended)
- 16GB+ VRAM for `batch_size=8`

</details>

<details>
<summary><strong>Step-by-step setup</strong></summary>

**1. Clone the repository**
```bash
git clone https://github.com/lordofaurore/rtdetr-pytorch.git
cd rtdetr-pytorch
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> Installs `torch==2.7.0+cu128`, `torchvision==0.22.0+cu128`, `pycocotools`, `onnx`, and other required packages.

**4. Download the COCO dataset**
```bash
mkdir -p dataset/coco/{annotations,train2017,val2017}

# Download from the official COCO website:
# http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# http://images.cocodataset.org/zips/train2017.zip
# http://images.cocodataset.org/zips/val2017.zip
```

</details>


## Training

<details>
<summary><strong>Train from scratch (12 epochs — "1x" schedule)</strong></summary>

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

</details>

<details>
<summary><strong>Resume from a checkpoint</strong></summary>

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r output/rtdetr_r50vd_6x_coco/checkpoint.pth
```

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

> **Windows users**: set `num_workers: 0` in config files to avoid memory issues.

</details>


## Evaluation

<details>
<summary><strong>Evaluate on COCO val2017</strong></summary>

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r output/rtdetr_r50vd_6x_coco/checkpoint.pth --test-only
```

</details>


## Inference

<details>
<summary><strong>Run inference on a single image</strong></summary>

```bash
python tools/infer.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r output/rtdetr_r50vd_6x_coco/checkpoint.pth \
    -f path/to/image.jpg
```

</details>

---

## Project Structure

```
rtdetr-pytorch/
├── configs/                 # Configuration files
│   ├── dataset/            # Dataset configs (COCO, etc.)
│   └── rtdetr/             # Model configs
├── src/                    # Core source code
│   ├── core/               # Core utilities
│   ├── data/               # Data loading & transforms
│   ├── misc/               # Miscellaneous helpers
│   ├── nn/                 # Neural network modules
│   ├── optim/              # Optimizers & schedulers
│   ├── solver/             # Training & evaluation loops
│   └── zoo/                # Model zoo (RT-DETR implementations)
├── tools/                  # Executable scripts
│   ├── train.py            # Training script
│   ├── infer.py            # Inference script
│   └── export_onnx.py      # ONNX export
├── output/                 # Checkpoints and logs (gitignored)
└── dataset/                # COCO dataset (gitignored)
```


## Configuration

Key training parameters in `configs/rtdetr/include/dataloader.yml`:

```yaml
train_dataloader:
  batch_size: 8        # Adjust based on available VRAM
  num_workers: 0       # Set to 0 on Windows
  shuffle: True

val_dataloader:
  batch_size: 8
  num_workers: 0
```

### VRAM usage by batch size (RTX 5060 Ti 16GB)

| Batch Size | Estimated VRAM | Status |
|-----------|----------------|--------|
| 4 | ~10 GB | ✅ Safe |
| 6 | ~13 GB | ✅ Comfortable |
| 8 | ~16 GB | ✅ Optimal |
| 12 | ~22 GB | ❌ Requires 24GB+ GPU |


## Training Progress

| Epoch | mAP | AP50 | Note |
|-------|-----|------|------|
| 6 | 45.3% | 62.4% | Intermediate checkpoint |
| **12** | **49.1%** | **66.9%** | **Final model (1x schedule)** |

**Training environment:**

| Parameter | Value |
|-----------|-------|
| GPU | RTX 5060 Ti (16GB VRAM) |
| CUDA | 12.8 (PyTorch 2.7.0+cu128) |
| Batch size | 8 |
| Input resolution | 640×640 |
| Optimizer | AdamW |
| LR Scheduler | MultiStepLR |
| Time per epoch | ~3h |


## ⚠️ Important Notes on CUDA

- **PyTorch version**: this repo uses `torch==2.7.0+cu128`, compiled with CUDA 12.8.
- **Driver compatibility**: your NVIDIA driver must support CUDA 12.8 (version ≥ 525.60.13).
- **RTX 50xx support**: PyTorch 2.7+ includes native support for Blackwell architecture (sm_120).
- **Verification**: always verify CUDA availability before launching training.


## Acknowledgments

- Original paper: [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- Official implementation: [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- PyTorch adaptation: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)


## 📄 License

This project is licensed under the **Apache 2.0 License** — see the [LICENSE](LICENSE) file for details.


## Author

**AARON CLARK** — [@lordofaurore](https://github.com/lordofaurore)


> **Note**: this is an independent implementation trained with a "1x" schedule (12 epochs) rather than the full "6x" schedule (72 epochs) from the original paper. The model achieves competitive results with significantly reduced training time.

⭐ If you find this repository useful, please consider giving it a star!
