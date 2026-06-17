# Dinov3 + YOLO — Detection & Segmentation

Customize YOLO with a DINOv3 backbone for improved object detection and semantic segmentation.

This repository integrates a DINOv3 visual backbone into a YOLO-style detection/segmentation framework to combine DINOv3's strong representation learning with YOLO's efficient detection head. It supports training, evaluation, and inference for object detection and per-instance/semantic segmentation on standard datasets (COCO, YOLO format, or custom datasets).

Table of contents
- [Features](#features)
- [Requirements](#requirements)
- [Quick install](#quick-install)
- [Repository layout](#repository-layout)
- [Quick start — inference](#quick-start--inference)
- [Training](#training)
- [Evaluation](#evaluation)
- [Datasets and formatting](#datasets-and-formatting)
- [Configuration](#configuration)
- [Tips for fine-tuning](#tips-for-fine-tuning)
- [Model zoo / pre-trained weights](#model-zoo--pre-trained-weights)
- [Contributing](#contributing)
- [License & citation](#license--citation)
- [Contact](#contact)

## Features
- Replace standard YOLO backbone with DINOv3 for more robust visual features.
- Support for object detection and semantic/instance segmentation heads.
- Training, evaluation, and inference scripts with configuration-driven experiments.
- Support for single-GPU and multi-GPU training (DistributedDataParallel).
- Dataset helpers for COCO and YOLO-style datasets.

## Requirements
- Python 3.8+
- PyTorch (1.12+ recommended) with CUDA (for GPU training)
- torchvision
- common ML libraries: numpy, tqdm, opencv-python, Pillow
- COCO tools (pycocotools) for COCO-style evaluation
- Optional: albumentations for augmentations, detectron2-style utilities if used

Example:
- torch >= 1.12
- torchvision >= 0.13
- pycocotools
- albumentations
- opencv-python
- tqdm

Note: Exact dependency versions are not listed here — if you maintain a requirements.txt in the repo, prefer installing from that file.

## Quick install
Clone and set up a virtual environment:

```bash
git clone https://github.com/sue0892/Dinov3_yolo_detection_segmentation.git
cd Dinov3_yolo_detection_segmentation
python -m venv .venv
source .venv/bin/activate
pip install -U pip
# If you have requirements.txt:
pip install -r requirements.txt
# Otherwise install main dependencies:
pip install torch torchvision pycocotools opencv-python tqdm albumentations
```

Prepare DINOv3 backbone weights (see [Model zoo](#model-zoo--pre-trained-weights) below) and point the training/inference scripts to the checkpoint.

## Repository layout
(Adjust to match the actual repo structure — these are suggested locations)
- configs/                — YAML config files for models and experiments
- datasets/               — dataset helpers and data loaders (COCO / YOLO)
- models/                 — model definitions (YOLO heads + DINOv3 backbone)
- tools/
  - train.py              — training entrypoint
  - eval.py               — evaluation script
  - inference.py          — run inference on images / video
- weights/                — place pretrained weights here (optional)
- README.md

## Quick start — inference
Run inference on a folder of images with a trained model:

```bash
python tools/inference.py \
  --config configs/dino_yolo_seg.yaml \
  --weights path/to/checkpoint.pth \
  --input data/images/ \
  --output runs/inference/
```

Options (common):
- --config : path to model/experiment YAML
- --weights: path to model checkpoint (.pth)
- --input  : file, folder, or glob of images
- --output : directory to save predictions/visualizations
- --conf-thres, --iou-thres: thresholding for NMS / detections

Output contains detection boxes, segmentation masks (if applicable), and visualization images.

## Training
Train a model (single-GPU example):

```bash
python tools/train.py \
  --config configs/dino_yolo_seg.yaml \
  --data data/my_dataset.yaml \
  --weights path/to/dinov3_backbone.pth \
  --epochs 50 \
  --batch-size 16 \
  --device cuda:0
```

Multi-GPU (launch with torch.distributed.launch or torchrun):

```bash
torchrun --nproc_per_node=4 tools/train.py \
  --config configs/dino_yolo_seg.yaml \
  --data data/my_dataset.yaml \
  --weights path/to/dinov3_backbone.pth \
  --epochs 50
```

Common flags:
- --config: model architecture and training hyperparameters
- --data: dataset YAML describing train/val paths and class names
- --weights: pretrained backbone or checkpoint to resume from
- --resume: resume from checkpoint (continue training)
- --lr, --optimizer, --scheduler: learning rate and optimizer settings

Checkpointing: training saves checkpoints (typically best and last). Keep an eye on learning-rate schedule and batch size scaling when changing GPUs.

## Evaluation
Evaluate a checkpoint on validation/test set (COCO metrics supported):

```bash
python tools/eval.py \
  --config configs/dino_yolo_seg.yaml \
  --weights runs/checkpoint_best.pth \
  --data data/my_dataset.yaml \
  --task detection  # or segmentation / both
```

Evaluation supports:
- COCO mAP for detection
- AP/IoU metrics for segmentation (if masks present)

## Datasets and formatting
This repository supports:
- COCO format (recommended for segmentation + detection)
- YOLO-style annotations (images + .txt label files) — conversion scripts may be required for segmentation masks

Example data YAML (my_dataset.yaml):

```yaml
train: /path/to/train/images
val:   /path/to/val/images
nc: 80
names: ['person','bicycle', ...]
```

For segmentation datasets you should provide masks in COCO RLE or polygon format. If you use instance segmentation, ensure annotations include segmentation polygons or RLE encoding.

If you need to convert between formats, use COCO API or common conversion scripts (not included here).

## Configuration
Configs live under `configs/`. Typical config sections:
- model:
  - backbone: dino_v3
  - head: yolovX-style detection/segmentation head
  - anchors, strides, channels
- training:
  - batch_size, epochs, optimizer, lr_schedule
- augmentations: training-time augmentations
- inference:
  - confidence, nms_iou, input_size

Edit or duplicate YAML configs to run different experiments.

## Tips for fine-tuning
- Freeze backbone initially and train heads for a few epochs, then unfreeze and fine-tune with lower LR.
- Use linear scaling rule for LR when changing batch size.
- Use mixed precision (AMP) for faster training and lower memory usage.
- Augmentations: mosaic, random scale, color jitter often help detection; for segmentation, keep geometry-preserving augmentations consistent with masks.
- Validate often and monitor both detection mAP and segmentation metrics.

## Model zoo / pre-trained weights
- DINOv3 backbone: download official DINOv3 pre-trained weights from the DINO authors/official repository and set the path in config or use env var `DINOV3_WEIGHTS`.
- Example:
  - place weights at `weights/dino_v3_ckpt.pth`
  - in config: `backbone.pretrained: weights/dino_v3_ckpt.pth`

If you publish trained checkpoints for this repo, include a table with model names, tasks (detection/segmentation), dataset, and mAP/mIoU.

## Contributing
Contributions are welcome.
- Create an issue for a bug, enhancement, or model/result you'd like to share.
- Fork the repo, create a feature branch, open a PR with a clear description and reproducible steps.
- Add tests or example configs where relevant.

Guidelines:
- Keep changes focused per PR.
- Provide training logs or evaluation metrics when proposing model changes.

## License & citation
License: MIT (or choose your preferred license). Add LICENSE file to the repository.

If you use DINOv3 or other works, please cite the original papers. Example citation (replace with exact references for DINOv3 and YOLO variants):

- Caron, M. et al., DINOv3 — [paper / repo]
- Redmon, J. et al. / Glenn Jocher et al. (YOLO variants) — [respective papers/repos]

## Contact
Maintainer: sue0892 (GitHub)  
For questions or model results, open an issue or PR.
