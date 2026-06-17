# Dinov3 + YOLO — Detection & Segmentation

Customize YOLO with a DINOv3 backbone for improved object detection and semantic segmentation.

This repository integrates a DINOv3 visual backbone into a YOLO-style detection/segmentation framework to combine DINOv3's strong representation learning with YOLO's efficient detection head. It supports training, evaluation, and inference for object detection and per-instance/semantic segmentation on standard datasets (YOLO format, or custom datasets).

Table of contents
- [Features](#features)
- [Requirements](#requirements)
- [Quick install](#quick-install)
- [Repository layout](#repository-layout)
- [Datasets and formatting](#datasets-and-formatting)
- [Training](#training)
- [Inference](#Inference)
- [Contributing](#contributing)
- [License & citation](#license--citation)
- [Contact](#contact)

## Features
- Replace standard YOLO backbone with DINOv3 for more robust visual features.
- Support for object detection and semantic/instance segmentation heads.
- Training, evaluation, and inference scripts with configuration-driven experiments.
- Support for single-GPU and multi-GPU training (DistributedDataParallel).
- Dataset helpers for YOLO-style datasets.

## Requirements
```bash
torch==2.5.1+cu121
python==3.12.9 
ultralytics==8.3.174
```

## Quick install
```bash
conda create -n dino_yolo python=3.12.9
conda activate dino_yolo
pip install -r requirements.txt
```

Clone the DINOv3 repository:
```bash
git clone https://github.com/facebookresearch/dinov3.git
```

Download [DINOv3](https://huggingface.co/MVRL/dinov3_vitl16_sat/blob/main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth)  pretrained weights and place them in:
```
./dinov3_pretrained_weight
```

## Repository layout
```bash
│  Dinov3_yolo_detection.yaml
│  Dinov3_yolo_structure.py
│  Inference_satellite_img.py
│  README.md
│  requirements.txt
│  Train_Dinov3_yolo_structure.py
│
├─data
│  │  data.yaml
│  │
│  ├─images
│  │  ├─train
│  │  └─val
│  └─labels
│      ├─train
│      └─val
├─dinov3_main
│      git_clone.txt
│
└─dinov3_pretrained_weight
        download_dinov3_pretrained_weight.txt
```

## Datasets and formatting
This repository supports:
- YOLO-style annotations (images + .txt label files) 

Example data YAML:
/data/data.yaml

For segmentation datasets you should provide masks in polygon format. 

## Training
Train a model

```bash
python Train_Dinov3_yolo_structure.py \
  --cfg Dinov3_yolo_detection.yaml \
  --data ./data/data.yaml \
  --dino_weights ./dinov3_pretrained_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
  --device 0
```

Common flags:
- --cfg: model architecture and training hyperparameters
- --data: dataset YAML describing train/val paths and class names
- --dino_weights: pretrained backbone or checkpoint to resume from

Checkpointing: training saves checkpoints (typically best and last). Keep an eye on learning-rate schedule and batch size scaling when changing GPUs.

## Inference
Run inference on a folder of images with a trained model:

Single image input:
```bash
python Inference_satellite_img.py \
  --weights weights/checkpoint.pth \
  --builder models:build_model \
  --input data/satellite/IMG_001.tif \
  --output runs/inference/IMG_001
```

Directory input:
```bash
python Inference_satellite_img.py \
  --weights weights/checkpoint.pth \
  --builder models:build_model \
  --input data/satellite/ \
  --output runs/inference/
```
Minimal: requires a model builder callable in your repo or a checkpoint that can be loaded into a model built from a builder.

## Contributing
Contributions are welcome.
- Create an issue for a bug, enhancement, or model/result you'd like to share.
- Fork the repo, create a feature branch, open a PR with a clear description and reproducible steps.
- Add tests or example configs where relevant.

Guidelines:
- Keep changes focused per PR.
- Provide training logs or evaluation metrics when proposing model changes.

## License & citation
License: Apache-2.0

If you only use DINOv3 or other works, please cite the original papers. 

## Contact
Maintainer: sue0892 (GitHub)  
For questions or model results, open an issue or PR.
