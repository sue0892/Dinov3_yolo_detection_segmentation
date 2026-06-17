#!/usr/bin/env python3
"""
Inference script for object detection on large-scale satellite images.

This script tiles large images, runs a PyTorch detection model on each tile,
merges detections into full-image coordinates, applies class-wise NMS, and
saves JSON results and optional visualization images.

Usage examples:

# Minimal (requires a model builder callable in your repo or a checkpoint that
# can be loaded into a model built from a builder):
python tools/inference_satellite.py \
  --weights weights/checkpoint.pth \
  --builder models:build_model \
  --input data/satellite/IMG_001.tif \
  --output runs/inference/IMG_001

# Directory input:
python tools/inference_satellite.py --weights weights/checkpoint.pth \
  --builder models:build_model --input data/satellite/ --output runs/inference/

Notes on model loading:
- Prefer using --builder to point to a callable that builds your model and
  returns a torch.nn.Module (e.g. "models:build_model" means `from models import build_model`).
- If you do not pass --builder, the script will attempt to load a checkpoint
  directly with torch.load and expects it to be a state_dict or a dict with
  key "model_state_dict" or "state_dict".
- The model is expected to return detections in one of these forms:
  1) A list of dicts like [{'boxes': Tensor[N,4], 'scores': Tensor[N], 'labels': Tensor[N]}]
  2) A Tensor[N, 5+] where columns are [x1,y1,x2,y2,score,(label)]
  Adapt the `parse_model_outputs()` function if your model returns something else.

Dependencies: torch, torchvision, numpy, opencv-python, tqdm

"""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from tqdm import tqdm


def import_callable(spec: str):
    """Import a callable using a spec like "module.sub:callable" or "module:callable".
    Returns the callable.
    """
    if ":" not in spec:
        raise ValueError("builder spec must be in the form module:callable")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def tile_image(img: np.ndarray, tile_size: int, overlap: int) -> List[Tuple[np.ndarray, int, int]]:
    """Split image into overlapping tiles.
    Returns list of (tile_image, x0, y0) where x0,y0 is top-left in original image.
    """
    h, w = img.shape[:2]
    stride = tile_size - overlap
    tiles = []
    if stride <= 0:
        raise ValueError("tile_size must be larger than overlap")
    xs = list(range(0, max(w - tile_size + 1, 1), stride))
    ys = list(range(0, max(h - tile_size + 1, 1), stride))

    # Ensure last tile covers the right/bottom edge
    if xs[-1] + tile_size < w:
        xs.append(max(w - tile_size, 0))
    if ys[-1] + tile_size < h:
        ys.append(max(h - tile_size, 0))

    for y in ys:
        for x in xs:
            tile = img[y : y + tile_size, x : x + tile_size].copy()
            tiles.append((tile, x, y))
    return tiles


def preprocess_tiles(tiles: List[np.ndarray], input_size: int = None, device="cpu") -> torch.Tensor:
    """Convert list of BGR numpy tiles to a torch tensor batch on device.

    If input_size is provided, tiles are resized (bilinear) to (input_size, input_size).
    Normalization: convert to RGB, float32, /255, and normalize using ImageNet mean/std.
    """
    imgs = []
    for tile in tiles:
        img = tile[:, :, ::-1]  # BGR->RGB
        if input_size is not None:
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        # ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)
    batch = np.stack(imgs, axis=0)
    tensor = torch.from_numpy(batch).to(device)
    return tensor


def parse_model_outputs(raw_outputs, tile_coords, scale_factors, conf_thres: float):
    """Convert model outputs into a flat list of detections in full-image coords.

    raw_outputs: either:
      - list (len=batch) of dicts {'boxes':Tensor[N,4], 'scores':Tensor[N], 'labels':Tensor[N]}
      - Tensor of shape (batch, N, 6) or list of tensors
    tile_coords: list of (x0,y0)
    scale_factors: list of scale factor applied to each tile (if resized to input_size)

    Returns list of dicts: {'x1','y1','x2','y2','score','label'}
    """
    detections = []

    # Case: list of dicts per image
    if isinstance(raw_outputs, (list, tuple)) and len(raw_outputs) == len(tile_coords):
        for out, (x0, y0), s in zip(raw_outputs, tile_coords, scale_factors):
            if not isinstance(out, dict):
                # Maybe it's a Tensor N x >=5
                if isinstance(out, torch.Tensor):
                    arr = out.detach().cpu().numpy()
                    for row in arr:
                        if row[4] < conf_thres:
                            continue
                        x1, y1, x2, y2 = row[:4] / s
                        score = float(row[4])
                        label = int(row[5]) if row.shape[0] > 5 else 0
                        detections.append({
                            'x1': float(x1 + x0),
                            'y1': float(y1 + y0),
                            'x2': float(x2 + x0),
                            'y2': float(y2 + y0),
                            'score': score,
                            'label': label,
                        })
                    continue
                else:
                    continue

            boxes = out.get('boxes')
            scores = out.get('scores')
            labels = out.get('labels', None)
            if boxes is None or scores is None:
                continue
            boxes = boxes.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            if labels is not None:
                labels = labels.detach().cpu().numpy()
            else:
                labels = np.zeros(len(scores), dtype=np.int32)

            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                if score < conf_thres:
                    continue
                # If tiles were resized, boxes are in resized tile coordinates; scale back
                x1 /= s
                y1 /= s
                x2 /= s
                y2 /= s
                detections.append({
                    'x1': float(x1 + x0),
                    'y1': float(y1 + y0),
                    'x2': float(x2 + x0),
                    'y2': float(y2 + y0),
                    'score': float(score),
                    'label': int(label),
                })
        return detections

    # Case: single tensor with shape (batch, N, >=5) or list of tensors for each batch item
    if isinstance(raw_outputs, torch.Tensor):
        raw = raw_outputs.detach().cpu().numpy()
        # Expect shape (batch, N, >=5)
        if raw.ndim == 3:
            for b_idx in range(raw.shape[0]):
                arr = raw[b_idx]
                (x0, y0) = tile_coords[b_idx]
                s = scale_factors[b_idx]
                for row in arr:
                    if row[4] < conf_thres:
                        continue
                    x1, y1, x2, y2 = row[:4] / s
                    score = float(row[4])
                    label = int(row[5]) if row.shape[0] > 5 else 0
                    detections.append({
                        'x1': float(x1 + x0),
                        'y1': float(y1 + y0),
                        'x2': float(x2 + x0),
                        'y2': float(y2 + y0),
                        'score': score,
                        'label': label,
                    })
        return detections

    # Unknown type
    return detections


def run_nms_all(detections: List[dict], iou_thres: float = 0.5):
    """Perform class-wise NMS on detections list. Returns filtered detections.
    Each detection dict must contain x1,y1,x2,y2,score,label.
    """
    if len(detections) == 0:
        return []
    boxes = np.array([[d['x1'], d['y1'], d['x2'], d['y2']] for d in detections], dtype=np.float32)
    scores = np.array([d['score'] for d in detections], dtype=np.float32)
    labels = np.array([d['label'] for d in detections], dtype=np.int64)

    keep_mask = np.zeros(len(detections), dtype=bool)
    final = []
    unique_labels = np.unique(labels)
    for c in unique_labels:
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        cls_boxes = torch.from_numpy(boxes[idxs]).float()
        cls_scores = torch.from_numpy(scores[idxs]).float()
        keep = nms(cls_boxes, cls_scores, iou_thres)
        kept_idxs = idxs[keep.numpy().astype(int)]
        for ki in kept_idxs:
            final.append(detections[int(ki)])
    # sort by score desc
    final = sorted(final, key=lambda x: x['score'], reverse=True)
    return final


def visualize_and_save(img: np.ndarray, detections: List[dict], out_path: Path, max_display: int = 200):
    vis = img.copy()
    color_map = {}
    for i, d in enumerate(detections[:max_display]):
        c = int(d['label'])
        if c not in color_map:
            color_map[c] = tuple(int(x) for x in np.random.randint(0, 255, size=3).tolist())
        x1, y1, x2, y2 = map(int, [d['x1'], d['y1'], d['x2'], d['y2']])
        score = d['score']
        color = color_map[c]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        txt = f"{c}:{score:.2f}"
        cv2.putText(vis, txt, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def infer_image(
    img_path: Path,
    model,
    device: str,
    tile_size: int,
    overlap: int,
    input_size: int,
    batch_size: int,
    conf_thres: float,
    iou_thres: float,
    visualize: bool,
    output_base: Path,
):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    H, W = img.shape[:2]
    tiles_info = tile_image(img, tile_size=tile_size, overlap=overlap)
    tiles = [t for (t, _, _) in tiles_info]
    coords = [(x, y) for (_, x, y) in tiles_info]

    # For each tile we may resize to input_size
    scale_factors = []
    for t in tiles:
        if input_size is None:
            scale_factors.append(1.0)
        else:
            h, w = t.shape[:2]
            # assume square resize
            s = input_size / max(h, w)
            # Our preprocess resizes to (input_size,input_size) regardless of aspect, so scale factor
            # is ratio between resized coordinates and original tile coords: s. But we later divide boxes by s
            scale_factors.append(s)

    detections = []
    # Process in batches
    for i in tqdm(range(0, len(tiles), batch_size), desc=f"Tiles ({img_path.name})"):
        batch_tiles = tiles[i : i + batch_size]
        batch_coords = coords[i : i + batch_size]
        batch_scales = scale_factors[i : i + batch_size]
        input_tensor = preprocess_tiles(batch_tiles, input_size=input_size, device=device)
        with torch.no_grad():
            outputs = model(input_tensor) if callable(model) else model(input_tensor)
        # outputs may be a list [out1, out2, ...] where each out corresponds to a tile
        batch_dets = parse_model_outputs(outputs, batch_coords, batch_scales, conf_thres)
        detections.extend(batch_dets)

    # Merge and NMS
    final_dets = run_nms_all(detections, iou_thres=iou_thres)

    # Save results
    out_base = output_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({'image': str(img_path.name), 'width': W, 'height': H, 'detections': final_dets}, f, indent=2)

    if visualize:
        vis_path = out_base.with_suffix('.jpg')
        visualize_and_save(img, final_dets, vis_path)

    return json_path


def build_model_from_builder(builder_spec: str, weights: str, device: str):
    builder = import_callable(builder_spec)
    model = builder()
    model.to(device)
    # load weights intelligently
    ckpt = torch.load(weights, map_location=device)
    if isinstance(ckpt, dict):
        # common keys
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model' in ckpt:
            state = ckpt['model']
        else:
            # may already be a state dict
            state = ckpt
        try:
            model.load_state_dict(state)
        except Exception as e:
            print("Warning: model.load_state_dict failed with:", e)
            # try to load with strict=False
            model.load_state_dict(state, strict=False)
    else:
        # ckpt might already be a state_dict
        try:
            model.load_state_dict(ckpt)
        except Exception as e:
            print("Warning: loading raw checkpoint failed:", e)
    model.eval()
    return model


def try_load_state_dict_only(weights: str, device: str):
    # Try to load a state_dict and wrap in a dummy pass-through model that expects images and returns
    # torchvision-like outputs. This is a fallback and will likely need adapting for your repo.
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
        print("Loaded a raw state_dict — you should pass --builder to construct the model architecture.")
    else:
        print("Checkpoint format not recognized. Pass --builder to build model or adapt this script.")
    raise RuntimeError("Unable to build model without a builder callable. Provide --builder.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='Path to model checkpoint (.pth)')
    p.add_argument('--builder', default=None, help='Python builder spec to create model, e.g. "models:build_model"')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--input', required=True, help='Image file or directory of images')
    p.add_argument('--output', required=True, help='Output file or directory base for results')
    p.add_argument('--tile-size', type=int, default=2048, help='Tile size (px) to split large images')
    p.add_argument('--overlap', type=int, default=200, help='Overlap (px) between tiles')
    p.add_argument('--input-size', type=int, default=None, help='Model input size (square). If given tiles are resized to this size for inference')
    p.add_argument('--batch-size', type=int, default=4, help='Number of tiles to run per batch')
    p.add_argument('--conf-thres', type=float, default=0.25)
    p.add_argument('--iou-thres', type=float, default=0.5)
    p.add_argument('--visualize', action='store_true', help='Save visualized image with boxes')
    args = p.parse_args()

    device = args.device

    # Build or load model
    if args.builder:
        model = build_model_from_builder(args.builder, args.weights, device)
    else:
        model = try_load_state_dict_only(args.weights, device)

    # Ensure model is callable: if it's an nn.Module we wrap a simple forward
    if isinstance(model, torch.nn.Module):
        def model_forward(x):
            # many detection models accept list[Tensor] of images in 0-1 normalized range
            try:
                out = model(x)
            except Exception:
                # try splitting to list
                imgs = [im for im in x]
                out = model(imgs)
            return out
        model_fn = model_forward
    else:
        model_fn = model

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        images = sorted([p for p in input_path.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']])
    else:
        images = [input_path]

    for img_path in images:
        rel_out = output_path
        if output_path.is_dir() or str(output_path).endswith('/'):
            rel_out = output_path / img_path.stem
        print(f"Processing {img_path} -> {rel_out}")
        try:
            json_path = infer_image(
                img_path=img_path,
                model=model_fn,
                device=device,
                tile_size=args.tile_size,
                overlap=args.overlap,
                input_size=args.input_size,
                batch_size=args.batch_size,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                visualize=args.visualize,
                output_base=Path(rel_out),
            )
            print(f"Saved results to {json_path}")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")


if __name__ == '__main__':
    main()
