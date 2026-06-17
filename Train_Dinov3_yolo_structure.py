import torch
from ultralytics import YOLO
import argparse
import os
# =========================================================
# 1. Importing the custom module to trigger the registration logic. 
#    Key point: This must be executed before YOLO (args.cfg).
# =========================================================
import Dinov3_yolo_structure

def print_model_size(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / 1e6  
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_m = trainable_params / 1e6  
    
    print(f"\n[{'-'*30}]")
    print(f"Total number of model parameters: {total_params_m:.2f} M")
    print(f"Number of trainable parameters: {trainable_params_m:.2f} M")
    print(f"Number of frozen parameters: {(total_params_m - trainable_params_m):.2f} M")
    print(f"[{'-'*30}]\n")
    
    return total_params_m, trainable_params_m

# =========================================================
# 2. Loading DINOv3 weights
# =========================================================
def load_dinov3_weights(vit_model, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"!!!Warning: File not found {ckpt_path}.")
        return

    print(f"Loading DINOv3 pre-trained weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if "teacher" in ckpt:
        state_dict = ckpt["teacher"]
    elif "student" in ckpt:
        state_dict = ckpt["student"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    state_dict = {k: v for k, v in state_dict.items() if "head" not in k}

    msg = vit_model.load_state_dict(state_dict, strict=False)
    
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")
    print("DINOv3 weights loading completed!\n")

# =========================================================
# 3. training YOLO-DINO
# =========================================================
def train_yolodino(args):
    print(f"Initializing YOLO model with config {args.cfg}...")
    model = YOLO(args.cfg)

    vit_instance = model.model.model[0].vit
    load_dinov3_weights(vit_instance, args.dino_weights)

    print("Freezing DINOv3 backbone...")
    for param in model.model.model[0].parameters():
        param.requires_grad = False
    print_model_size(model.model)

    print("Starting training...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=False, 
        freeze=[0],       
        save=True,
        save_period=args.save_period,
        augment=True,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.1,
        flipud=0.1,
        fliplr=0.1,
        optimizer='AdamW',
        lr0=1e-4,
        weight_decay=0.05,
        batch=args.batch, 
    )

    print("\nTraining is complete; now performing the final evaluation on the validation set...")
    metrics = model.val()
    print("Evaluation results on validation set:", metrics)


# =========================================================
# 4. argparse
# =========================================================
def build_argparser():
    parser = argparse.ArgumentParser(description="Train YOLO-DINO model with configurable args")
    parser.add_argument("--cfg", type=str, default="Dinov3_yolo_detection.yaml", help="Model configuration file path")
    parser.add_argument("--data", type=str, default="./data/data.yaml")
    parser.add_argument("--dino_weights", type=str, default="./weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth") 
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=str, default="./runs/yolodino")
    parser.add_argument("--name", type=str, default="finetune")
    parser.add_argument("--save_period", type=int, default=-1)
    return parser

def main():
    args = build_argparser().parse_args()
    train_yolodino(args)

if __name__ == '__main__':
    main()
