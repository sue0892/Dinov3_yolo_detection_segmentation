'''
Customize an Ultralytics YOLO model (e.g, yolov11) with a DINOv3 backbone
Model_structure: 4 transformer blocks + 4 pyramid features, with an adaptation layer consisting of 1*1 Conv + BN + LeakyReLU, and FPN-based feature fusion
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Handling the dimensions of the input tensors
# ---------------------------
def ensure_divisible_by_16(x):
    h, w = x.shape[-2], x.shape[-1]

    if h % 16 != 0 or w % 16 != 0:
        new_h = (h // 16 + 1) * 16
        new_w = (w // 16 + 1) * 16
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return x

def ensure_divisible_by_32(x):
    h, w = x.shape[-2], x.shape[-1]

    if h % 32 != 0 or w % 32 != 0:
        new_h = (h // 32 + 1) * 32
        new_w = (w // 32 + 1) * 32
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return x

# ---------------------------
# Loading vit backbone，getting patch tokens' features，reshaping to（B,C,H,W）
# ---------------------------       
class DINOv3Backbone(nn.Module):
    def __init__(self, model_name='dinov3_vitl16', output_indices=[5, 11, 17, 23], repo_or_dir="dinov3_main"):
        super().__init__()
        self.vit = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model=model_name,
            source='local',
            pretrained=False,
        )
        self.output_indices = output_indices
        self.embed_dim = self.vit.embed_dim
        self.out_channels = self.embed_dim  # Single value for backward compatibility

    def forward(self, x):
        x = ensure_divisible_by_32(x)
        B, _, H1, W1 = x.shape
        H, W = H1 // 16, W1 // 16
        
        layers = self.vit.get_intermediate_layers(x, n=self.output_indices, return_class_token=False)
        feats = []
        for i, layer in enumerate(layers):
            feat = layer.permute(0, 2, 1).reshape(B, self.embed_dim, H, W).contiguous()
            feats.append(feat)
        
        return feats

# ---------------------------
# ViT → YOLO adapter
# ---------------------------
class ViTMultiDepthAdapter(nn.Module):
    def __init__(self, out_channels=384, in_channels=1024):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # P2: from layer 6 (feats[0]), 1/16 -> 1/4 (4x Upsample)
        self.p2_gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        # P3: from layer 12 (feats[1]), 1/16 -> 1/8 (2x Upsample)
        self.p3_gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        # P4: from layer 18 (feats[2]), 1/16 -> 1/16 (Keep)
        self.p4_gen = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        # P5: from layer 24 (feats[3]), 1/16 -> 1/32 (2x Downsample)
        self.p5_gen = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, feats): 
        f6, f12, f18, f24 = feats
        return [
            self.p2_gen(f6),  # P2 (1/4)
            self.p3_gen(f12), # P3 (1/8)
            self.p4_gen(f18), # P4 (1/16)
            self.p5_gen(f24)  # P5 (1/32)
        ]
    
# ---------------------------
# creating "index layer" for YOLO YAML by splitting the lists
# ---------------------------
class Split(nn.Module):
    def __init__(self, out_channels=384, n=0): 
        super().__init__()
        self.n = n
        self.out_channels = out_channels

    def forward(self, x):
        # x is the list from Adapter [P2, P3, P4, P5]  
        return x[self.n] 


# ---------------------------
# Registering Custom Modules
# ---------------------------
import ultralytics.nn.modules as ul_modules
from ultralytics.nn import tasks
module_dict = tasks.__dict__
module_dict['DINOv3Backbone'] = DINOv3Backbone
module_dict['ViTMultiDepthAdapter'] = ViTMultiDepthAdapter
module_dict['Split'] = Split

ul_modules.DINOv3Backbone = DINOv3Backbone
ul_modules.ViTMultiDepthAdapter = ViTMultiDepthAdapter
ul_modules.Split = Split