# ==========================================================
# AI Background Remover
# Copyright (C) 2026 Saw it See had
# Licensed under the MIT License
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import os

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class MyModel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(MyModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256 // factor)
        
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)

class RembgEngine:
    def __init__(self, model_name="tiny_reMBG.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--> Engine running on: {self.device}")

        self.model = MyModel(n_channels=3, n_classes=1).to(self.device)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', model_name)

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval() 
                print(f"--> Model loaded: {model_name}")
            except Exception as e:
                print(f"--> [ERROR] Failed to load model weights: {e}")
        else:
            print(f"--> [ERROR] Model file not found at: {model_path}")

        self.preprocess = transforms.Compose([
            transforms.Resize((320, 320)), 
            transforms.ToTensor(),
        ])

    def remove_background(self, input_path):
        """
        Takes an image path, returns a PIL Image with background removed.
        """
        try:
            original_image = Image.open(input_path).convert("RGB")
            original_size = original_image.size

            input_tensor = self.preprocess(original_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                mask_tensor = self.model(input_tensor)

            mask_np = mask_tensor.squeeze().cpu().numpy()
            
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
            mask_img = mask_img.resize(original_size, Image.Resampling.BILINEAR)
            
            mask_np_final = np.array(mask_img)
            threshold = 128
            mask_np_final = np.where(mask_np_final > threshold, 255, 0).astype(np.uint8)
            final_mask = Image.fromarray(mask_np_final, mode='L')

            result = original_image.convert("RGBA")
            result.putalpha(final_mask)
            
            return result
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
