"""U-Net avec encodeur ResNet18 pré-entraîné (ImageNet) et tête de sortie en
heatmaps (une carte par landmark). Dimensionné pour tourner confortablement
sur une RTX 2070 8 Go avec des images 256x256 (batch 16-32 en mixed precision).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """Encodeur ResNet18 (poids ImageNet) + décodeur U-Net -> heatmaps.

    Pour une entrée 256x256, la sortie heatmap est en 64x64 (stride 4), ce qui
    correspond à `heatmap_size` dans dataset.py / utils.py.
    """

    def __init__(self, n_keypoints: int = 19, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2,  64 ch
        self.pool = resnet.maxpool  # /2
        self.layer1 = resnet.layer1  # /4,  64 ch
        self.layer2 = resnet.layer2  # /8,  128 ch
        self.layer3 = resnet.layer3  # /16, 256 ch
        self.layer4 = resnet.layer4  # /32, 512 ch

        self.up3 = UpBlock(512, 256, 256)  # /32 -> /16
        self.up2 = UpBlock(256, 128, 128)  # /16 -> /8
        self.up1 = UpBlock(128, 64, 64)  # /8  -> /4  (= heatmap_size cible)

        self.head = nn.Conv2d(64, n_keypoints, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)  # /2
        p0 = self.pool(s0)  # /4
        s1 = self.layer1(p0)  # /4,  64
        s2 = self.layer2(s1)  # /8,  128
        s3 = self.layer3(s2)  # /16, 256
        s4 = self.layer4(s3)  # /32, 512

        d3 = self.up3(s4, s3)  # /16
        d2 = self.up2(d3, s2)  # /8
        d1 = self.up1(d2, s1)  # /4

        heatmaps = self.head(d1)  # (B, n_keypoints, H/4, W/4)
        return heatmaps


if __name__ == "__main__":
    model = ResNetUNet(n_keypoints=19)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print("Output shape:", out.shape)  # attendu: (2, 19, 64, 64)
