"""
UNet architecture used to predict landmark heatmaps from wing crops.

Moved here from landmarks/UNet_class_and_functions.py (Gabriel's original),
architecture unchanged. This module is now the single owner of the model
class: landmarks/predict.py should import UNet from here rather than
carrying its own copy.

Input:  (B, 3, IMG_HEIGHT, IMG_WIDTH) RGB crop, float in [0, 1]
Output: (B, 1, IMG_HEIGHT, IMG_WIDTH) single-channel heatmap, float in [0, 1]
        (one blob per landmark, landmarks are NOT distinguished by channel --
        identity/numbering is a separate downstream step)

Usage:
    from landmarks_trainer.model import UNet
    model = UNet()
    heatmap = model(crop_batch)  # crop_batch: (B, 3, 256, 512)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 conv + batchnorm + relu blocks, the basic UNet building block."""

    def __init__(self, in_channels: int, out_channels: int, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Standard 4-level UNet, reduced filter counts (16 at the first level)
    to limit overfitting given the small training set."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.encoder1 = DoubleConv(in_channels, 16)
        self.encoder2 = DoubleConv(16, 32)
        self.encoder3 = DoubleConv(32, 64)
        self.encoder4 = DoubleConv(64, 128)
        self.encoder5 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(64, 32)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(32, 16)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.final_activation = nn.Hardsigmoid()  # output bounded in [0, 1], matches heatmap range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        dec4 = self.decoder4(torch.cat((self.upconv4(enc5), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        return self.final_activation(self.final_conv(dec1))


def load_weights(weights_path: str, device: str = "cpu", in_channels: int = 3,
                  out_channels: int = 1) -> UNet:
    """Build a fresh UNet and load a state_dict checkpoint into it.

    This is the ONLY supported way to load a model going forward -- no more
    torch.save(model)/torch.load(weights_only=False) on the full pickled
    object. See migrate_legacy_weights.py for converting old-format .pth
    files.
    """
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
