"""
Reference Crop Classifier for Construction Site Safety Monitoring.
Used to evaluate Class-Conditional Accuracy (CCA) for generative models.
Classes:
  0: Compliant (Helmet + Vest)
  1: Missing Helmet
  2: Missing Vest
  3: Both Missing
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        return out


class ReferenceClassifier(nn.Module):
    """
    Lightweight ResNet specifically optimized for 64x64 worker crops.
    Outputs 4 class logits.
    """
    def __init__(self, num_classes=4, in_channels=3, base_channels=32):
        super().__init__()
        # 64x64 -> 64x64
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        # Stage 1: 64x64 -> 32x32
        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 2, stride=1)
        )
        # Stage 2: 32x32 -> 16x16
        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock(base_channels * 4, base_channels * 4, stride=1)
        )
        # Stage 3: 16x16 -> 8x8
        self.stage3 = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 8, stride=2),
            ResidualBlock(base_channels * 8, base_channels * 8, stride=1)
        )
        # Global Pooling + Classifier: 8x8 -> 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def forward(self, x):
        # Expects input normalized to [-1, 1] or [0, 1]
        out = self.init_conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

    @torch.no_grad()
    def predict_class(self, x):
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


# Alias for backward compatibility
CropClassifier = ReferenceClassifier

