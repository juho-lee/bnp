import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3,
                stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.pool(self.relu(x))
        return x

class CNNBackBone(nn.Module):
    def __init__(self, in_channels, hid_channels=64):
        super().__init__()
        self.block1 = ConvBlock(in_channels, hid_channels)
        self.block2 = ConvBlock(hid_channels, hid_channels)
        self.block3 = ConvBlock(hid_channels, hid_channels)
        self.block4 = ConvBlock(hid_channels, hid_channels)

    def forward(self, x):
        shape = x.shape
        batch_shape = shape[:-3]
        C, H, W = shape[-3:]
        x = x.view(-1, C, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.mean((-2, -1))
        x = x.view(batch_shape + (-1,))
        return x
