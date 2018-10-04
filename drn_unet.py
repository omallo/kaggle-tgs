import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from drn import drn_d_38

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        self.delegate = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            nn.Upsample(size=size, mode="bilinear", align_corners=False)
        )

    def forward(self, x):
        return self.delegate(x)


class DrnUNet(nn.Module):
    def __init__(self, num_classes, input_size, dropout_2d=0.2, pretrained=False):
        super().__init__()
        self.dropout_2d = dropout_2d

        channels = (16, 32, 64, 128, 256, 512, 512, 512)

        self.encoder = drn_d_38(pretrained=pretrained, num_classes=num_classes, channels=channels)

        self.conv0 = self.encoder.layer0
        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4
        self.conv5 = self.encoder.layer5
        self.conv6 = self.encoder.layer6
        self.conv7 = self.encoder.layer7
        self.conv8 = self.encoder.layer8

        self.dec8 = DecoderBlockV2(channels[7], channels[6], size=int(np.ceil(input_size / 8)))
        self.dec7 = DecoderBlockV2(channels[6], channels[5], size=int(np.ceil(input_size / 8)))
        self.dec6 = DecoderBlockV2(channels[5], channels[4], size=int(np.ceil(input_size / 8)))
        self.dec5 = DecoderBlockV2(channels[4], channels[3], size=int(np.ceil(input_size / 8)))
        self.dec4 = DecoderBlockV2(channels[3], channels[2], size=int(np.ceil(input_size / 4)))
        self.dec3 = DecoderBlockV2(channels[2], channels[1], size=int(np.ceil(input_size / 2)))
        self.dec2 = DecoderBlockV2(channels[1], channels[0], size=int(np.ceil(input_size / 1)))
        self.dec1 = DecoderBlockV2(channels[0], channels[0], size=int(np.ceil(input_size / 1)))

        self.final = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        center = self.conv8(conv7)

        dec8 = self.dec8(center)

        dec7 = self.dec7(torch.cat([dec8, conv7], 1))
        dec6 = self.dec6(torch.cat([dec7, conv6], 1))
        dec5 = self.dec5(torch.cat([dec6, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.final(F.dropout2d(dec1, p=self.dropout_2d))
