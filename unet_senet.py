from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from se_models import SpatialChannelSEBlock
from senet import se_resnet50, senet154


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
        super(DecoderBlockV2, self).__init__()
        self.upsample = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            nn.Upsample(size=size, mode="bilinear", align_corners=False),
            SpatialChannelSEBlock(out_channels)
        )

    def forward(self, x):
        return self.upsample(x)


class UNetSeNet(nn.Module):
    def __init__(self, backbone, num_classes, input_size, num_filters=32, dropout_2d=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if backbone == "senet154":
            self.encoder = senet154()
            bottom_channel_nr = 2048

            layer0_modules = [
                ('conv1', self.encoder.layer0.conv1),
                ('bn1', self.encoder.layer0.bn1),
                ('relu1', self.encoder.layer0.relu1),
                ('conv2', self.encoder.layer0.conv2),
                ('bn2', self.encoder.layer0.bn2),
                ('relu2', self.encoder.layer0.relu2),
                ('conv3', self.encoder.layer0.conv3),
                ('bn3', self.encoder.layer0.bn3),
                ('relu3', self.encoder.layer0.relu3),
            ]
        elif backbone == "se_resnet50":
            self.encoder = se_resnet50()
            bottom_channel_nr = 2048

            layer0_modules = [
                ('conv1', self.encoder.layer0.conv1),
                ('bn1', self.encoder.layer0.bn1),
                ('relu1', self.encoder.layer0.relu1),
            ]
        else:
            raise Exception("Unsupported backbone type: '{}".format(backbone))

        self.input_adjust = nn.Sequential(OrderedDict(layer0_modules))

        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.dec4 = DecoderBlockV2(bottom_channel_nr, num_filters * 8, size=int(np.ceil(input_size / 8)))
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8,
                                   size=int(np.ceil(input_size / 4)))
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 2,
                                   size=int(np.ceil(input_size / 2)))
        self.dec1 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, size=input_size)

        self.final = nn.Conv2d(num_filters * 2 * 2, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)
        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)
        return self.final(dec1)
