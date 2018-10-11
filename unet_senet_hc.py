from collections import OrderedDict
from math import ceil

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from torchvision.models import ResNet
from torchvision.models.resnet import model_urls

from se_models import SEBasicBlock, SpatialChannelSEBlock
from senet import senet154, se_resnext50_32x4d


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels_next, up_size_next, out_channels_input, up_size_input):
        super().__init__()

        self.conv_next = ConvBnRelu(in_channels, out_channels_next)
        self.up_next = nn.Upsample(size=up_size_next, mode="bilinear", align_corners=False)
        self.se_next = SpatialChannelSEBlock(out_channels_next)

        self.conv_input = ConvBnRelu(in_channels, out_channels_input)
        self.up_input = nn.Upsample(size=up_size_input, mode="bilinear", align_corners=False)

    def forward(self, x):
        x_next = self.conv_next(x)
        x_next = self.up_next(x_next)
        x_next = self.se_next(x_next)

        x_up = self.conv_input(x)
        x_up = self.up_input(x_up)

        return x_next, x_up


class UNetSeNetHc(nn.Module):
    def __init__(self, num_classes, input_size, num_filters=32, dropout_2d=0.2, pretrained=False):
        super().__init__()
        self.dropout_2d = dropout_2d

        if True:
            self.encoder = senet154(pretrained="imagenet" if pretrained else None)
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
        else:
            self.encoder = se_resnext50_32x4d(pretrained="imagenet" if pretrained else None)
            bottom_channel_nr = 2048

            layer0_modules = [
                ('conv1', self.encoder.layer0.conv1),
                ('bn1', self.encoder.layer0.bn1),
                ('relu1', self.encoder.layer0.relu1),
            ]


        self.conv0 = nn.Sequential(OrderedDict(layer0_modules))
        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        dec_in_channels = [
            bottom_channel_nr,
            bottom_channel_nr // 2 + num_filters * 8,
            bottom_channel_nr // 4 + num_filters * 8,
            bottom_channel_nr // 8 + num_filters * 2
        ]

        dec_out_channels = [
            num_filters * 8,
            num_filters * 8,
            num_filters * 2,
            num_filters * 2 * 2
        ]

        dec_sizes = [
            ceil(input_size / 8),
            ceil(input_size / 4),
            ceil(input_size / 2),
            input_size
        ]

        hc_out_channels = dec_out_channels[3]
        final_in_channels = dec_out_channels[3]
        final_mid_channels = dec_out_channels[3]

        self.dec4 = DecoderBlock(dec_in_channels[0], dec_out_channels[0], dec_sizes[0], hc_out_channels, input_size)
        self.dec3 = DecoderBlock(dec_in_channels[1], dec_out_channels[1], dec_sizes[1], hc_out_channels, input_size)
        self.dec2 = DecoderBlock(dec_in_channels[2], dec_out_channels[2], dec_sizes[2], hc_out_channels, input_size)
        self.dec1 = DecoderBlock(dec_in_channels[3], dec_out_channels[3], dec_sizes[3], hc_out_channels, input_size)

        self.final = nn.Sequential(
            ConvBnRelu(final_in_channels, final_mid_channels // 2),
            nn.Conv2d(final_mid_channels // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)

        dec4, dec4_input = self.dec4(center)
        dec3, dec3_input = self.dec3(torch.cat([dec4, conv3], 1))
        dec2, dec2_input = self.dec2(torch.cat([dec3, conv2], 1))
        dec1, dec1_input = self.dec1(torch.cat([dec2, conv1], 1))

        out = dec1 + dec1_input + dec2_input + dec3_input + dec4_input

        return self.final(F.dropout2d(out, p=self.dropout_2d))
