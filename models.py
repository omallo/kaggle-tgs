from math import ceil

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from torchvision.models import ResNet
from torchvision.models.resnet import model_urls

from se_models import SEBasicBlock, SpatialChannelSEBlock


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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        self.upsample = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            nn.Upsample(size=size, mode="bilinear", align_corners=False),
            SpatialChannelSEBlock(out_channels)
        )

    def forward(self, x):
        return self.upsample(x)


class UNetResNet(nn.Module):
    def __init__(self, num_classes, input_size, num_filters=32, dropout_2d=0.2, pretrained=False, output_classification=False):
        super().__init__()
        self.dropout_2d = dropout_2d
        self.output_classification = output_classification

        self.encoder = ResNet(SEBasicBlock, [3, 4, 6, 3])
        if pretrained:
            self.encoder.load_state_dict(model_zoo.load_url(model_urls["resnet34"]), strict=False)
        bottom_channel_nr = 512

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

        self.input_adjust = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)

        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.classifier = nn.Sequential(
            self.encoder.avgpool,
            nn.Linear(bottom_channel_nr, num_classes)
        )

        self.dec4 = DecoderBlock(dec_in_channels[0], dec_out_channels[0], size=dec_sizes[0])
        self.dec3 = DecoderBlock(dec_in_channels[1], dec_out_channels[1], size=dec_sizes[1])
        self.dec2 = DecoderBlock(dec_in_channels[2], dec_out_channels[2], size=dec_sizes[2])
        self.dec1 = DecoderBlock(dec_in_channels[3], dec_out_channels[3], size=dec_sizes[3])

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

        if self.output_classification:
            return self.final(dec1), self.classifier(center)
        else:
            return self.final(dec1)
