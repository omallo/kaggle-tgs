import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from torchvision.models import ResNet
from torchvision.models.resnet import model_urls

from se_models import SEBasicBlock, SEBottleneck
from utils import with_he_normal_weights


def create_model(pretrained):
    return UNetResNet(34, 1, num_filters=32, dropout_2d=0.5, pretrained=pretrained, is_deconv=False)


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            with_he_normal_weights(nn.Conv2d(in_channels, out_channels, 3, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.is_deconv = is_deconv

        self.deconv = nn.Sequential(
            ConvBnRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

    def forward(self, x):
        if self.is_deconv:
            x = self.deconv(x)
        else:
            x = self.upsample(x)
        return x


class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 50, 101 or 152) encoder.
    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.
    """

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            # self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            self.encoder = ResNet(SEBasicBlock, [3, 4, 6, 3])
            if pretrained:
                self.encoder.load_state_dict(model_zoo.load_url(model_urls["resnet34"]), strict=False)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            # self.encoder = torchvision.models.resnet50(pretrained=pretrained)
            self.encoder = ResNet(SEBottleneck, [3, 4, 6, 3])
            if pretrained:
                self.encoder.load_state_dict(model_zoo.load_url(model_urls["resnet50"]), strict=False)
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            # self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            self.encoder = ResNet(SEBottleneck, [3, 4, 23, 3])
            if pretrained:
                self.encoder.load_state_dict(model_zoo.load_url(model_urls["resnet101"]), strict=False)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            # self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            self.encoder = ResNet(SEBottleneck, [3, 8, 36, 3])
            if pretrained:
                self.encoder.load_state_dict(model_zoo.load_url(model_urls["resnet152"]), strict=False)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 50, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.input_adjust = nn.Sequential(self.encoder.conv1,
                                          self.encoder.bn1,
                                          self.encoder.relu)

        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.dec4 = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
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
