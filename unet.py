import torch
import torch.nn as nn
import torchvision


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        encoder = torchvision.models.resnet34(pretrained=True)

        self.down1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # 64
        self.down2 = nn.Sequential(encoder.maxpool, encoder.layer1)  # 64
        self.down3 = encoder.layer2  # 128
        self.down4 = encoder.layer3  # 256
        self.down5 = encoder.layer4  # 512

        self.middle = UNetMiddle(512, 1024)

        self.up5 = UNetUp(1024, 512)
        self.up4 = UNetUp(512, 256)
        self.up3 = UNetUp(256, 128)
        self.up2 = UNetUp(128, 64)
        self.up1 = UNetUp(64, 64, conv_in_channels=128)
        self.up0 = UNetUp(64, 3, conv_in_channels=6)

        self.output = UNetOutput(3, 1)

    def forward(self, x):
        x_skip0 = x
        x_skip1 = self.down1(x_skip0)
        x_skip2 = self.down2(x_skip1)
        x_skip3 = self.down3(x_skip2)
        x_skip4 = self.down4(x_skip3)
        x_skip5 = self.down5(x_skip4)

        x = self.middle(x_skip5)

        x = self.up5(x, x_skip5)
        x = self.up4(x, x_skip4)
        x = self.up3(x, x_skip3)
        x = self.up2(x, x_skip2)
        x = self.up1(x, x_skip1)
        x = self.up0(x, x_skip0)

        x = self.output(x)

        return x


class UNetMiddle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.delegate = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            UNetConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.delegate(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, conv_in_channels=None):
        super().__init__()
        self.up = with_he_normal_weights(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
        self.conv = UNetConvBlock(conv_in_channels or in_channels, out_channels)

    def forward(self, x, x_skip):
        up_out = self.up(x)
        print(x.shape)
        print(x_skip.shape)
        print(up_out.shape)
        print("foo")
        v = torch.cat([up_out, x_skip], dim=1)
        print(v.shape)
        print("bar")
        conv_out = self.conv(v)
        return conv_out


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.delegate = nn.Sequential(
            UNetConv(in_channels, out_channels),
            UNetConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.delegate(x)


class UNetConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.delegate = nn.Sequential(
            with_he_normal_weights(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.delegate(x)


class UNetOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.delegate = with_he_normal_weights(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        return self.delegate(x)


def with_he_normal_weights(layer):
    nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
    return layer
