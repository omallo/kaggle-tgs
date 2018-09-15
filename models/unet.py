import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_depth, out_depth, base_channels):
        super().__init__()

        self.down1 = UNetDown(in_depth, 1 * base_channels)
        self.down2 = UNetDown(1 * base_channels, 2 * base_channels)
        self.down3 = UNetDown(2 * base_channels, 4 * base_channels)
        self.down4 = UNetDown(4 * base_channels, 8 * base_channels)

        self.middle = UNetMiddle(8 * base_channels, 16 * base_channels)

        self.up4 = UNetUp(16 * base_channels, 8 * base_channels)
        self.up3 = UNetUp(8 * base_channels, 4 * base_channels)
        self.up2 = UNetUp(4 * base_channels, 2 * base_channels)
        self.up1 = UNetUp(2 * base_channels, 1 * base_channels)

        self.output = UNetOutput(1 * base_channels, out_depth)

    def forward(self, x):
        x_skip1, x = self.down1(x)
        x_skip2, x = self.down2(x)
        x_skip3, x = self.down3(x)
        x_skip4, x = self.down4(x)

        x = self.middle(x)

        x = self.up4(x, x_skip4)
        x = self.up3(x, x_skip3)
        x = self.up2(x, x_skip2)
        x = self.up1(x, x_skip1)

        x = self.output(x)

        return x


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = UNetDoubleConv(in_channels, out_channels)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        conv_out = self.conv(x)
        down_out = self.down(conv_out)
        return conv_out, down_out


class UNetMiddle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.delegate = UNetDoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.delegate(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = with_he_normal_weights(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
        self.conv = UNetDoubleConv(in_channels, out_channels)

    def forward(self, x, x_skip):
        up_out = self.up(x)
        conv_out = self.conv(torch.cat([up_out, x_skip], dim=1))
        return conv_out


class UNetDoubleConv(nn.Module):
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
