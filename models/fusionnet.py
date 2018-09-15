import torch.nn as nn


class FusionNet(nn.Module):
    def __init__(self, in_depth, out_depth, base_channels):
        super().__init__()

        self.down1 = FusionNetDown(in_depth, 1 * base_channels)
        self.down2 = FusionNetDown(1 * base_channels, 2 * base_channels)
        self.down3 = FusionNetDown(2 * base_channels, 4 * base_channels)
        self.down4 = FusionNetDown(4 * base_channels, 8 * base_channels)
        self.down5 = FusionNetDown(8 * base_channels, 16 * base_channels)

        self.middle = FusionNetMiddle(16 * base_channels, 32 * base_channels)

        self.up5 = FusionNetUp(32 * base_channels, 16 * base_channels)
        self.up4 = FusionNetUp(16 * base_channels, 8 * base_channels)
        self.up3 = FusionNetUp(8 * base_channels, 4 * base_channels)
        self.up2 = FusionNetUp(4 * base_channels, 2 * base_channels)
        self.up1 = FusionNetUp(2 * base_channels, 1 * base_channels)

        self.output = FusionNetOutput(1 * base_channels, out_depth)

    def forward(self, x):
        x_skip1, x = self.down1(x)
        x_skip2, x = self.down2(x)
        x_skip3, x = self.down3(x)
        x_skip4, x = self.down4(x)
        x_skip5, x = self.down5(x)

        x = self.middle(x)

        x = self.up5(x, x_skip5)
        x = self.up4(x, x_skip4)
        x = self.up3(x, x_skip3)
        x = self.up2(x, x_skip2)
        x = self.up1(x, x_skip1)

        x = self.output(x)

        return x


class FusionNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = FusionNetConvBlock(in_channels, out_channels)
        self.down = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        conv_out = self.conv(x)
        down_out = self.down(conv_out)
        return conv_out, down_out


class FusionNetMiddle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.delegate = FusionNetConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.delegate(x)


class FusionNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = with_he_normal_weights(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
        self.conv = FusionNetConvBlock(out_channels, out_channels)

    def forward(self, x, x_skip):
        up_out = self.up(x)
        conv_out = self.conv(up_out + x_skip)
        return conv_out


class FusionNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.delegate = nn.Sequential(
            FusionNetConv(in_channels, out_channels),
            FusionNetResidualBlock(out_channels),
            FusionNetConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.delegate(x)


class FusionNetConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.delegate = nn.Sequential(
            with_he_normal_weights(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.delegate(x)


class FusionNetResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.delegate = nn.Sequential(
            FusionNetConv(channels, channels),
            FusionNetConv(channels, channels),
            FusionNetConv(channels, channels)
        )

    def forward(self, x):
        return x + self.delegate(x)


class FusionNetOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.delegate = with_he_normal_weights(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        return self.delegate(x)


def with_he_normal_weights(layer):
    nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
    return layer
