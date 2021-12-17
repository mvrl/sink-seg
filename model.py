import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def count_trainable_parameters(model):
    """ to count trainable parameters """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class double_conv(nn.Module):
    """ implements two Conv-BatchNorm blocks, as used in UNet """

    def __init__(self, in_ch, out_ch, norm_type='batch'):
        super(double_conv, self).__init__()
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                  norm_layer(out_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                  norm_layer(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    """ Down sampling by MaxPool and double_conv """

    def __init__(self, in_ch, out_ch, norm_type='batch'):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch, norm_type))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    """ Upsampling by interpolation and double_conv """

    def __init__(self, in_ch, out_ch, norm_type='batch'):
        super(up, self).__init__()
        self.conv = double_conv(in_ch, out_ch, norm_type)

    def forward(self, x1, x2):
        x1 = nn.functional.interpolate(x1, scale_factor=2, mode='nearest')

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# The UNet Model
class Unet(nn.Module):
    """
    This is a standard UNet with minor tweaks. The changes are:
    - in upsamping, we use nearest neighbor interpolation to avpoid aliasing artifacts
    - we have a parameter, feature_reduction, that can be used to use fewer feature maps. When feature_reduction=2, the model will have half the feature maps than the original UNet.
    - We can replace BatchNorm with a norm, such as InstanceNorm.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 feature_reduction=1,
                 norm_type='batch'):
        super(Unet, self).__init__()

        self.inc = inconv(in_channels, int(64 / feature_reduction))
        self.down1 = down(int(64 / feature_reduction),
                          int(128 / feature_reduction), norm_type)
        self.down2 = down(int(128 / feature_reduction),
                          int(256 / feature_reduction), norm_type)
        self.down3 = down(int(256 / feature_reduction),
                          int(512 / feature_reduction), norm_type)
        self.down4 = down(int(512 / feature_reduction),
                          int(512 / feature_reduction), norm_type)
        self.up1 = up(int(1024 / feature_reduction),
                      int(256 / feature_reduction), norm_type)
        self.up2 = up(int(512 / feature_reduction),
                      int(128 / feature_reduction), norm_type)
        self.up3 = up(int(256 / feature_reduction), int(64 / feature_reduction),
                      norm_type)
        self.up4 = up(int(128 / feature_reduction), int(64 / feature_reduction),
                      norm_type)
        self.outc = outconv(int(64 / feature_reduction), out_channels)

        self.input_now = []  # just a place holder

    def set_input(self, shaded, dem, naip, dem_dxy, dem_dxy_pre):
        """
        Writing this general function to setup input, as specified in the config.py file.
        For different input modalities, only this function will need to be changed. train/eval loops will not be affected.
        """
        # dem only
        if cfg.data.input_type == 'dem':
            input_now = dem

        # shaded relief only
        elif cfg.data.input_type == 'shaded_relief':
            input_now = shaded

        # naip only
        elif cfg.data.input_type == 'naip':
            input_now = naip

        elif cfg.data.input_type == 'dem_derivative':
            input_now = dem_dxy

        elif cfg.data.input_type == 'dem_dxy_pre':
            input_now = dem_dxy_pre

        return input_now

    def forward(self, shaded, dem, naip, dem_dxy, dem_dxy_pre):
        # set appropriate input
        input_now = self.set_input(shaded, dem, naip, dem_dxy, dem_dxy_pre)

        x1 = self.inc(input_now)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
