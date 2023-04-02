import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from config import cfg

"""
    FuseNet implementation
    FuseNet: https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf
    Code adopted from: https://github.com/MehmetAygun/fusenet-pytorch
    """

def set_parameter_requires_grad(model, feature_extracting):
    # we do not need gradients for feature extraction
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


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


class triple_conv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type='batch'):
        super(triple_conv, self).__init__()
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                  norm_layer(out_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                  norm_layer(out_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                  norm_layer(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

# FuseNet


class FuseNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 feature_reduction=1,
                 norm_type='batch'):

        super(FuseNet, self).__init__()

        # use the pre_trained vgg16 model
        vgg16_naip = torchvision.models.vgg16_bn(weights='DEFAULT')
        avg = torch.mean(vgg16_naip.features[0].weight.data, dim=1)
        avg = avg.unsqueeze(1)
        conv11d = nn.Conv2d(4, 64, kernel_size=(3, 3),
                            stride=(1, 1), padding=(1, 1))
        conv11d.weight.data = torch.concat([avg]*4, axis=1)
        vgg16_naip.features[0] = conv11d

        vgg16_dem = torchvision.models.vgg16_bn(weights='DEFAULT')
        avg = torch.mean(vgg16_dem.features[0].weight.data, dim=1)
        avg = avg.unsqueeze(1)
        depth_cn = in_channels-4
        conv11d = nn.Conv2d(depth_cn, 64, kernel_size=(3, 3),
                            stride=(1, 1), padding=(1, 1))
        if depth_cn == 1:
            conv11d.weight.data = avg
        elif depth_cn == 2:
            conv11d.weight.data = torch.concat([avg]*2, axis=1) 

        vgg16_dem.features[0] = conv11d

        set_parameter_requires_grad(vgg16_naip, feature_extracting=True)
        set_parameter_requires_grad(vgg16_dem, feature_extracting=True)

        if cfg.model.pre_trained:
            print('Using pre-trained vgg16 model')
            self.dem_down1 = vgg16_dem.features[0:6]
            self.naip_down1 = vgg16_naip.features[0:6]

            self.dem_down2 = vgg16_dem.features[7:13]
            self.naip_down2 = vgg16_naip.features[7:13]

            self.dem_down3 = vgg16_dem.features[14:23]
            self.naip_down3 = vgg16_naip.features[14:23]

            self.dem_down4 = vgg16_dem.features[24:33]
            self.naip_down4 = vgg16_naip.features[24:33]

            self.dem_down5 = vgg16_dem.features[34:43]
            self.naip_down5 = vgg16_naip.features[34:43]
        else:
            print('Not using pre-trained vgg16 model')
            self.dem_down1 = double_conv(
                in_channels-4, int(64 / feature_reduction))
            self.naip_down1 = double_conv(4, int(64 / feature_reduction))

            self.dem_down2 = double_conv(
                int(64 / feature_reduction), int(128 / feature_reduction))
            self.naip_down2 = double_conv(
                int(64 / feature_reduction), int(128 / feature_reduction))

            self.dem_down3 = double_conv(
                int(128 / feature_reduction), int(256 / feature_reduction))
            self.naip_down3 = double_conv(
                int(128 / feature_reduction), int(256 / feature_reduction))

            self.dem_down4 = triple_conv(
                int(256 / feature_reduction), int(512 / feature_reduction))
            self.naip_down4 = triple_conv(
                int(256 / feature_reduction), int(512 / feature_reduction))

            self.dem_down5 = triple_conv(
                int(512 / feature_reduction), int(512 / feature_reduction))
            self.naip_down5 = triple_conv(
                int(512 / feature_reduction), int(512 / feature_reduction))

        self.up1 = triple_conv(int(512 / feature_reduction),
                               int(512 / feature_reduction))
        self.up2 = triple_conv(int(512 / feature_reduction),
                               int(256 / feature_reduction))
        self.up3 = double_conv(int(256 / feature_reduction),
                               int(128 / feature_reduction))
        self.up4 = double_conv(int(128 / feature_reduction),
                               int(64 / feature_reduction))
        self.up5 = outconv(int(64 / feature_reduction), out_channels)

        self.input_now = []  # just a place holder

    def set_input(self, shaded, dem, naip, dem_dxy, dem_dxy_pre):
        """
        Writing this general function to setup input, as specified in the config.py file.
        For different input modalities, only this function will need to be changed. train/eval loops will not be affected.
        """

        # new added
        if cfg.data.input_type == 'shaded_relief_naip':
            input_now = [shaded, naip]

        elif cfg.data.input_type == 'dem_naip':
            input_now = [dem, naip]

        elif cfg.data.input_type == 'dem_derivative_naip':
            input_now = [dem_dxy, naip]

        elif cfg.data.input_type == 'dem_dxy_pre_naip':
            input_now = [dem_dxy_pre, naip]

        return input_now

    def forward(self, shaded, dem, naip, dem_dxy, dem_dxy_pre):

        # set appropriate input
        input_now = self.set_input(shaded, dem, naip, dem_dxy, dem_dxy_pre)

        if input_now[0].shape[2] != input_now[1].shape[2]:
            diff = abs(input_now[0].shape[2]-input_now[1].shape[2])
            pad = nn.ZeroPad2d((0, 0, 0, diff))
            input_now[0] = pad(input_now[0])
        #print(input_now[0].shape)
        x1_dem = self.dem_down1(input_now[0])
        x1_naip = self.naip_down1(input_now[1])
        x1 = (x1_dem + x1_naip) / 2
        x1_dem = F.max_pool2d(x1_dem, kernel_size=2, stride=2)
        x1, indices1 = F.max_pool2d(
            x1, kernel_size=2, stride=2, return_indices=True)
        #print(x1.shape, indices1.shape)

        x2_dem = self.dem_down2(x1_dem)
        x2_naip = self.naip_down2(x1)
        x2 = (x2_dem + x2_naip) / 2
        x2_dem = F.max_pool2d(x2_dem, kernel_size=2, stride=2)
        x2, indices2 = F.max_pool2d(
            x2, kernel_size=2, stride=2, return_indices=True)
        #print(x2.shape, indices2.shape)

        x3_dem = self.dem_down3(x2_dem)
        x3_naip = self.naip_down3(x2)
        x3 = (x3_dem + x3_naip) / 2
        x3_dem = F.dropout(F.max_pool2d(
            x3_dem, kernel_size=2, stride=2), p=0.2)
        x3, indices3 = F.max_pool2d(
            x3, kernel_size=2, stride=2, return_indices=True)
        
        if cfg.data.mode == 'train':
            x3 = F.dropout(x3, p=0.4)
        #print(x3.shape, indices3.shape)

        x4_dem = self.dem_down4(x3_dem)
        x4_naip = self.naip_down4(x3)
        x4 = (x4_dem + x4_naip) / 2
        x4_dem = F.dropout(F.max_pool2d(
            x4_dem, kernel_size=2, stride=2), p=0.2)
        x4, indices4 = F.max_pool2d(
            x4, kernel_size=2, stride=2, return_indices=True)
        
        if cfg.data.mode == 'train':
            x4 = F.dropout(x4, p=0.4)
        #print(x4.shape, indices4.shape)

        x5_dem = self.dem_down5(x4_dem)
        x5_naip = self.naip_down5(x4)
        x5 = (x5_dem + x5_naip) / 2
        x5, indices5 = F.max_pool2d(
            x5, kernel_size=2, stride=2, return_indices=True)
        
        if cfg.data.mode == 'train':
            x5 = F.dropout(x5, p=0.4)
        #print(x5.shape, indices5.shape)

        x = F.max_unpool2d(x5, indices5, kernel_size=2,
                           stride=2, output_size=x5_naip.shape)
        x = self.up1(x)
        if cfg.data.mode == 'train':
            x = F.dropout(x, p=0.4)

        x = F.max_unpool2d(x, indices4, kernel_size=2,
                           stride=2, output_size=x4_naip.shape)
        x = self.up2(x)
        if cfg.data.mode == 'train':
            x = F.dropout(x, p=0.4)

        x = F.max_unpool2d(x, indices3, kernel_size=2,
                           stride=2, output_size=x3_naip.shape)
        x = self.up3(x)
        if cfg.data.mode == 'train':
            x = F.dropout(x, p=0.4)

        x = F.max_unpool2d(x, indices2, kernel_size=2,
                           stride=2, output_size=x2_naip.shape)
        x = self.up4(x)

        x = F.max_unpool2d(x, indices1, kernel_size=3,
                           stride=2, output_size=x1_naip.shape)
        x = self.up5(x)
        #print(x.shape)

        return x
