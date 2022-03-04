from .resnet import resnet50_modify
import torch.nn as nn
import torch.nn.functional as F


def SimpleUpsampleHead(feature_channel, layer_channels):
    modules = []
    modules.append(nn.Conv2d(feature_channel, layer_channels[0], kernel_size=3, stride=1, padding=1, bias=False))
    for layer_index in range(len(layer_channels) - 1):
        modules.extend([
            nn.BatchNorm2d(layer_channels[layer_index]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(layer_channels[layer_index], layer_channels[layer_index + 1], kernel_size=2, stride=2, padding=0, bias=False),
        ])
    return nn.Sequential(*modules)

class Deformable_convolution(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Deformable_convolution, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                from assets.ops.dcn import DeformConv
                conv_op = DeformConv
                offset_channels = 18
            else:
                from assets.ops.dcn import ModulatedDeformConv
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes, deformable_groups * offset_channels,
                kernel_size=3,
                padding=1)
            self.conv2 = conv_op(
                planes, planes, kernel_size=3, padding=1, stride=stride,
                deformable_groups=deformable_groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4 , kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FPN_Deform(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.upsample1 = nn.ConvTranspose2d()

    def upsample2(self,x,y):
        _, _, h, w, = y.size()
        return F.interpolate(x, size = (h, w), mode= 'bilinear')

    def forward(self, pyramid_features):

class Resnet50_FPN_Deform(nn.Module):
    def __init__(self):
        super(Resnet50_FPN_Deform, self).__init__()
        self.bottom_up = resnet50_modify()
        self.fpn_deform = FPN_Deform()
    #top_down = FPNTopDown([512, 512, 256, 128], 128)
    #feature_pyramid = FeaturePyramid(bottom_up, top_down)

    def forward(self,feature):
        pyramid_features = self.bottom_up(feature)
        feature_new = self.fpn_deform(pyramid_features)
        return feature_new
