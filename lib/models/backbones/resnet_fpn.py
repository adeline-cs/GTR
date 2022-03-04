from .resnet import resnet18, resnet34, resnet50,resnet101,resnet152,deformable_resnet50
from .resnet import resnet18_modify, resnet34_modify,resnet50_modify,resnet101_modify,resnet152_modify,deformable_resnet50_modify
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class FeaturePyramid(nn.Module):
    def __init__(self, bottom_up, top_down):
        nn.Module.__init__(self)
        self.bottom_up = bottom_up
        self.top_down = top_down

    def forward(self, feature):
        pyramid_features = self.bottom_up(feature)
        feature = self.top_down(pyramid_features[::-1])
        return feature

class FPNTopDown(nn.Module):
    def __init__(self, pyramid_channels, feature_channel):
        nn.Module.__init__(self)
        self.reduction_layers = nn.ModuleList()
        for pyramid_channel in pyramid_channels:
            reduction_layer = nn.Conv2d(pyramid_channel, feature_channel, kernel_size=1, stride=1, padding=0, bias=False)
            self.reduction_layers.append(reduction_layer)
        self.merge_layer = nn.Conv2d(feature_channel, feature_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def upsample_add(self,x,y):
        _, _, h, w, = y.size()
        return F.interpolate(x, size = (h, w), mode= 'bilinear') + y

    def forward(self, pyramid_features):
        feature =  None
        for pyramid_feature, reduction_layer in zip(pyramid_features, self.reduction_layers):
            pyramid_feature = reduction_layer(pyramid_feature)
            if feature is None:
                feature = pyramid_feature
            else:
                feature = self.upsample_add(feature, pyramid_feature)
        feature = self.merge_layer(feature)
        return feature

def Resnet18FPN(resnet_pretrained=True):
    bottom_up = resnet18(pretrained=resnet_pretrained)
    top_down = FPNTopDown([512, 256, 128, 64], 256)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Resnet34FPN(resnet_pretrained=True):
    bottom_up = resnet34(pretrained=resnet_pretrained)
    top_down = FPNTopDown([512, 256, 128, 64], 256) # original is [2048, 1024, 512, 256], 256
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Resnet50FPN(resnet_pretrained=True):
    bottom_up = resnet50(pretrained=resnet_pretrained)
    top_down = FPNTopDown([512, 256, 128, 64], 256)   # original is [2048, 1024, 512, 256], 256
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Resnet101FPN(resnet_pretrained=True):
    bottom_up = resnet101(pretrained=resnet_pretrained)
    top_down = FPNTopDown([512, 256, 128, 64], 256)   # original is [2048, 1024, 512, 256], 256
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Resnet152FPN(resnet_pretrained=True):
    bottom_up = resnet152(pretrained=resnet_pretrained)
    top_down = FPNTopDown([512, 256, 128, 64], 256)    # original is [2048, 1024, 512, 256], 256
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Deformable_Resnet50FPN(resnet_pretrained=True):
    bottom_up = resnet50(pretrained=resnet_pretrained)
    top_down = FPNTopDown([512, 256, 128, 64], 256)     # original is [2048, 1024, 512, 256], 256
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Resnet18FPN_v1():
    bottom_up = resnet18_modify()
    top_down = FPNTopDown([512, 512, 256, 128], 128)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Resnet34FPN_v1():
    bottom_up = resnet34_modify()
    top_down = FPNTopDown([512, 512, 256, 128], 128)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Resnet50FPN_v1():
    bottom_up = resnet50_modify()
    top_down = FPNTopDown([512, 512, 256, 128], 128)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Resnet101FPN_v1():
    bottom_up = resnet101_modify()
    top_down = FPNTopDown([512, 512, 256, 128], 128)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Resnet152FPN_v1():
    bottom_up = resnet152_modify()
    top_down = FPNTopDown([512, 512, 256, 128], 128)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def Deformable_Resnet50FPN_v1():
    bottom_up = deformable_resnet50_modify()
    top_down = FPNTopDown([512, 512, 256, 128], 128)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid


def Resnet50_FPN_Deform():
    bottom_up = resnet50_modify()
    top_down = FPNTopDown([512, 512, 256, 128], 128)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid

def ResnetFPN(res_type):
    if res_type == "res18":
        model = Resnet18FPN()
    if res_type == "res34":
        model = Resnet34FPN()
    if res_type == "res50":
        model = Resnet50FPN()
    if res_type == "res101":
        model = Resnet101FPN()
    if res_type == "res152":
        model = Resnet152FPN()
    if res_type == "deform-res50":
        model = Deformable_Resnet50FPN()
    if res_type == "res18-v1":
        model = Resnet18FPN_v1()
    if res_type == "res34-v1":
        model = Resnet34FPN_v1()
    if res_type == "res50-v1":
        model = Resnet50FPN_v1()
    if res_type == "res101-v1":
        model = Resnet101FPN_v1()
    if res_type == "res152-v1":
        model = Resnet152FPN_v1()
    if res_type == "deform-res50-v1":
        model = Deformable_Resnet50FPN_v1()