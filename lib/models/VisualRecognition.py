import os
import sys
import time
import random
import string
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from einops import reduce, rearrange, repeat
import datetime
import os
import numpy as np

from backbones import resnet_ppm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
FeatureOrdering layer is a preprocess for GTR, and features are based on visual recognition module
'''
# Current version only supports input whose size is a power of 2, such as 32, 64, 128 etc.
# You can adapt it to any input size by changing the padding or stride.
# just a simple encoder-decoder framework
class FeatureOrdering(nn.Module):
    def __init__(self, scales, maxT, depth, num_channels):
        super(FeatureOrdering, self).__init__()
        # cascade multiscale features
        fpn = []
        for i in range(1, len(scales)):
            assert not (scales[i - 1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            assert not (scales[i - 1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            ksize = [3, 3, 5]  # if downsampling ratio >= 3, the kernel size is 5, else 3
            r_h, r_w = int(scales[i - 1][1] / scales[i][1]), int(scales[i - 1][2] / scales[i][2])
            ksize_h = 1 if scales[i - 1][1] == 1 else ksize[r_h - 1]
            ksize_w = 1 if scales[i - 1][2] == 1 else ksize[r_w - 1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i - 1][0], scales[i][0],
                                               (ksize_h, ksize_w),
                                               (r_h, r_w),
                                               (int((ksize_h - 1) / 2), int((ksize_w - 1) / 2))),
                                     nn.BatchNorm2d(scales[i][0]),
                                     nn.ReLU(True)))
        self.fpn = nn.Sequential(*fpn)
        # convolutional alignment
        # convs
        assert depth % 2 == 0, 'the depth of CAM must be a even number.'
        in_shape = scales[-1]
        strides = []
        conv_ksizes = []
        deconv_ksizes = []
        h, w = in_shape[1], in_shape[2]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2 ** (depth / 2 - i) <= h else [1]
            stride = stride + [2] if 2 ** (depth / 2 - i) <= w else stride + [1]
            strides.append(stride)
            conv_ksizes.append([3, 3])
            deconv_ksizes.append([_ ** 2 for _ in stride])

        convs = [nn.Sequential(nn.Conv2d(in_shape[0], num_channels,
                                         tuple(conv_ksizes[0]),
                                         tuple(strides[0]),
                                         (int((conv_ksizes[0][0] - 1) / 2), int((conv_ksizes[0][1] - 1) / 2))),
                               nn.BatchNorm2d(num_channels),
                               nn.ReLU(True))]
        for i in range(1, int(depth / 2)):
            convs.append(nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                                 tuple(conv_ksizes[i]),
                                                 tuple(strides[i]),
                                                 (int((conv_ksizes[i][0] - 1) / 2), int((conv_ksizes[i][1] - 1) / 2))),
                                       nn.BatchNorm2d(num_channels),
                                       nn.ReLU(True)))
        self.convs = nn.Sequential(*convs)
        # deconvs
        deconvs = []
        for i in range(1, int(depth / 2)):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                            tuple(deconv_ksizes[int(depth / 2) - i]),
                                                            tuple(strides[int(depth / 2) - i]),
                                                            (int(deconv_ksizes[int(depth / 2) - i][0] / 4.),
                                                             int(deconv_ksizes[int(depth / 2) - i][1] / 4.))),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)))
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, maxT,
                                                        tuple(deconv_ksizes[0]),
                                                        tuple(strides[0]),
                                                        (int(deconv_ksizes[0][0] / 4.), int(deconv_ksizes[0][1] / 4.))),
                                     nn.Sigmoid()))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, basic_feature):
        x = input[0]
        for i in range(0, len(self.fpn)):
            x = self.fpn[i](x) + input[i + 1]
        conv_feats = []
        for i in range(0, len(self.convs)):
            x = self.convs[i](x)
            conv_feats.append(x)
        for i in range(0, len(self.deconvs) - 1):
            x = self.deconvs[i](x)
            x = x + conv_feats[len(conv_feats) - 2 - i]
        x = self.deconvs[-1](x)
        return x

class FeatureOrdering_transposed(nn.Module):
    # In this version, the input channel is reduced to 1-D with sigmoid activation.
    # We found that this leads to faster convergence for 1-D recognition.
    def __init__(self, scales, maxT, depth, num_channels):
        super(FeatureOrdering_transposed, self).__init__()
        # cascade multiscale features
        fpn = []
        for i in range(1, len(scales)):
            assert not (scales[i - 1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            assert not (scales[i - 1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            ksize = [3, 3, 5]
            r_h, r_w = scales[i - 1][1] / scales[i][1], scales[i - 1][2] / scales[i][2]
            ksize_h = 1 if scales[i - 1][1] == 1 else ksize[r_h - 1]
            ksize_w = 1 if scales[i - 1][2] == 1 else ksize[r_w - 1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i - 1][0], scales[i][0],
                                               (ksize_h, ksize_w),
                                               (r_h, r_w),
                                               ((ksize_h - 1) / 2, (ksize_w - 1) / 2)),
                                     nn.BatchNorm2d(scales[i][0]),
                                     nn.ReLU(True)))
        fpn.append(nn.Sequential(nn.Conv2d(scales[i][0], 1,
                                           (1, ksize_w),
                                           (1, r_w),
                                           (0, (ksize_w - 1) / 2)),
                                 nn.Sigmoid()))
        self.fpn = nn.Sequential(*fpn)
        # convolutional alignment
        # deconvs
        in_shape = scales[-1]
        deconvs = []
        ksize_h = 1 if in_shape[1] == 1 else 4
        for i in range(1, depth / 2):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                            (ksize_h, 4),
                                                            (r_h, 2),
                                                            (int(ksize_h / 4.), 1)),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)))
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, maxT,
                                                        (ksize_h, 4),
                                                        (r_h, 2),
                                                        (int(ksize_h / 4.), 1)),
                                     nn.Sigmoid()))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, input):
        x = input[0]
        for i in range(0, len(self.fpn) - 1):
            x = self.fpn[i](x) + input[i + 1]
        # Reducing the input to 1-D form
        x = self.fpn[-1](x)
        # Transpose B-C-H-W to B-W-C-H
        x = x.permute(0, 3, 1, 2).contiguous()

        for i in range(0, len(self.deconvs)):
            x = self.deconvs[i](x)
        return x



class PositionAwareLayer(nn.Module):

    def __init__(self, dim_model, rnn_layers=2):
        super().__init__()

        self.dim_model = dim_model

        self.rnn = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers,
            batch_first=True)

        self.mixer = nn.Sequential(
            nn.Conv2d(
                dim_model, dim_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(
                dim_model, dim_model, kernel_size=3, stride=1, padding=1))

    def forward(self, img_feature):
        n, c, h, w = img_feature.size()

        rnn_input = img_feature.permute(0, 2, 3, 1).contiguous()
        rnn_input = rnn_input.view(n * h, w, c)
        rnn_output, _ = self.rnn(rnn_input)
        rnn_output = rnn_output.view(n, h, w, c)
        rnn_output = rnn_output.permute(0, 3, 1, 2).contiguous()

        out = self.mixer(rnn_output)

        return out

'''
using for character segmentation position attention
'''
class PositionAttention(nn.Module):

    def __init__(self, input_dim):
        super(PositionAttention, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(self.input_dim, self.input_dim, 3, stride=1)
        self.conv2 = nn.Conv2d(self.input_dim, self.input_dim, 1, stride=1 )
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        attend_feature = x * x2 + x
        return attend_feature
'''
Character segmentation of visual recognition module
'''
class Character_segmentation(nn.Module):

    def __init__(self, char_voc, input_dim):
        super(Character_segmentation, self).__init__()
        self.voc = char_voc
        self.input_dim = input_dim
        self.ppm = resnet_ppm.PPMDeepsup()
        self.attention = PositionAttention(self.input_dim)
        self.conv1 = nn.Conv2d(4 * self.input_dim ,self.voc+1 , 3, stride=1)
    def forward(self, basic_feature):
        ppm_feature =self.ppm(basic_feature)
        attention_feature = self.attention(ppm_feature)
        char_seg_map = self.conv1(attention_feature)
        return char_seg_map

'''The visual recognition model and ordering feature for GTR module'''
class VisualRecognitionModule(nn.Module):
    def __init__(self, char_voc, input_dim, scales, maxT, depth, num_channels):
        super(VisualRecognitionModule, self).__init__()
        self.voc = char_voc
        self.input_dim = input_dim
        self.scales = scales
        self.maxT  = maxT
        self.depth = depth
        self.num_channels = num_channels
        self.Character_segmentation = Character_segmentation(self.voc, self.input_dim)
        self.FeatureOrdering = FeatureOrdering(self.scales, self.maxT, self.depth, self.num_channels)

    def forward(self,basic_feature):
        visual_feature = self.Character_segmentation(basic_feature)
        ord_seg_maps = self.FeatureOrdering(basic_feature)
        nB, nC, nH, nW = char_seg_map.size()
        # nB, nT, nH, nW = ord_seg_maps.size()
        nT = ord_seg_maps.size()[1]
        #normalize
        ord_seg_maps = ord_seg_maps / ord_seg_maps.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1)
        # single feature map for nB, nT, nC, nH, nW
        # view for numpy , and einops conduct the rearrange reduce repeat
        ordered_features = char_seg_map.view(nB, 1, nC, nH, nW) * ord_seg_maps.view(nB, nT, 1, nH, nW)
        # visual_features size (nB, nT, nC, nH, nW)
        # weighted sum , translate to 1-D dim sequence
        # visual_features = reduce(visual_features, 'b t c h w -> b c t', reduction='sum')
        # pred = visual_features.view(nB, nT, nC, -1).sum(3).tranpose(1,0) #nB, nC, nT
        pred_vector = ordered_features.view(nB, nT, nC, -1).sum(3) # nB, nT, nC
        return visual_feature, ordered_features, pred_vector

