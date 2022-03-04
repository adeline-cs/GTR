from __future__ import absolute_import
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import numpy as np
from backbones import resnet, resnet_dilated, resnet_fpn, resnet_ppm

class Backbone(nn.Module):
    def __init__(self, res_type,  add_variant ):
        super(Backbone, self).__init__()
        self.res_type = res_type
        self.add_variant = add_variant

        '''
        basemodel select: resnet18, resnet34, resnet50 , 
                        resnet101, deformable_resnet50, resnet152
        the modify is designed as v1 follow CA-FCN
        '''
        if self.res_type == "res18":
            self.basemodel = resnet.resnet18()
        if self.res_type == "res34":
            self.basemodel = resnet.resnet34()
        if self.res_type == "res50":
            self.basemodel = resnet.resnet50()
        if self.res_type == "res101":
            self.basemodel = resnet.resnet101()
        if self.res_type == "res152":
            self.basemodel = resnet.resnet152()
        if self.res_type == "deform-res50":
            self.basemodel = resnet.deformable_resnet50()
        if self.res_type == "res18-v1":
            self.basemodel = resnet.resnet18_modify()
        if self.res_type == "res34-v1":
            self.basemodel = resnet.resnet34_modify()
        if self.res_type == "res50-v1":
            self.basemodel = resnet.resnet50_modify()
        if self.res_type == "res101-v1":
            self.basemodel = resnet.resnet101_modify()
        if self.res_type == "res152-v1":
            self.basemodel = resnet.resnet152_modify()
        if self.res_type == "deform-res50-v1":
            self.basemodel = resnet.deformable_resnet50_modify()

        '''
        backbone model select: basemodel + add
        add: no add, fpn, ppm, dilated, fpn_deform,
        '''
        if self.add_variant == 'no_add':
            self.model = self.basemodel
        if self.add_variant == 'add_fpn':
            self.model = resnet_fpn.ResnetFPN(self.res_type)
        if self.add_variant == 'add_dilated':
            self.model = resnet_dilated.resnet50()
        if self.add_variant == 'add_ppm':
            self.model = resnet_ppm.resnet50dilated_ppm()
        if self.add_variant == 'add_fpn_deform':
            self.model = resnet_fpn.Resnet50_FPN_Deform()

    def forward(self, img):
        base_feature = self.model(img)
        return base_feature
