import torch
import torch.nn as nn
from Nets.EfficientNet_utils import MemoryEfficientSwish


class GCN(nn.Module):
    def __init__(self, d_in, n_in, d_out=None, n_out=None, dropout=0.1):
        super().__init__()

        if d_out is None:
            d_out = d_in
        if n_out is None:
            n_out = n_in

        self.conv_n = nn.Conv1d(n_in, n_out, 1)
        self.linear = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = MemoryEfficientSwish()

    def forward(self, x):
        '''
        :param x: [b, nin, din]
        :return: [b, nout, dout]
        '''

        x = self.conv_n(x)  # [b, nout, din]
        x = self.dropout(self.linear(x))  # [b, nout, dout]

        return self.activation(x)


class PoolAggregate(nn.Module):
    def __init__(self, n_r, d_in, d_middle=None, d_out=None):
        super().__init__()

        if d_middle is None:
            d_middle = d_in
        if d_out is None:
            d_out = d_in

        self.d_in = d_in
        self.d_middle = d_middle
        self.d_out = d_out
        self.activation = MemoryEfficientSwish()

        self.n_r = n_r
        self.aggs = self.build_aggs()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def build_aggs(self):
        aggs = nn.ModuleList()

        for i in range(self.n_r):
            aggs.append(nn.Sequential(
                nn.Conv2d(self.d_in, self.d_middle, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.d_middle, momentum=0.01, eps=0.001),
                self.activation,
                nn.Conv2d(self.d_middle, self.d_out, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.d_out, momentum=0.01, eps=0.001),
            ))

        return aggs

    def forward(self, x):
        '''
        :param x: [b, din, h, w]
        :return: [b, n_r, dout]
        '''

        b = x.size(0)
        out = []
        fmaps = []

        for agg in self.aggs:
            y = agg(x)  # [b, d_out, 1, 1]
            p = self.pool(y)
            fmaps.append(y)
            out.append(p.view(b, 1, -1))

        out = torch.cat(out, dim=1)  # [b, n_r, d_out]

        return out


class WeightAggregate(nn.Module):

    def __init__(self, n_r, d_in, d_middle=None, d_out=None):
        super().__init__()

        if d_middle is None:
            d_middle = d_in
        if d_out is None:
            d_out = d_in

        self.conv_n = nn.Sequential(
            nn.Conv2d(d_in, d_in, 3, 1, 1, bias=False),
            nn.BatchNorm2d(d_in, momentum=0.01, eps=0.001),
            MemoryEfficientSwish(),
            nn.Conv2d(d_in, n_r, 1, bias=False),
            nn.BatchNorm2d(n_r, momentum=0.01, eps=0.001),
            nn.Sigmoid())

        self.conv_d = nn.Sequential(
            nn.Conv2d(d_in, d_middle, 3, 1, 1, bias=False),
            nn.BatchNorm2d(d_middle, momentum=0.01, eps=0.001),
            MemoryEfficientSwish(),
            nn.Conv2d(d_middle, d_out, 1, bias=False),
            nn.BatchNorm2d(d_out, momentum=0.01, eps=0.001))

        self.n_r = n_r
        self.d_out = d_out

    def forward(self, x):
        '''
        :param x: [b, d_in, h, w]
        :return: [b, n_r, dout]
        '''
        b = x.size(0)

        hmaps = self.conv_n(x)  # [b, n_r, h, w]
        fmaps = self.conv_d(x)  # [b, d_out, h, w]

        r = torch.bmm(hmaps.view(b, self.n_r, -1),
                      fmaps.view(b, self.d_out, -1).permute(0, 2, 1))

        return r
