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

def get_pad_mask(seq, pad_idx):
    return (seq == pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)       # 返回上三角矩阵
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def get_attn_key_pad_mask(seq_k, seq_q, PAD):
    ''' For masking out the padding part of key sequence.
        seq_k:src_seq
        seq_q:tgt_seq
    '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)                       # 目标序列
    padding_mask = seq_k.eq(PAD)      # 源序列
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        # print(torch.FloatTensor(sinusoid_table).unsqueeze(0).shape)
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # return x + self.pos_table[:, :x.size(1)].clone().detach()
        # print("positionalencoding input", x.shape)
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            # print(mask.shape, attn.shape, v.shape)
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn)       # 第3个维度为权重
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)     # 4*21*512 ---- 4*21*8*64
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) if mask is not None else None # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Torch_transformer_encoder(nn.Module):
    '''
        use pytorch transformer for sequence learning

    '''
    def __init__(self, d_word_vec=512, n_layers=2, n_head=8, d_model=512, dim_feedforward=1024, n_position=256):
        super(Torch_transformer_encoder, self).__init__()

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward)
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=self.layer_norm)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, cnn_feature, src_mask=None, return_attns=False):
        enc_slf_attn_list = []

        # -- Forward
        #print("cnn feature",cnn_feature.shape)
        enc_output = self.dropout(self.position_enc(cnn_feature))  # position embeding

        enc_output = self.encoder(enc_output)

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output

class Transforme_Encoder(nn.Module):
    ''' to capture the global spatial dependencies'''
    '''
    d_word_vec: 位置编码，特征空间维度
    n_layers: transformer的层数
    n_head：多头数量
    d_k: 64
    d_v: 64
    d_model: 512,
    d_inner: 1024
    n_position: 位置编码的最大值
    '''
    def __init__(
            self, d_word_vec=512, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=1024, dropout=0.1, n_position=256):

        super().__init__()

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, cnn_feature, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        #print("cnn feature",cnn_feature.shape)
        enc_output = self.dropout(self.position_enc(cnn_feature))   # position embeding

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output

class VisualSemanticConversion(nn.Module):
    """
    the visual to semantic encoding
    input B T C H W ->   B T C -> ouput B T D
    """
    def __int__(self, n_dim = 512, n_class = 64, n_maxlength = 25):
        super(VisualSemanticConversion, self).__int__()
        self.n_maxlength = n_maxlength
        self.n_dim = n_dim
        self.n_class = n_class
        self.w0 = nn.Linear(self.n_class, self.n_dim)
        self.softmax = nn.Softmax(dim = 2)
        self.argmax = torch.argmax(dim = 2)
        self.sum  = torch.sum(dim = (3,4))
    def forward(self, visual_feature):
        """ squeeze to 1D vector with class prediction, and convert to semantic encoding"""
        softmax_feature  = self.softmax(visual_feature)
        index_feature = self.argmax(softmax_feature)
        index_feature[index_feature>0] = 1
        nB, nT, nC, nH, nW = visual_feature.size()
        """case 1 sum point which channel have argmax index """
        soft_feature = self.sum(softmax_feature * index_feature.view(nB, nT, 1, nH, nW))
        """case 2 sum the whole points for each channel """
        # soft_feature = self.sum(softmax_feature)
        squeezed_feature = self.softmax(soft_feature)
        semantic_feature = self.w0(squeezed_feature)
        return squeezed_feature, semantic_feature

class LanguageModule(nn.Module):
    """
    the language module to refine the incorrect text vector by change the probability of character vector.
    structure  is a stack of  Transformer encoder block
    """
    def __init__(self, n_dim = 512, n_layers = 4, n_class = 64, n_maxlength = 25):
        super(LanguageModule, self).__init__()
        # PAD用于计算mask
        self.PAD = n_class - 1
        self.n_dim = n_dim
        self.n_layer = n_layers
        self.n_class = n_class
        self.n_maxlength = n_maxlength
        self.argmax_embed = nn.Embedding(self.n_class, self.n_dim)
        """global spatial context information, no self attention for original feature  """
        self.transformer_units = Transforme_Encoder(n_layers = self.n_layers, n_position = self.n_maxlength)
        """additional self attention for original head context information"""
        # self.transformer_units = Torch_transformer_encoder(n_layers = self.n_layers, n_position = self.n_maxlength)
        self.w1 = nn.Linear(self.n_dim, self.n_class)
    def forward(self, visual_feature):
        squeezed_feature, semantic_feature = VisualSemanticConversion(visual_feature)
        """
        squeezed_feature: nB, nT, nC
        semantic_feature: nB, nT, dim
        e_argmax：nB, nT
        e： nB, nT, dim
        e_mask: nB, nT, 1
        s: nB, nT, dim
        """
        e_argmax = squeezed_feature.argmax(dim=-1)
        """case1  the embedding from torch embedding"""
        # e = self.argmax_embed(e_argmax)
        """case2 the embedding from linear operation"""
        e = semantic_feature
        e_mask = get_pad_mask(e_argmax, self.PAD)
        s = self.transformer_units(e, None)
        pred_vector = self.w1(s)
        return s, pred_vector