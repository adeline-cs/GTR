from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from ..utils.labelmaps import get_vocabulary

def to_contiguous(tensor):
  if tensor.is_contiguous():
    return tensor
  else:
    return tensor.contiguous()

def _assert_no_grad(variable):
  assert not variable.requires_grad, \
    "nn criterions don't compute the gradient w.r.t. targets - please " \
    "mark these variables as not requiring gradients"


class RecognitionCrossEntropyLoss(nn.Module):  
    
    def __init__(self, voc_type):
        super(RecognitionCrossEntropyLoss, self).__init__()
        self.vocabulary = get_vocabulary(voc_type, EOS = 'EOS', PADDING = 'PADDING', UNKNOWN = 'UNKNOWN')
        self.fc = torch.nn.Linear(256, len(vocabulary))
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, feature_map, targets=None, train=False):
        x = torch.max(torch.max(feature_map, dim=3)[0], dim=2)[0]
        x = self.fc(x)
        pred = x
        if train:
            loss = self.criterion(pred, targets)
            return loss, pred
        else:
            return pred


class CharSegmentLoss(nn.Module):
    
    def __init__(self, voc_type, in_channels=256, inner_channels=256, training=False):
        super(CharSegmentLoss, self).__init__()

	self.mask = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
                nn.Conv2d(inner_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid())
        self.vocabulary = get_vocabulary(voc_type, EOS = 'EOS', PADDING = 'PADDING', UNKNOWN = 'UNKNOWN')
        self.classify = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
                nn.Conv2d(inner_channels, len(vocabulary), kernel_size=1, padding=0))

        self.training = training

    def forward(self, cur_feature, classify):
        mask_pred = self.mask(cur_feature)
        mask_pred = mask_pred.squeeze(1)  # N, H, W
        classify_pred = self.classify(cur_feature)
        pred = dict(mask=mask_pred, classify=classify_pred)
        if self.training:
            mask_loss = nn.functional.binary_cross_entropy(mask_pred, mask)
            classify_loss = nn.functional.cross_entropy(classify_pred, classify)
            loss = mask_loss + classify_loss
            metrics = dict(mask_loss=mask_loss, classify_loss=classify_loss)
            return loss, pred, metrics
        #pred['classify'] = nn.functional.softmax(classify_pred, dim=1)
        return pred

class KLDLoss(nn.Module):
    def __init__(self, ):
        super(KLDLoss, self).__init__()
    
    def forward(self, x1, x2):
        return -0.5 * torch.sum(1 + x2 - x1.pow(2) - x2.exp())

class SequenceCrossEntropyLoss(nn.Module):
  def __init__(self, 
               weight=None,
               size_average=True,
               ignore_index=-100,
               sequence_normalize=False,
               sample_normalize=True):
    super(SequenceCrossEntropyLoss, self).__init__()
    self.weight = weight
    self.size_average = size_average
    self.ignore_index = ignore_index
    self.sequence_normalize = sequence_normalize
    self.sample_normalize = sample_normalize

    assert (sequence_normalize and sample_normalize) == False

  def forward(self, input, target, length):
    _assert_no_grad(target)
    # length to mask
    batch_size, def_max_length = target.size(0), target.size(1)
    mask = torch.zeros(batch_size, def_max_length)
    for i in range(batch_size):
      mask[i,:length[i]].fill_(1)
    mask = mask.type_as(input)
    # truncate to the same size
    max_length = max(length)
    assert max_length == input.size(1)
    target = target[:, :max_length]
    mask =  mask[:, :max_length]
    input = to_contiguous(input).view(-1, input.size(2))
    input = F.log_softmax(input, dim=1)
    target = to_contiguous(target).view(-1, 1)
    mask = to_contiguous(mask).view(-1, 1)
    output = - input.gather(1, target.long()) * mask
  
    output = torch.sum(output)
    if self.sequence_normalize:
      output = output / torch.sum(mask)
    if self.sample_normalize:
      output = output / batch_size

    return output

class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt, mask):
        loss = (torch.abs(pred[:, 0] - gt) * mask).sum() / mask.sum()
        return loss, dict(l1_loss=loss)

class BalanceL1Loss(nn.Module):
    def __init__(self, negative_ratio=3.):
        super(BalanceL1Loss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred: torch.Tensor, gt, mask):
        '''
        Args:
            pred: (N, 1, H, W).
            gt: (N, H, W).
            mask: (N, H, W).
        '''
        loss = torch.abs(pred[:, 0] - gt)
        positive = loss * mask
        negative = loss * (1 - mask)
        positive_count = int(mask.sum())
        negative_count = min(
                int((1 - mask).sum()),
                int(positive_count * self.negative_ratio))
        negative_loss, _ = torch.topk(negative.view(-1), negative_count)
        negative_loss = negative_loss.sum() / negative_count
        positive_loss = positive.sum() / positive_count
        return positive_loss + negative_loss,\
            dict(l1_loss=positive_loss, nge_l1_loss=negative_loss)

class SmoothL1loss(nn.Module):
    def __init__(self):
	super(SmoothL1loss, self).__init__()
	self.HUBER_DELTA = 0.5
	
    def forward(self, pred, gt_mask, gt):
        x = torch.abs(pred - gt)
        x = torch.switch(x < self.HUBER_DELTA, 0.5 * x ** 2, self.HUBER_DELTA * (x - 0.5 * self.HUBER_DELTA))
        return torch.sum(x)


class EmbeddingRegressionLoss(nn.Module):
    def __init__(self,
                 weight=None,
                 size_average=True,
                 ignore_index=-100,
                 sequence_normalize=False,
                 sample_normalize=True,
                 loss_func='cosin'):
        super(EmbeddingRegressionLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.sequence_normalize = sequence_normalize
        self.sample_normalize = sample_normalize
        # self.loss_func = torch.nn.MSELoss()
        self.is_cosin_loss = False
        if loss_func == 'smooth_l1':
            self.loss_func = torch.nn.SmoothL1Loss()
        elif loss_func == 'cosin':
            self.loss_func = torch.nn.CosineEmbeddingLoss()
            self.is_cosin_loss = True

    def forward(self, input, target):
        _assert_no_grad(target)

        if not self.is_cosin_loss:
            Loss = self.loss_func(input, target)
        else:
            label_target = torch.ones(input.size(0)).cuda()
            Loss = self.loss_func(input, target, label_target)

        return Loss
    def logistic_dot_loss(self, input, target):
        dot_result = torch.mm(input, target.t())
        _diagaonal = dot_result.diagonal()
        logistic_loss = torch.log(1 + torch.exp(-1 * _diagaonal))

        # logistic_loss = torch.mean(logistic_loss, dim=0)

        return logistic_loss


"""Implementation of loss module for encoder-decoder based text recognition method with CrossEntropy loss."""
class CELoss(nn.Module):
    def __init__(self, ignore_index=-1, reduction='none'):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']

        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']

        return outputs.permute(0, 2, 1).contiguous(), targets

    def forward(self, outputs, targets_dict, img_metas=None):
        outputs, targets = self.format(outputs, targets_dict)

        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        losses = dict(loss_ce=loss_ce)

        return losses


'''Implementation of loss module in `SAR'''
class SARLoss(CELoss):
    def __init__(self, ignore_index=0, reduction='mean', **kwargs):
        super().__init__(ignore_index, reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        # targets[0, :], [start_idx, idx1, idx2, ..., end_idx, pad_idx...]
        # outputs[0, :, 0], [idx1, idx2, ..., end_idx, ...]

        # ignore first index of target in loss calculation
        targets = targets[:, 1:].contiguous()
        # ignore last index of outputs to be in same seq_len with targets
        outputs = outputs[:, :-1, :].permute(0, 2, 1).contiguous()

        return outputs, targets

"""Implementation of loss module for transformer."""
class TFLoss(CELoss):
    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 flatten=True,
                 **kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)

        self.flatten = flatten

    def format(self, outputs, targets_dict):
        outputs = outputs[:, :-1, :].contiguous()
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()

        return outputs, targets

class SegLoss(nn.Module):
    def __init__(self,
                 seg_downsample_ratio=0.5,
                 seg_with_loss_weight=True,
                 ignore_index=255,
                 **kwargs):
        super().__init__()

        assert isinstance(seg_downsample_ratio, (int, float))
        assert 0 < seg_downsample_ratio <= 1
        assert isinstance(ignore_index, int)

        self.seg_downsample_ratio = seg_downsample_ratio
        self.seg_with_loss_weight = seg_with_loss_weight
        self.ignore_index = ignore_index

    def seg_loss(self, out_head, gt_kernels):
        seg_map = out_head  # bsz * num_classes * H/2 * W/2
        seg_target = [
            item[1].rescale(self.seg_downsample_ratio).to_tensor(
                torch.long, seg_map.device) for item in gt_kernels
        ]
        seg_target = torch.stack(seg_target).squeeze(1)

        loss_weight = None
        if self.seg_with_loss_weight:
            N = torch.sum(seg_target != self.ignore_index)
            N_neg = torch.sum(seg_target == 0)
            weight_val = 1.0 * N_neg / (N - N_neg)
            loss_weight = torch.ones(seg_map.size(1), device=seg_map.device)
            loss_weight[1:] = weight_val

        loss_seg = F.cross_entropy(
            seg_map,
            seg_target,
            weight=loss_weight,
            ignore_index=self.ignore_index)

        return loss_seg

    def forward(self, out_head, gt_kernels):

        losses = {}

        loss_seg = self.seg_loss(out_head, gt_kernels)

        losses['loss_seg'] = loss_seg

        return losses

