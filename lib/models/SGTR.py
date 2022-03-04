from __future__ import absolute_import
import sys
import torch
from torch.nn as nn

from .backbone import Backbone
from .VisualRecognition import VisualRecognitionModule
from .LanguageReasoning import LanguageModule
from .GraphTextualReasoning import GraphTextualReasoning
from .DynamicalFusion import DynamicalFusionLayer
from .embedding_head import Embedding, Embedding_self_att
from .loss import RecognitionCrossEntropyLoss, CharSegmentLoss, SmoothL1loss, KLDLoss

from ...config import get_args
global_args = get_args(sys.argv[1:])



class SGTR(nn.Module):
  def __init__(self, backbone_arch1, backbone_arch2, voc_type, max_len_labels, eos):
    super(SGTR, self).__init__()

    self.arch1 = backbone_arch1
    self.arch2 = backbone_arch2
    self.max_len_labels = max_len_labels
    self.eos = eos

    self.backbone = Backbone(self.arch1, self.arch2)
    self.VRM = VisualRecognitionModule()
    self.LM = LanguageModule()
    self.GTR = GraphTextualReasoning(max_len_labels=self.max_len_labels)
    self.fusion = DynamicalFusionLayer()

    self.embeder = Embedding(self.time_step, encoder_out_planes)
    # self.embeder = Embedding_self_att(self.time_step, encoder_out_planes, n_head=4, n_layers=4)

    self.rec_crit = SequenceCrossEntropyLoss()
    self.embed_crit = EmbeddingRegressionLoss(loss_func='cosin')
    self.seg_crit = CharSegmentLoss(voc_type=voc_type)
    self.ord_crit = SmoothL1loss(voc_type=voc_type, training=False)
    self.kl_crit = KLDLoss()


  def forward(self, input_dict):
    return_dict = {}
    return_dict['losses'] = {}
    return_dict['output'] = {}

    x, rec_targets, rec_lengths, rec_embeds = input_dict['images'], \
                                              input_dict['rec_targets'], \
                                              input_dict['rec_lengths'], \
                                              input_dict['rec_embeds']


    base_feats = self.backbone(x)
    visual_feats, ordered_feats, visual_pred = self.VRM(base_feats)
    language_feats, language_pred = self.LM(visual_feats) 
    gtr_feats, gtr_pred = self.GTR(ordered_feats)

    gtr_feats = gtr_feats.contiguous()
    embedding_vectors = self.embeder(gtr_feats)

    if self.training:
      rec_pred = self.fusion(visual_pred, gtr_pred, language_pred)
      loss_rec = self.rec_crit(rec_pred, rec_targets)
      loss_seg = self.seg_crit(visual_feats, target_feats)
      loss_ord = self.ord_crit(ordered_feats, target_feats)
      loss_kl = self.kl_crit(language_pred, gtr_pred) + self.kl_crit(gtr_pred, language_pred)
      return_dict['losses']['loss_rec'] = loss_rec
      return_dict['losses']['loss_vr'] = loss_seg + 0.1 * loss_ord
      return_dict['losses']['loss_kl'] = loss_kl 
    else:
      rec_pred_ = self.fusion(visual_pred, gtr_pred, language_pred)
      #rec_pred, rec_pred_scores = self.decoder.beam_search(encoder_feats, global_args.beam_width, self.eos, embedding_vectors)
      loss_rec = self.rec_crit(rec_pred_, rec_targets)
      loss_seg = 0
      loss_ord = 0
      loss_kl = self.kl_crit(language_pred, gtr_pred) + self.kl_crit(gtr_pred, language_pred)
      return_dict['losses']['loss_rec'] = loss_rec
      return_dict['losses']['loss_kl'] = loss_kl
      return_dict['output']['pred_rec'] = rec_pred_
      return_dict['output']['pred_language'] = language_pred
      return_dict['output']['pred_gtr'] = gtr_pred

    # pytorch0.4 bug on gathering scalar(0-dim) tensors
    for k, v in return_dict['losses'].items():
      return_dict['losses'][k] = v.unsqueeze(0)

    return return_dict
