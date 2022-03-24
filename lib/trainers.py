from __future__ import print_function, absolute_import
import time
from datetime import datetime
import os.path as osp
import sys
from  PIL import Image
import numpy as np
import torch
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis, flop_count_table
from . import  evaluation_metrics
from . import metric
from metric import Accuracy, EditDistance, RecPostProcess
from ..config import get_args
global_args = get_args(sys.argv[1:])

class BaseTrainer(object):
    def __init__(self, model, metric, log_dir, iters = 0, grad_clip = -1, use_cuda = True, loss_weights = {} ):
        super(BaseTrainer, self).__init__()
        self.model  = model
        self.metric = metric
        self.log_dir = log_dir
        self.iters = iters
        self.grad_clip = grad_clip
        self.use_cuda = use_cuda
        self.loss_weights = loss_weights
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def train(self, epoch, data_loader, optimizer, current_lr = 0.0, print_freq = 100, train_tfLogger = None,
              evaluation = None, test_loader = None, eval_tfLogger = None, test_dataset = None, test_freq = 1000):
        self.model.train()
        # batch_time =
        # data_time =
        losses = {}
        end = time.time()
        for i, input in enumerate(data_loader):
            self.model.train()
            self.iters +=1
            data_time.update(time.time()-end)
            input_dict = self._parse_data(input) # input dict contain four keys: images, rec_targets, rec_length, rec_embeds.
            output_dict = self._forward(input_dict) # the responding output dict
            batch_size = input_dict['images'].size(0) # batch_size for each train loader dataset.
            total_loss = 0
            loss_dict ={}
            for k, loss in output_dict['losses'].items():
                loss = loss.mean(dim = 0, keepdim = True)
                total_loss += self.loss_weights * loss
                loss_dict[k] = loss.item()
                #print('{0}:{1}'.format((k, loss.item())))
            losses.update(total_loss.item(), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils,clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            ## two time relative: time, datetime
            if self.iters % print_freq ==0:
                print('[{}]\t'
                      'Epoch : [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Embed loss: {:.5f}\t'
                      'Recog loss: {:.3f}\t'
                      .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              epoch, i+1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              loss_dict['loss_embed'],
                              #loss_dict['loss_rec']
                        )
                      )

            #=======TensorBoard logging=====
            # parameter : iters, print_freq, train_tfLogger,
            # train_tfLogger parameter: tag(loss_dict.keys), value(loss_dict.value), step = epoch * len(data_loader) + (i + 1)
            if self.iters % (print_freq*10) == 0:
                if train_tfLogger is not None:
                    step = epoch * len(data_loader) + (i + 1)
                    info = {
                        "lr": current_lr,
                        'loss': total_loss.item(),} # this is total loss
                    # add each loss
                    for k, loss in loss_dict.items():
                        info[k] = loss
                    for tag, value in info.items():
                        train_tfLogger.scalar_summary(tag, value, step)

            #========evaluation==========
            # parameter: iters, test_freq, output_dict, is_best, test_loader, eval_tfLogger, test_dataset, best_res
            # is_best: True or False
            # function: save_checkpoint
            if self.iters % test_freq == 0:
                #only symmetry branch
                if 'loss_rec' not in output_dict['losses']:
                    is_best = True
                    self.best_res = evaluator.evaluate(test_loader, step = self.iters, tfLogger = eval_tfLogger, dataset = test_dataset)
                else:
                    res = evaluator.evaluate(test_loader, step = self.iters, tfLogger = eval_tfLogger, dataset = test_dataset)
                    if self.metric == 'accurary':
                        is_best = res > self.best_res
                        self.best_res = max(res, self.best_res)
                    elif self.metric == 'editdistance':
                        is_best = res < self.best_res
                        self.best_res = min(res, self.best_res)
                    else:
                        raise ValueError('unsupported evaluation metric:', self.metric)
                    print('\n Finishied iters {:.3d} accuracy :{:5.1%} best :{:5.1%}{}\n'
                          .format(self.iters, res, self.best_res, '*' if is_best else ''))
                save_checkpoint({
                    'state_dict':self.model.module.state_dict(),
                    'iters': self.iters,
                    'best_res': self.best_res,
                    },
                    is_best,fpath = osp.join(self.log_dir, 'checkpoint.pth')
                )

    ## privated tensor, but all can visite it in python.
    def _parse_data(self, inputs):
        raise NotImplementedError
    def _forward(self, inputs, targets):
        raise NotImplementedError

class Trainer(BaseTrainer):
    ## it will cover the base same named funtion.
    ## it define the parse about train data for _forward function
    def _parse_data(self, inputs):
        input_dict = {}
        #images , label_encs, lengths = inputs
        images, label_enc, lengths, embeddings_ = inputs
        imgs = images.to(self.device)

        if label_enc is not None:
            labels = label_enc.to(self.device)
        if embeddings_ is not None:
            embeds = embeddings_.to(self.device)

        input_dict['images'] = imgs
        input_dict['rec_labels'] = labels
        input_dict['rec_lengths'] = lengths
        input_dict['rec_embeds'] = embeds
        return input_dict

    def _forward(self, input_dict):
        self.model.train()
        output_dict = self.model(input_dict)
        print('Flops counting', flop_count_table(flops=FlopCountAnalysis(self.model, input_dict)))
        return output_dict




