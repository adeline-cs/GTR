from Configs.trainConf import configs
from data.dataset import TrainLoader
from Nets.model import Model
from Utils.utils import *

import random, pprint
import numpy as np
from progressbar import *

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer(object):

    def __init__(self, model, trainloader):

        self.device = torch.device(configs.device)

        self.trainloader = trainloader
        self.configs = configs

        with open(self.configs.alphabet) as f:
            alphabet = f.readline().strip()
        self.converter = strLabelConverter(alphabet)

        self.savedir = os.path.join(self.configs.savedir, time.strftime('%Y%m%d-%H:%M:%S', time.localtime()))
        os.makedirs(self.savedir)
        self.logfile = os.path.join(self.savedir, 'log.txt')

        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)  # ignore pad

        if not self.configs.continue_train:
            self.model = model.to(self.device)
            self.optimizer = ScheduledOptim(optim.Adadelta(filter(lambda x: x.requires_grad, self.model.parameters()),
                                                           lr=self.configs.lr, weight_decay=self.configs.weight_decay),
                                            init_lr = self.configs.lr, milestones=self.configs.lr_milestones, gammas=self.configs.lr_gammas)

        else:
            ckpt = torch.load(self.configs.continue_path)
            self.model = Model(ckpt['model_config'])
            self.model.load_state_dict(ckpt['state_dict'])
            self.model = self.model.to(self.device)

            self.optimizer = ScheduledOptim(optim.Adadelta(filter(lambda x: x.requires_grad, self.model.parameters()),
                                                           lr=self.configs.lr, weight_decay=self.configs.weight_decay),
                                            init_lr = self.configs.lr, milestones=self.configs.lr_milestones, gammas=self.configs.lr_gammas)
            self.optimizer._optimizer.load_state_dict(ckpt['optimizer'])
            self.configs.net = ckpt['model_config']
        
        self.log(pprint.pformat(self.configs))


    def savemodel(self, savename):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer._optimizer.state_dict(),
                    'model_config': self.configs.net},
                    savename)
        self.log('-'*50 + '\n[Info] Model saved as {}\n'.format(savename) + '-'*50)


    def train(self):

        for epoch in range(self.configs.n_epochs):
            self.log('=' * 25 + 'Training Epoch {} Start'.format(epoch + 1) + '=' * 25)
            loss = self.train_epoch(epoch)
            self.log('Epoch [{}/{}]  train loss = {:.3f}'
                     .format(epoch + 1, self.configs.n_epochs, loss))

            savename = os.path.join(self.savedir, 'm_epoch{}.pth'.format(epoch + 1))
            self.savemodel(savename)


    def train_epoch(self, epoch):

        self.optimizer.update_lr(epoch)
        self.model.train()

        total_loss = 0.

        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        progress = ProgressBar(widgets=widgets, maxval=10 * len(self.trainloader)).start()

        for step, (ims, texts, *_) in enumerate(self.trainloader):

            # prepare data
            ims = ims.to(self.device)  # [B, 3, 64, 256]
            targets = self.converter.encode(texts).to(self.device)  # [B, L]

            # forward
            self.optimizer.zero_grad()

            logits = self.model(ims)

            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if step % self.configs.displayInterval == 0:
                pred = logits[0].detach().argmax(1)
                pred = list(pred.cpu().numpy())
                if 1 in pred:
                    pred = pred[:pred.index(1)]
                pred = self.converter.decode(pred)

                self.log('[{}] [{}/{}]  loss = {:.3f}  {} ==> {}'
                         .format(epoch + 1, step, len(self.trainloader),
                                 loss.item(), texts[0], pred))

            progress.update(10 * step + 1)
            step += 1
        progress.finish()

        return total_loss / len(self.trainloader)


    def log(self, results):
        if not isinstance(results, str):
            results = str(results)

        print(results)
        with open(self.logfile, 'a') as f:
            f.write(results + '\n')


def main():
    # ---- set random seed
    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)
    torch.cuda.manual_seed(configs.random_seed)
    torch.backends.cudnn.deterministic = True

    trainloader = TrainLoader(configs)
    print('load train data from {}'.format(configs.train_list))

    model = Model(configs.net)
    print('# Model Params = {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    trainer = Trainer(model, trainloader)
    trainer.train()


if __name__ == '__main__':
    main()
