from Nets.model import Model
from Utils.utils import *
from Configs.testConf import configs
from data.dataset import TestLoader
import numpy as np

from progressbar import *


class Tester(object):

    def __init__(self, model, testloader):

        self.device = torch.device('cuda' if configs.cuda else 'cpu')

        self.model = model.to(self.device)
        self.model.eval()

        self.testloader = testloader

        with open(configs.alphabet) as f:
            alphabet = f.readline().strip()
        self.converter = strLabelConverter(alphabet)


    def vert_val(self):
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        progress = ProgressBar(widgets=widgets, maxval=10 * len(self.testloader)).start()

        n_correct = 0.
        n_ims = 0

        with torch.no_grad():

            for step, (ims, texts, ims_clock, ims_counter, is_vert, imgpath) in enumerate(self.testloader):

                ims = ims.to(self.device)
                logits = self.model(ims)  # [1, L, n_class]

                if is_vert[0]:
                    ims_clock = ims_clock.to(self.device)
                    ims_counter = ims_counter.to(self.device)
                    logits_clock = self.model(ims_clock)
                    logits_counter = self.model(ims_counter)

                    score, pred = logits[0].log_softmax(1).max(1)  # [L]
                    pred = list(pred.cpu().numpy())
                    score_clock, pred_clock = logits_clock[0].log_softmax(1).max(1)
                    pred_clock = list(pred_clock.cpu().numpy())
                    score_counter, pred_counter = logits_counter[0].log_softmax(1).max(1)
                    pred_counter = list(pred_counter.cpu().numpy())

                    scores = np.ones(3) * -np.inf

                    if 1 in pred:
                        score = score[:pred.index(1)]
                        scores[0] = score.mean()
                    if 1 in pred_clock:
                        score_clock = score_clock[:pred_clock.index(1)]
                        scores[1] = score_clock.mean()
                    if 1 in pred_counter:
                        score_counter = score_counter[:pred_counter.index(1)]
                        scores[2] = score_counter.mean()

                    c = scores.argmax()
                    if c == 0:
                        pred = pred[:pred.index(1)]
                    elif c == 1:
                        pred = pred_clock[:pred_clock.index(1)]
                    else:
                        pred = pred_counter[:pred_counter.index(1)]

                else:
                    pred = logits[0].argmax(1)
                    pred = list(pred.cpu().numpy())
                    if 1 in pred:
                        pred = pred[:pred.index(1)]

                pred = self.converter.decode(pred)
                pred = pred.replace('<unk>', '')
                gt = texts[0]
                n_correct += (pred == gt)
                n_ims += 1

                if configs.display:
                    print('{} ==> {}  {}'
                          .format(gt, pred, '' if pred == gt else 'error'))

                progress.update(10 * step + 1)
            progress.finish()

            print('-' * 50)
            print('Acc_word = {:.3f}%'.format(100 * n_correct / n_ims))


    def val(self):
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        progress = ProgressBar(widgets=widgets, maxval=10 * len(self.testloader)).start()

        n_correct = 0.
        n_ims = 0

        self.model.eval()
        with torch.no_grad():

            for step, (ims, texts, *_) in enumerate(self.testloader):

                ims = ims.to(self.device)
                logits = self.model(ims)  # [B, L, n_class]
                preds = logits.argmax(2)  # [B, L]

                for pred, gt in zip(preds, texts):
                    pred = list(pred.cpu().numpy())
                    if 1 in pred:
                        pred = pred[:pred.index(1)]
                    pred = self.converter.decode(pred)
                    pred = pred.replace('<unk>', '')
                    n_correct += (pred == gt)
                    n_ims += 1

                    if configs.display:
                        print('{} ==> {}  {}'
                             .format(gt, pred, '' if pred == gt else 'error'))

                progress.update(10 * step + 1)
            progress.finish()

        print('-'*50)
        print('Acc_word = {:.3f}%'.format(100 * n_correct / n_ims))


def main():

    testloader = TestLoader(configs)
    print('[Info] Load data from {}'.format(configs.val_list))

    checkpoint = torch.load(configs.model_path)

    model = Model(checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    print('[Info] Load model from {}'.format(configs.model_path))

    print('# Model Params = {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    tester = Tester(model, testloader)
    if configs.vert_test:
        tester.vert_val()
    else:
        tester.val()


if __name__== '__main__':
    main()
