import torch


class strLabelConverter(object):
    """Convert between str and label.

    Args:
        alphabet (str): set of the possible characters.
    """

    def __init__(self, alphabet, maxT=25):
        self.alphabet = alphabet
        self.maxT = maxT

        self.dict = {}

        self.dict['<pad>'] = 0  # pad
        self.dict['<eos>'] = 1  # EOS
        self.dict['<unk>'] = 2  # OOV
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 3  # encoding from 3 for characters in alphabet

        self.chars = list(self.dict.keys())

    def encode(self, text):
        """
        Args:
            text (list of str): texts to convert.
        Returns:
            torch.IntTensor targets: [b, L]
        """

        tars = []
        for s in text:
            tar = []
            for c in s:
                if c in self.dict.keys():
                    tar.append(self.dict[c])
                else:
                    tar.append(self.dict['<unk>'])
            tars.append(torch.LongTensor(tar))

        b = len(tars)
        targets = self.dict['<pad>'] * torch.ones(b, self.maxT)

        for i in range(b):
            targets[i][:len(tars[i])] = tars[i]
            targets[i][len(tars[i])] = self.dict['<eos>']

        return targets.long()

    def decode(self, t):
        texts = [self.chars[i] for i in t]
        return ''.join(texts)


# ---- learning rate scheduler
class ScheduledOptim():
    '''A wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, milestones, gammas):
        self._optimizer = optimizer
        self.lr = init_lr
        self.milestones = milestones
        self.gammas = gammas

    def step(self):
        "Step with the inner optimizer"
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def update_lr(self, epoch):
        ''' Learning rate scheduling per step '''
        if epoch in self.milestones:
            self.lr *= self.gammas[self.milestones.index(epoch)]

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr
