import os
import cv2
from data.data_utils import Augmenter

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class listDataset(Dataset):
    def __init__(self, imgdir=None, list_file=None, transform=None, inTrain=False, p_aug=0, vert_test=False):
        '''
        :param imgdir: path to root directory
        :param list_file: path to ground truth file
        :param transform: torchvison transforms object
        :param inTrain: True for training stage and False otherwise
        :param p_aug: probability of data augmentation
        '''

        self.list_file = list_file
        with open(list_file) as fp:
            self.lines = fp.readlines()
            self.nSamples = len(self.lines)

        self.transform = transform
        self.imgdir = imgdir
        self.inTrain = inTrain
        self.p_aug = p_aug
        self.vert_test = vert_test

        if inTrain:
            self.aug = Augmenter(p=self.p_aug)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # -- get image
        line_splits = self.lines[index].split()
        imgpath = os.path.join(self.imgdir, line_splits[0])

        img = cv2.imread(imgpath)

        # ignore invalid images
        if img is None:
            #print('Invalid image {}, use next one.'.format(imgpath))
            return self[index + 1]

        # ignore too small images
        h, w, _ = img.shape
        if min(h, w) <= 5:
            # print('Too small image {}, use next one.'.format(imgpath))
            return self[index + 1]

        # -- get text label
        label = ' '.join(line_splits[1:])
        label = label.lower()

        # ignore too long texts in training stage
        if len(label) >= 25 and self.inTrain:
            # print('Too long text: {}, use next one.'.format(imgpath))
            return self[index + 1]

        # -- data preprocess
        if self.inTrain:
            img = self.aug.apply(img, len(label))

        x = self.transform(img)
        x.sub_(0.5).div_(0.5)  # normalize to [-1, 1)

        # for vertical test samples, return rotated versions
        x_clock, x_counter = 0, 0
        is_vert = False
        if self.vert_test and not self.inTrain and h > w:
            is_vert = True
            img_clock = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img_counter = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_clock = self.transform(img_clock)
            x_counter = self.transform(img_counter)
            x_clock.sub_(0.5).div_(0.5)
            x_counter.sub_(0.5).div_(0.5)

        return (x, label, x_clock, x_counter, is_vert, imgpath)


def TrainLoader(configs):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.Resize((configs.imgH, configs.imgW)),
        transforms.ToTensor()
    ])

    dataset = listDataset(imgdir=configs.image_dir,
                          list_file=configs.train_list,
                          transform=transform,
                          inTrain=True,
                          p_aug=configs.aug_prob)

    return DataLoader(dataset,
                      batch_size=configs.batchsize,
                      shuffle=True,
                      num_workers=configs.workers)


def TestLoader(configs):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((configs.imgH, configs.imgW)),
        transforms.ToTensor()
    ])

    dataset = listDataset(imgdir=configs.image_dir,
                          list_file=configs.val_list,
                          transform=transform,
                          inTrain=False,
                          vert_test=configs.vert_test)

    return DataLoader(dataset,
                      batch_size=configs.batchsize,
                      shuffle=False,
                      num_workers=configs.workers)


if __name__== '__main__':

    from Configs.trainConf import configs
    import matplotlib.pyplot as plt

    train_loader = TrainLoader(configs)
    l = iter(train_loader)
    im, la, *_ = next(l)
    for i in range(100):
        plt.imshow(im[i].permute(1,2,0) * 0.5 + 0.5)
        plt.show()

    # import matplotlib.pyplot as plt
    # from Configs.testConf import configs
    # valloader = TestLoader(configs)
    # l = iter(valloader)
    # im, la, *_ = next(l)
    # plt.imshow(im[0].permute(1, 2, 0) * 0.5 + 0.5)

