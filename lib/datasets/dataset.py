from __future__ import absolute_import
import os
import os.path as osp
from PIL import Image,  ImageFile
import numpy as  np
import lmdb
import six
import sys
import torch
from torch.utils import data
from torch.utils.data import sampler, Dataset, Subset, ConcatDataset
from torchvision import transforms
from ..utils.labelmaps import get_vocabulary, labels2strs
#from ..utils import to_numpy

# what is this function when using ImageFile.
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .. .config import get_args
global_args = get_args(sys.arg1[1:])

class CustomDataset(data.Dataset):
    def __init__(self, root, gt_file_dir, embed_path, voc_type, max_length, num_sample, transform = None):
        super(CustomDataset, self).__init__()
        self.root = root
        self.embed_path = embed_path
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.transform = transform
        # self.num_sample = min(self.nSamples, num_sample)
        assert voc_type in ['LOWERCASE', 'ALLCASE', 'ALLCASE_SYMBOL']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.vocabulary = get_vocabulary(voc_type, EOS = self.EOS, PADDING = self.PADDING, UNKNOWN = self.UNKNOWN)
        self.char2id = dict(zip(self.vocabulary,range(len(self.vocabulary))))
        self.id2char = dict(zip(range(len(self.vocabulary)), self.vocabulary))
        self.rec_num_class = len(self.vocabulary)
        self.lowercase_flag = (voc_type == 'LOWERCASE')

        if osp.basename(gt_file_dir).split('.')[-1] == 'json':
            self.image_path, self.transcription, self.embed_path = self.load_gt_json(gt_file_dir)
        elif osp.basename(gt_file_dir).split('.')[-1] == 'txt':
            self.image_path, self.transcription, self.embed_path = self.load_gt_txt(gt_file_dir)
        self.nSamples_real = min(len(self.image_path),num_sample)

    def __len__(self):
        return self.nSamples_real
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # index += 1
        img_path = self.images_path[index]
        embed_path = self.embeds_path[index]
        word = self.transcriptions[index]
        try:
            img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            if embed_path is not None:
                embed_vector = np.load(os.path.join(self.embed_path, embed_path))
            else:
                embed_vector = np.zeros(300)
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.lowercase:
            word = word.lowercase()

        ## fill with the padding token
        label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
        label_list = []
        for char in word:
            if char in self.char2id:
                label_list.append(self.char2id[char])
            else:
                ## add the unknown token
                # print('{0} is out of vocabulary.'.format(char))
                label_list.append(self.char2id[self.UNKNOWN])
        ## add a stop token
        label_list = label_list + [self.char2id[self.EOS]]
        assert len(label_list) <= self.max_len
        label[:len(label_list)] = np.array(label_list)

        if len(label) <= 0:
            return self[index + 1]

        # label length
        label_len = len(label_list)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, label_len, embed_vector
    def load_gt_json(self, gt_path):
        assert isinstance(gt_path, str), "load_gt_txt need ground truth path"
        with open(gt_path) as f:
            gt_file = json.load(f)
        images_path = []
        transcriptions = []
        embeds = []
        for k in gt_file.keys():
            annotation = gt_file[k]
            """
            if annotation['illegibility'] == True or annotation['laguage'] != 'Latin':
              continue
            """
            # images_path.append(os.path.join(self.root, k))
            images_path.append(k)
            transcriptions.append(annotation['transcription'])
            if self.embed_path is None:
                embeds.append(None)
            else:
                # embed_file_path = os.path.join(self.embed_path, k.replace("jpg", "npy"))
                embed_file_path = k.replace("jpg", "npy")
                if not os.path.exists(os.path.join(self.embed_path, k.replace("jpg", "npy"))):
                    embed_file_path = k.split("/")[5] + "/" + k.split("/")[6].replace("jpg", "npy")
                # embeds.append(os.path.join(self.embed_path, k.replace("jpg", "npy")))
                embeds.append(embed_file_path)
        return image_path, transcription, embeds

    def load_gt_txt(self, gt_path):
        assert isinstance(gt_path, str), "load_gt_txt need ground truth path"
        images_path = []
        transcriptions = []
        embeds = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                line = line.split()
                if len(line) != 2:
                    continue
                # images_path.append(os.path.join(self.root, line[0]))
                images_path.append(line[0])
                transcriptions.append(line[1])
                if self.embed_path is None:
                    embeds.append(None)
                else:
                    # embeds.append(os.path.join(self.embed_path, os.path.basename(line[0]).replace("jpg", "npy")))
                    # embeds.append(os.path.join(self.embed_path, line[0].replace("jpg", "npy")))
                    if "jpg" in line[0]:
                        embeds.append(line[0].replace("jpg", "npy"))
                    elif "png" in line[0]:
                        embeds.append(line[0].replace("png", "npy"))
        return image_path, transcription, embeds

class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        print('-' * 80)
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        self.nums_samples = 0.
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print('-' * 80)
            _dataset = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """

            # opt.total_data_usage_ratio = 1.0 if selected_d == 'ICDAR2019' else 0.5

            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            print(f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}')
            print(f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            self.nums_samples += len(_dataset)
            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=False)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))
        print('-' * 80)
        print('Total_batch_size: ', '+'.join(batch_size_list), '=', str(Total_batch_size))
        opt.batch_size = Total_batch_size
        print('-' * 80)

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []
        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts

    def __len__(self):
        return self.nums_samples

def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    print(f'dataset_root:    {root}\t dataset: {select_data[0]}')
    for dirpath, dirnames, filenames in os.walk(root):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt.voc_type, max_length, num_sample, transform)
                print(f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}')
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset

class LmdbDataset(data.Dataset):
    def __init__(self,root, voc_type, max_length, num_sample, transform = None):
        super(LmdbDataset, self).__init__()
        self.env = lmdb.open(root, max_readers = 32, readonly = True)
        assert self.env is not  None, 'cannot create lmdb from % s'%root
        self.txn = self.env.begin()
        self.voc_type = voc_type
        self.transform = transform
        self.max_length = max_length
        self.nSample = min(self.txn.get(b"num-samples"))
        nSamples = int(self.txn.get('num-samples'.encode()))
        self.nSample = min(self.nSample, num_sample)
        assert voc_type in ['LOWERCASE', 'ALLCASE', 'ALLCASE_SYMBOL']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.vocabulary = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.vocabulary, range(len(self.vocabulary))))
        self.id2char = dict(zip(range(len(self.vocabulary)), self.vocabulary))
        self.rec_num_class = len(self.vocabulary)
        self.lowercase_flag = (voc_type == 'LOWERCASE')

    def __len__(self):
        return self.nSample

    def __getitem__(self, index):
        assert index<=len(self), 'index range error'
        index += 1
        img_key = b'image-%09d' % index
        imgbuf = self.txn.get(img_key)
        label_key = b'label-%09d' % index
        label = self.txn.get(label_key).decode('utf-8')

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert('RGB')  # for color image
            # img = Image.open(buf).convert('L')
            # img = img.convert('RGB')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        if self.lowercase_flag:
            label = label.lower()

        label_pad = np.full((self.max_length,), self.char2id[self.PADDING], dtype = np.int)
        label_list = []

        for char in label:
            if char in self.char2id:
                label_list.append(self.char2id[char])
            else:
            ## add the unknown token
            # print('{0} is out of vocabulary.'.format(char))
                label_list.append(self.char2id[self.UNKNOWN])
        ## add a stop token
        label_list = label_list + [self.char2id[self.EOS]]
        assert len(label_list) <= self.max_length
        label_pad[:len(label_list)] = np.array(label_list)

        if len(label_pad) <= 0:
            return self[index + 1]
        # label length
        label_length = len(label_list)
        # Embedding vectors
        embed_key = b'embed-%09d' % index
        embed_vec = self.txn.get(embed_key)
        # the embedding vector dim is 300
        if embed_vec is not None:
            embed_vec = embed_vec.decode()
        else:
            embed_vec = ' '.join(['0']*300)
        # make string vector to numpy ndarray
        embed_vec = np.array(embed_vec.split()).astype(np.float32)
        if embed_vec.shape[0] != 300:
            return self[index + 1]
        if transforms is not None:
            img = self.transform(img)
        return  img, label, label_length, embed_vec

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

class AlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        #images, labels = zip(*batch)
        images, labels, lengths, embeds  =  zip(*batch)
        labels = torch.IntTensor(labels)
        lengths = torch.IntTensor(lengths)
        embeds = torch.FloatTensor(embeds)


        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            transform = NormalizePAD((1, self.imgH, resized_max_w))
            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels, lengths, embeds


def debug():
    img_root_dir = "/data2/data/ART/train_images/"
    gt_file_path = "/data2/data/ART/train_labels.json"
    train_dataset = CustomDataset(root=img_root_dir, gt_file_path=gt_file_path, voc_type="ALLCASES_SYMBOLS", max_len=50,
                                  num_samples=5000)
    batch_size = 4
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=AlignCollate(imgH=64, imgW=256, keep_ratio=False))
    for i, (images, labels, lengths, masks) in enumerate(train_dataloader):
        print(i)
        # images = images.permute(0, 2, 3, 1)
        # images = to_numpy(images)
        # images = images * 0.5 + 0.5
        # images = images * 255
        # for id, (image, label, label_len) in enumerate(zip(images, labels, lengths)):
        #   image = Image.fromarray(np.uint8(image))
        #   trans = labels2strs(label, train_dataset.id2char, train_dataset.char2id)[0]
        # image.save("show_crop/" + trans + "_" + str(i) + ".jpg")
        # image = toPILImage(image)
        # image.show()
        #       # print(image.size)
        # print(labels2strs(label, train_dataset.id2char, train_dataset.char2id))
        # print(label_len.item())
        # input()
        # if i == 4:
        #   break

if __name__ == "__main__":
    debug()