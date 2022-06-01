from easydict import EasyDict as edict

configs = edict()

# ---- training
configs['image_dir'] = '/home/yrj/Dataset/SceneText/English/'
configs['train_list'] = '/home/yrj/Dataset/SceneText/English/Synth.txt'
configs['savedir'] = './models'
configs['imgH'] = 64
configs['imgW'] = 256

configs['alphabet'] = 'data/alphabet_en.txt'

f = open(configs.alphabet, 'r')
l = f.readline().rstrip()
f.close()
configs['n_class'] = len(l) + 3  # pad, unk, eos

configs['device'] = 'cuda'
configs['random_seed'] = 1
configs['batchsize'] = 128
configs['workers'] = 8

configs['n_epochs'] = 8
configs['lr'] = 0.5
configs['lr_milestones'] = [2, 5, 7]
configs['lr_gammas'] = [0.2, 0.1, 0.1]
configs['weight_decay'] = 0.

configs['aug_prob'] = 0.3
configs['continue_train'] = False
configs['continue_path'] = ''
configs['displayInterval'] = 1000

# ---- model
configs['net'] = edict()

configs.net['n_class'] = configs.n_class
configs.net['max_len'] = 25
configs.net['n_r'] = 5  # number of primitive representations
configs.net['d_model'] = 384
configs.net['dropout'] = 0.1
