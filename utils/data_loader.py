#from util import model_utils
import os
import random
import argparse
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.serialization import load_lua

class Dataset(data.Dataset):
        ''' FUNCTIONS DO FOLLOWINGS ###
        1. Read manifest.txt to save file names to be read ex. ~/DATA/CUB/manifest.txt
        2. Read trainvalids.txt to read train file only (total #150) ex. ~/trainvalids.txt
        3. Read image files (train ids) (.t7) ex. ~/DATA/CUB/images/200.Common_Yellowthroat.t7
        4. Read text files (train ids) (.t7)  ex. ~/DATA/CUB/text_c10/200.Common_Yellowthroat.t7

        '''''''''''''''''''''''''''''''''
        def __init__(self, config):
            self.ntrain = config.ntrain
            self.data_dir = config.data_dir
            self.img_dim = config.image_dim
            self.doc_length = config.doc_length
            self.randomize_pair = config.randomize_pair
            self.num_caption = config.num_caption
            self.image_dir = config.image_dir
            self.flip = config.flip
            self.ids_file = config.ids_file
            self.alphabet = config.alphabet
            self.dict = {}
            for i in range(len(self.alphabet)):
                self.dict[self.alphabet[i]] = i
            self.alphabet_size = len(self.alphabet) # size: 70

            ## load manifest file.
            self.files = []
            # path of file names: /home/cvpr19/scottreed/DATA/CUB/manifest.txt
            file_list = open(os.path.join(self.data_dir, 'manifest.txt')).readlines()
            for line in file_list:
                # ex. self.files[0]: 001.Black_footed_Albatross.t7
                self.files.append(line)

            ## load train / val / test splits.
            self.trainids = []
            # path of train ids: /home/cvpr19/scottreed/DATA/CUB/trainvalids.txt
            train_id_list = open(os.path.join(self.data_dir, self.ids_file)).readlines()
            for line in train_id_list:
                # ex. self.trainids[0]: 003 (three digits)
                self.trainids.append(int(line))
            self.ntrain_train = len(self.trainids) # length of trainids: 150
            self.rand_sample_ix = torch.randperm(self.ntrain_train) # randperm 1 ~ 200

        def __getitem__(self, index):
            txt = torch.zeros(self.doc_length, self.alphabet_size) # 201 x 70
            img = torch.zeros(self.img_dim)

            ## *** Example *** ##
            #  fname[190]: 191.Red_headed_Woodpecker.t7
            #  path of file(image): /home/cvpr19/scottreed/DATA/CUB/images/191.Red_headed_Woodpecker.t7
            #  size of cls_imgs: [# of images per class, 1d img dim, # of captions]=torch.Size([60, 1024, 10])
            #  path of captions: /home/cvpr19/scottreed/DATA/CUB/text_c10/191.Red_headed_Woodpecker.t7
            #  size of cls_sens(captions): [# of images per class , doc_length, # of captions] = torch.Size([60, 201, 10])
            #print ('index = {}'.format(index))
            id = self.trainids[int(self.rand_sample_ix[index])] - 1
            fname = self.files[id][:-1]
            if self.image_dir in ['', None]:
                cls_imgs = load_lua(os.path.join(self.data_dir, 'images', fname))
            else:
                # [# of images per class, 1d img dim, # of captions] = [60, 1024, 10]
                cls_imgs = load_lua(os.path.join(self.data_dir, self.image_dir, fname))
            # [# of images per class , doc_length, # of captions] = [60, 201, 10]
            cls_sens = load_lua(os.path.join(self.data_dir,'text_c{}'.format(self.num_caption), fname))

            sen_ix = torch.Tensor(1)
            sen_ix = random.randint(0, cls_sens.size(2)-1) # random pick one of 10 text captions
            ix = torch.randperm(cls_sens.size(0))[0] # random select an image among all images per class
            ix_view = torch.randperm(cls_imgs.size(2))[0] # random select one view of 10 different view images?
            img = cls_imgs[ix, :, ix_view] # 1024 dim

            # txt: alphabet on
            for j in range(cls_sens.size(1)): # 201
                if cls_sens.size(0) == 1:
                    on_ix = int(cls_sens[0, j, sen_ix]) - 1
                else:
                    on_ix = int(cls_sens[ix, j, sen_ix]) - 1
                if on_ix == -1: # end of text
                    break
                if random.random() < self.flip: # default self.flip=0, random.random()=0 ~ 1
                    txt[cls_sens.size(1) - j + 1, on_ix] = 1
                else:
                    txt[j, on_ix] = 1
            return txt, img#torch.FloatTensor?

        def __len__(self):
            return self.ntrain_train

        def vocab_mapping():
            '''
            1. Read Vocab.t7 file
            2. convert the given sentences along with Vocabs
            '''
            vocab = 0
            return vocab

def get_loader(config):
    """Build and return a data loader."""
    dataset = Dataset(config)
    print('Dataset size: {}'.format(len(dataset)))

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=(config.mode=='train'),
                                  num_workers=config.num_workers)
    return data_loader

if __name__=="__main__":

    print('*** Dataset loader for Testing (python version) ***')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/cub', help='data directory.')
    parser.add_argument('--ntrain', type=int, default=200, help='number of classes')
    parser.add_argument('--doc_length', type=int, default=201, help='document length')
    parser.add_argument('--image_dim', type=int, default=1024, help='image feature dimension')
    parser.add_argument('--batch_size',type=int, default=40, help='number of sequences to train on in parallel')
    parser.add_argument('--randomize_pair', type=int, default=0, help='if 1, images and captions of the same class are randomly paired.')
    #parser.add_argument('--ids_file', type=str, default='trainids.txt', help='file specifying which class labels are used for training. Can also be trainvalids.txt')
    parser.add_argument('--ids_file', type=str, default='trainvalids.txt', help='file specifying which class labels are used for training. Can also be trainvalids.txt')
    #parser.add_argument('--num_caption',type=int, default=5, help='number of captions per image to be used for training')
    parser.add_argument('--num_caption',type=int, default=10, help='number of captions per image to be used for training')
    # parser.add_argument('--image_dir', type=str, default='images_th3', help='image directory in data')
    parser.add_argument('--image_dir', type=str, default='images', help='image directory in data')
    parser.add_argument('--flip',type=int, default=0, help='flip sentence')
    parser.add_argument('--num_workers',type=int, default=4, help='num workers')
    parser.add_argument('--mode',type=str, default='train', help='[train|test] mode')

    config = parser.parse_args()

    loader = get_loader(config)
    print('Batch size: {}'.format(config.batch_size))
    print('The number of batches: {}'.format(len(loader)))

    data_iter = iter(loader)
    txt, img = next(data_iter)
    print('Size of txt: [batch_size, doc_length, alphabet size]={}'.format(txt.size()))
    print('Size of img: [batch_size, 1d image dim]={}'.format(img.size()))


