#from util import model_utils
import os
import random
import argparse
import torch
import torch.nn as nn
from termcolor import cprint
from torch.utils.serialization import load_lua

#data.Dataset

class MultimodalMinibatchLoaderCaption(nn.Module):
        ''' FUNCTIONS DO FOLLOWINGS ###
        1. Read manifest.txt to save file names to be read ex. ~/DATA/CUB/manifest.txt
        2. Read trainvalids.txt to read train file only (total #150) ex. ~/trainvalids.txt
        3. Read image files (train ids) (.t7) ex. ~/DATA/CUB/images/200.Common_Yellowthroat.t7
        4. Read text files (train ids) (.t7)  ex. ~/DATA/CUB/text_c10/200.Common_Yellowthroat.t7

        '''''''''''''''''''''''''''''''''
        def __init__(self, config):
            self.nclass = config.nclass
            self.batch_size = config.batch_size
            self.data_dir = config.data_dir
            self.img_dim = config.image_dim
            self.doc_length = config.doc_length
            self.randomize_pair = config.randomize_pair
            self.num_caption = config.num_caption
            self.image_dir = config.image_dir
            self.flip = config.flip
            self.ids_file = config.ids_file
            self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
            self.dict = {}
            for i in range(len(self.alphabet)):
                self.dict[self.alphabet[i]] = i
            self.alphabet_size = len(self.alphabet) # size: 70

            ## load manifest file.
            self.files = []
            # path of file names: /home/cvpr19/scottreed/DATA/CUB/manifest.txt
            file_list = open(os.path.join(self.data_dir, 'manifest.txt')).readlines()
            for i, line in enumerate(file_list):
                # ex. self.files[0]: 001.Black_footed_Albatross.t7
                self.files.append(line)

            ## load train / val / test splits.
            self.trainids = []
            # path of train ids: /home/cvpr19/scottreed/DATA/CUB/trainvalids.txt
            train_id_list = open(os.path.join(self.data_dir, self.ids_file)).readlines()
            for i, line in enumerate(train_id_list):
                # ex. self.trainids[0]: 003 (three digits)
                self.trainids.append(int(line))

            self.nclass_train = len(self.trainids) # length of trainids: 150

        def next_batch(self):
            sample_ix = torch.randperm(self.nclass_train)
            sample_ix = sample_ix.narrow(0,0,self.batch_size)

            txt = torch.zeros(self.batch_size, self.doc_length, self.alphabet_size)
            img = torch.zeros(self.batch_size, self.img_dim)
            labels = torch.zeros(self.batch_size)

            ## *** Example *** ##
            #  fname[190]: 191.Red_headed_Woodpecker.t7
            #  path of file(image): /home/cvpr19/scottreed/DATA/CUB/images/191.Red_headed_Woodpecker.t7
            #  size of cls_imgs: [# of images per class, 1d img dim, 10 diff views]=torch.Size([60, 1024, 10])
            #  path of captions: /home/cvpr19/scottreed/DATA/CUB/text_c10/191.Red_headed_Woodpecker.t7
            #  size of cls_sens(captions): [# of images per class , doc_length, # of captions] = torch.Size([60, 201, 10])
            for i in range(self.batch_size):
                id = self.trainids[int(sample_ix[i])] - 1
                fname = self.files[id][:-1]
                if self.image_dir in ['', None]:
                    cls_imgs = load_lua(os.path.join(self.data_dir, 'images', fname))
                else:
                    # [# of images per class, 1d img dim, # of captions] = [60, 1024, 10]
                    cls_imgs = load_lua(os.path.join(self.data_dir, self.image_dir, fname))
                # [# of images per class , doc_length, # of captions] = [60, 201, 10]
                cls_sens = load_lua(os.path.join(self.data_dir,'text_c{}'.format(self.num_caption), fname))

                sen_ix = torch.Tensor(1)
                sen_ix = random.randint(0, cls_sens.size(2)-1) # random pick one of 10 text captions 0 ~ 9
                ix = torch.randperm(cls_sens.size(0))[0] # random select an image among all images per class 0 ~ 59
                ix_view = torch.randperm(cls_imgs.size(2))[0] # random select one view of 10 different view images? 
                img[i] = cls_imgs[ix, :, ix_view] # 1024 dim (per a random view of a random image)
                labels[i] = i

                # txt: alphabet on
                for j in range(cls_sens.size(1)): # 201
                    if cls_sens.size(0) == 1:
                        on_ix = int(cls_sens[0, j, sen_ix]) - 1
                    else:
                        on_ix = int(cls_sens[ix, j, sen_ix]) - 1

                    if on_ix == -1: # end of text
                        break

                    if random.random() < self.flip:
                        txt[i, cls_sens.size(1) - j + 1, on_ix] = 1
                    else:
                        txt[i, j, on_ix] = 1

            return txt, img, labels

        def vocab_mapping():
            '''
            1. Read Vocab.t7 file
            2. convert the given sentences along with Vocabs
            '''
            vocab = 0
            return vocab

if __name__=="__main__":

    print('*** Dataset loader for Testing (python version) ***')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/cvpr19/scottreed/DATA/CUB', help='data directory.')
    parser.add_argument('--nclass', type=int, default=200, help='number of classes')
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
    config = parser.parse_args()

    loader = MultimodalMinibatchLoaderCaption(config)
    txt, img, labels = loader.next_batch()
    print('size of txt: [batch_size, doc_length, alphabet size]={}'.format(txt.size()))
    print('size of img: [batch_size, 1d image dim]={}'.format(img.size()))
    print('size of labels: {}'.format(labels.size()))

