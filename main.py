import argparse
import os
import torch
from torch.backends import cudnn
from utils import *
from solver import Solver
#from MultimodalMinibatchLoaderCaption import *
#from data_loader import Dataset
#from data_loader import get_loader


def main(config):

    cudnn.benchmark = True

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    #data_loader = MultimodalMinibatchLoaderCaption(config)
    data_loader = get_loader(config) # Child of torch.util.data.DataLoader

    solver = Solver(data_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

if __name__ == '__main__':
    print('Train a multi-modal embedding model')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/cvpr19/scottreed/DATA/CUB', help='data directory.')
    parser.add_argument('--ntrain', type=int, default=150, help='number of train pairs')
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
    parser.add_argument('--emb_dim', type=int, default=1536, help='embedding dimension')
    parser.add_argument('--image_noop', type=int, default=1, help='if 1, the image encoder is a no-op. In this case emb_dim and image_dim must match')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
    parser.add_argument('--seed', type=int, default=123, help='torch manual random number generator seed')
    parser.add_argument('--model_name', type=str, default='sje_hybrid',help='filename to autosave the checkpont to. Will be inside checkpoint_dir/')
    parser.add_argument('--checkpoint_dir', type=str, default='ckpt_rnn_lr_bias', help='output directory where checkpoints get written')
    parser.add_argument('--init_from', type=str, default='', help='initialize network parameters from checkpoint at this path')
    parser.add_argument('--max_epochs', type=int, default=10000, help='number of full passes through the training data')
    parser.add_argument('--grad_clip', type=int, default=5, help='clip gradients at this value')
    parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='learning rate decay')
    parser.add_argument('--lr_decay_after', type=int, default=1, help='in number of epochs, when to start decaying the learning rate')
    parser.add_argument('--lr_update_step', type=int, default=300)
    # parser.add_argument('--print_every', type=int, default=100, help='how many steps/minibatches between printing out the loss')
    # parser.add_argument('--model_save_step',type=int, default=1000,help='step size of iterations to save checkpoint(model)')
    parser.add_argument('--print_every', type=int, default=10, help='how many steps/minibatches between printing out the loss')
    parser.add_argument('--model_save_step',type=int, default=100,help='step size of iterations to save checkpoint(model)')
    parser.add_argument('--symmetric',type=int, default=1, help='whether to use symmetric form of SJE')
    parser.add_argument('--bidirectional',type=int, default=0, help='use bidirectional version')
    parser.add_argument('--avg', type=int, default=0, help='whether to time-average hidden units')
    parser.add_argument('--cnn_dim', type=int, default=256, help='char-cnn embedding dimension')
    parser.add_argument('--mode', type=str, default='train', help='mode: [train|test]')
    parser.add_argument('--num_workers', type=int, default=1)
    # move from the data loader
    parser.add_argument('--alphabet', type=str, default="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} ")


    config = parser.parse_args()
    if config.image_noop:
        config.emb_dim = config.image_dim
    torch.manual_seed(config.seed)
    print(config)

    main(config)
