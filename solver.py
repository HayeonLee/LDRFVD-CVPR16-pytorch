import os
#from modules import *
#from utils import *
#from MultimodalMinibatchLoaderCaption import *
#from ImageEncoder import *
#from HybridCNN import *
from modules import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
import time
import datetime

class Solver(object):
    """Solver for training and testing multi-modal embedding model."""
    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.bidirectional = config.bidirectional
        self.image_dim = config.image_dim
        self.emb_dim = config.emb_dim
        self.image_noop = config.image_noop
        self.dropout = config.dropout
        self.cnn_dim = config.cnn_dim
        self.avg = config.avg
        self.symmetric = config.symmetric
        self.alphabet = config.alphabet
        self.ntrain = config.ntrain

        # Training configurations.
        self.batch_size = config.batch_size
        self.ids_file = config.ids_file
        self.num_caption = config.num_caption
        self.init_from = config.init_from
        self.max_epochs = config.max_epochs
        self.grad_clip = config.grad_clip
        self.lr = config.lr # 0.0004
        self.lr_decay = config.lr_decay # 0.98
        self.lr_decay_after = config.lr_decay_after # 1
        self.lr_update_step = config.lr_update_step # 1

        # Miscellaneous.
        self.gpuid = config.gpuid
        self.seed = config.seed
        self.print_every = config.print_every
        #self.eval_val_every = config.eval_val_every
        self.model_save_step = config.model_save_step
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.image_dir = config.image_dir
        #self.model_name = config.savefile
        self.checkpoint_dir = config.checkpoint_dir

        # Build model.
        self.build_model()

    def build_model(self):

        if len(self.init_from) > 0:
            self.restore_model()
        else:
            print('Create model...')
            ###********************************** Change this parts to check your model ***************************************####

            self.DocumentCNN = HybridCNN(len(self.alphabet), self.emb_dim, self.dropout, self.avg, self.cnn_dim, random_init=True)
            self.ImageEncoder = ImageEncoder(self.image_dim, self.emb_dim, self.image_noop, random_init=True)

            ####***************************************************************************************************************####

        print('Create optimizer...')
        params_list = list(self.ImageEncoder.parameters()) + list(self.DocumentCNN.parameters())
        self.optimizer = optim.RMSprop(params_list, lr=self.lr)
        #self.optimizer = optim.Adam(params_list, lr=self.lr)
        #self.optimizer = optim.SGD(params_list, lr=self.lr, momentum=0.9)
        self.print_network(self.ImageEncoder, 'image encoder')
        self.print_network(self.DocumentCNN, 'text encoder')

        # GPU mode.
        print('Change to GPU mode')
        self.ImageEncoder.to(self.device)
        self.DocumentCNN.to(self.device)

        self.acc_batch = 0.0
        self.acc_smooth = 0.0

    def restore_model(self):
        print('Loading the trained models from the checkpoint: {}'.format(self.init_from))
        DocumentCNN_path = os.path.join('{}-E_doc.ckpt'.format(self.init_from))
        enc_img_path = os.path.join('{}-E_img.ckpt'.format(self.init_from))
        self.DocumentCNN.load_state_dict(torch.load(DocumentCNN_path, map_location=lambda storage, loc: storage))
        self.ImageEncoder.load_state_dict(torch.load(enc_img_path, map_location=lambda storage, loc: storage))

    def save_model(self, i):
        #checkpoint.vocab = loader.vocab_mapping
        DocumentCNN_path = os.path.join(
            self.checkpoint_dir,
            '{0:d}-E_doc.ckpt'.format(i+1))
        ImageEncoder_path = os.path.join(
            self.checkpoint_dir,
            '{0:d}-E_img.ckpt'.format(i+1))
        torch.save(self.DocumentCNN.state_dict(), DocumentCNN_path)
        torch.save(self.ImageEncoder.state_dict(), ImageEncoder_path)
        print('Saved model checkpoints into {}...'.format(self.checkpoint_dir))

    def update_lr(self, epoch):
        if epoch >= self.lr_decay_after:
            self.lr = self.lr * self.lr_decay #decay it
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            #print('Decayed learning rate by a factor {} to {}'.format(self.lr_decay, self.lr))

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    ###************ Complete this loss function ************####
    def joint_embedding_loss(self, fea_img, fea_txt, labels):
        num_class = fea_txt.size(0)
        pair_size = fea_img.size(0)
        #print('num_class = {}'.format(num_class)) #batch_size
        #print('pair_size = {}'.format(pair_size)) #batch_size
        score = torch.zeros(pair_size, num_class) #[40,40]

        loss = 0
        #self.acc_batch = 0.0
        acc_batch = 0.0
        for i in range(pair_size):
            for j in range(num_class):
                #print('fea_img size= {}'.format(fea_img[i].size()))
                #print('fea_txt size= {}'.format(fea_txt[j].size()))
                score[i,j] = torch.dot(fea_img[i], fea_txt[j])
            label_score = score[i, labels[i]]
            for j in range(num_class):
                if j != labels[i]:
                    cur_score = score[i,j]
                    thresh = cur_score - label_score + 1
                    if thresh > 0:
                        loss = loss + thresh
            #print('score size= {}'.format(score.size()))
            #max_score, max_ix = torch.max(score, dim=1)
            #print('label size = {}'.format(labels.size()))
            #print('max_score size = {}'.format(max_score.size()))
            #print('max_ix size = {}'.format(max_ix.size()))
            #self.acc_batch = self.acc_batch + sum(max_ix == labels)
        max_score, max_ix = torch.max(score, dim=1)
        #self.acc_batch = (self.acc_batch + sum(max_ix == labels)).float()
        #self.acc_batch = 100 * (self.acc_batch / pair_size)
        acc_batch = sum(max_ix == labels).float()
        self.acc_batch = self.acc_batch + acc_batch
        denom = pair_size * num_class
        self.acc_smooth = 0.99 * self.acc_smooth + 0.01 * self.acc_batch
        return loss / denom

    def train(self):

        iters = self.max_epochs * len(self.data_loader)
        data_iter = iter(self.data_loader)
        print('Start training...')

        start_time = time.time()
        for i in range(iters):
            epoch = i / len(self.data_loader)
            ## get minibatch.
            try:
                txt, img = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                txt, img = next(data_iter)
            labels = torch.arange(img.size(0), dtype=torch.long)

            txt = txt.to(self.device) #torch tensor [batch_size, doc_length, 70]
            img = img.to(self.device) #torch tensor [batch_size, 1024]
            #print('txt size: {}'.format(txt.size()))
            #print('img size: {}'.format(img.size()))

            ## Forward pass.
            fea_txt = self.DocumentCNN(txt).type(torch.cuda.DoubleTensor) #tensor [batch_size, 1024])
            fea_img = self.ImageEncoder(img).type(torch.cuda.DoubleTensor) #tensor [batch_size, 1024])
            #print('fea_txt size: {}'.format(fea_txt.size()))
            #print('fea_img size: {}'.format(fea_img.size()))

            ## Compute loss.
            loss = self.joint_embedding_loss(fea_txt, fea_img, labels)

            if self.symmetric == 1:
                loss2 = self.joint_embedding_loss(fea_img, fea_txt, labels)
                loss = loss + loss2

            ## Reset the gradient buffers.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ## Exponential learning rate decay.
            if (i+1) % (len(self.data_loader) * self.lr_update_step) == 0 and self.lr_decay < 1:
                self.update_lr(epoch)


            ## Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                self.save_model(i)

            ## Print out training information.
            if (i+1) % self.print_every == 0:
                self.acc_batch = 100 * (self.acc_batch / (self.batch_size * self.print_every))
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration=[{}/{}]".format(et, i+1, iters)
                log += ", Epoch=[{0}], Loss=[{1:4.5f}], lr=[{2:4.5f}], Acc1=[{3:.2f}%], Acc2=[{4:.2f}%]".format(int(epoch+1), loss, self.lr, self.acc_batch, self.acc_smooth)
                self.acc_batch = 0.0
                # log += ", epoch=[{0:3d}], loss=[{1:4.2f}], acc1=[{2:4.2f}], acc2=[{3:4.2f}], g/p=[{4:6.4e}]".format(epoch, train_loss, self.acc_batch, self.acc_smooth, grad_params.norm() / params.norm())
                print(log)

    def test(self):
        pass



