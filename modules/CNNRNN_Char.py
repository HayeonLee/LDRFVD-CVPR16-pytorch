import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.autograd import Function

class CNNRNN_Char(nn.Module):
    def __init__(self, alphasize, emb_dim, dropout=0.0, avg=0, cnn_dim=256):
        super(CNNRNN_Char, self).__init__()
        # 201 x alphasize
        self.Temporal_1 = nn.Sequential(
            nn.Conv1d(alphasize, 384,  kernel_size=4)
            nn.MaxPool1d(3, stride=3))
        # 66 x 256 

        self.Temporal_2 = nn.Sequential(
            nn.Conv1d(384, 512,  kernel_size=4)
            nn.MaxPool1d(3, stride=3))
        # 21 x 256

        self.Temporal_3 = nn.Sequential(
            nn.Conv1d(512, cnn_dim,  kernel_size=4)
            nn.MaxPool1d(3, stride=2))
        # 8 x 256 = 2048

        self.rnn = nn.Sequential()
        for i in range(1,8):
            self.rnn.add_module('rnn {}'.format(i), nn.RNN(256, cnn_dim)
        self.fc1 = nn.Linear(cnn_dim, emb_dim)

    def forward(self, x):
        out = F.relu(self.Temporal_1(x))
        out = F.relu(self.Temporal_2(out))
        out = F.relu(self.Temporal_3(out))
        out = self.rnn(out)
        # add the output of each RNN cells / sequence length?
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        return out # or avg?

