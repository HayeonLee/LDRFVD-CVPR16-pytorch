import torch
import torch.nn as nn

class HybridCNN(nn.Module):
  def __init__(self, alphasize, emb_dim, dropout=0.0, avg=0, cnn_dim=256, random_init=True):
    super(HybridCNN, self).__init__()
    ## *** Caution *** ##
    # input shapes of 1d conv between torch and pytorch are different.
    # torch: nn.TemporalConvolution(inputFrameSize, outputFrameSize, kW)
    # pytorch: nn.Conv1d(in_channels, out_channels, kernel_size)
    # torch input:   [B x Seq x CH] (CH = inputFramSize)
    # pytorch input: [B x CH x Seq] (CH = in_channels)
    # N is a batch size, Cin denotes a number of channels, L is a length of signal sequence.
    # input of TemporalMaxPooling of torch: [B x Seq x CH]
    # input of MaxPool1d of pytorch: [B x CH x Seq]

    layers = []
    ## [B x CH x Seq] = B x alphasize x 201 (torch: 201 x alphasize)
    layers.append(nn.Conv1d(in_channels=alphasize, out_channels=384, kernel_size=4))
    #layers.append(nn.Threshold(threshold=1e-6, value=0)) #torch default value
    layers.append(nn.ReLU()) #torch default value
    layers.append(nn.MaxPool1d(kernel_size=3, stride=3))
    ## [B x CH x Seq] = B x 384 x 66
    layers.append(nn.Conv1d(384, 512, 4))
    #layers.append(nn.Threshold(1e-6, 0))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool1d(3,3))
    ## [B x CH x Seq] = B x 512 x 21
    layers.append(nn.Conv1d(512, cnn_dim, 4))
    #layers.append(nn.Threshold(1e-6, 0))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool1d(3,2))
    ## [B x CH x Seq] = B x 256 x 8

    self.cnns = nn.Sequential(*layers)
    #batch_first=True: [B x Seq x CH]
    #for i in range(1, 8):
    self.fixedRNN = nn.RNN(input_size=256, hidden_size=cnn_dim, bias=True, nonlinearity='relu', batch_first=True)
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(cnn_dim, emb_dim)

  def forward(self, x):
    x = torch.transpose(x, 2, 1) # [B x Seq x CH] -> [B x CH x Seq] = [40, 201, 70] -> [40, 70, 201]
    x = self.cnns(x)
    x = torch.transpose(x, 2, 1) # [B x CH x Seq] -> [B x Seq x CH] B x 8 x 256
    hidden = torch.zeros(1, x.size(0), 256).cuda() # 256 is cnn_dim
    for i in range(1, 8):
        out, hidden = self.fixedRNN(x[:,i,:].unsqueeze(1), hidden)
    #out, hidden = self.fixedRNN(x, hidden)
    
    out = out[:, -1, :] #This value can be modified. (batch x 256)
    out = self.dropout(out)
    out = self.fc(out)
    return out

