
import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
  def __init__(self, input_size, hid_size, noop, random_init=True):
    super(ImageEncoder, self).__init__()
    self.noop = noop
    self.fc = nn.Linear(input_size, hid_size)

  def forward(self, x):
    if self.noop !=1:
      x = self.fc(x)

    return x


