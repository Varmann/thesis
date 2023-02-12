import torch
import torch.nn as nn
import torch.nn.functional as F


class BobNet(torch.nn.Module):
  def __init__(self):
    super(BobNet, self).__init__()
    self.linear1 = nn.Linear(784, 128, bias=False)
    self.linear2 = nn.Linear(128, 10, bias=False)
    self.logsoftmax = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = x.reshape((-1, 784))
    # x should have 784 and not 28 x 28
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = self.logsoftmax(x)
    return x
