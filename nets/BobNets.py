import torch
import torch.nn as nn
import torch.nn.functional as F


class BobNet_2(torch.nn.Module):
  def __init__(self):
    super(BobNet_2, self).__init__()
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


class BobNet_3(torch.nn.Module):
  def __init__(self):
    super(BobNet_3, self).__init__()
    self.linear1 = nn.Linear(784, 400, bias=False)
    self.linear2 = nn.Linear(400, 100, bias=False)
    self.linear3 = nn.Linear(100, 10, bias=False)
    self.logsoftmax = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = x.reshape((-1, 784))
    # x should have 784 and not 28 x 28
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    x = self.linear3(x)
    x = self.logsoftmax(x)
    return x



class BobNet_4(torch.nn.Module):
  def __init__(self):
    super(BobNet_4, self).__init__()
    self.linear1 = nn.Linear(784, 500, bias=False)
    self.linear2 = nn.Linear(500, 300, bias=False)
    self.linear3 = nn.Linear(300, 100, bias=False)
    self.linear4 = nn.Linear(100, 10, bias=False)
    self.logsoftmax = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = x.reshape((-1, 784))
    # x should have 784 and not 28 x 28
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    x = self.linear3(x)
    x = F.relu(x)
    x = self.linear4(x)
    x = self.logsoftmax(x)
    return x


class BobNet_5(torch.nn.Module):
  def __init__(self):
    super(BobNet_5, self).__init__()
    self.linear1 = nn.Linear(784, 500, bias=False)
    self.linear2 = nn.Linear(500, 400, bias=False)
    self.linear3 = nn.Linear(400, 300, bias=False)
    self.linear4 = nn.Linear(300, 100, bias=False)    
    self.linear5 = nn.Linear(100, 10, bias=False)
    self.logsoftmax = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = x.reshape((-1, 784))
    # x should have 784 and not 28 x 28
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    x = self.linear3(x)
    x = F.relu(x)
    x = self.linear4(x)
    x = F.relu(x)
    x = self.linear5(x)
    x = self.logsoftmax(x)
    return x


class BobNet_6(torch.nn.Module):
  def __init__(self):
    super(BobNet_6, self).__init__()
    self.linear1 = nn.Linear(784, 500, bias=False)
    self.linear2 = nn.Linear(500, 400, bias=False)
    self.linear3 = nn.Linear(400, 300, bias=False)
    self.linear4 = nn.Linear(300, 200, bias=False)
    self.linear5 = nn.Linear(200, 100, bias=False)    
    self.linear6 = nn.Linear(100, 10, bias=False)
    self.logsoftmax = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = x.reshape((-1, 784))
    # x should have 784 and not 28 x 28
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    x = self.linear3(x)
    x = F.relu(x)
    x = self.linear4(x)
    x = F.relu(x)
    x = self.linear5(x)
    x = F.relu(x)
    x = self.linear6(x)
    x = self.logsoftmax(x)
    return x

class BobNet_7(torch.nn.Module):
  def __init__(self):
    super(BobNet_7, self).__init__()
    self.linear1 = nn.Linear(784, 600, bias=False)
    self.linear2 = nn.Linear(600, 500, bias=False)
    self.linear3 = nn.Linear(500, 400, bias=False)
    self.linear4 = nn.Linear(400, 300, bias=False)
    self.linear5 = nn.Linear(300, 200, bias=False)    
    self.linear6 = nn.Linear(200, 100, bias=False)
    self.linear7 = nn.Linear(100, 10, bias=False)
    self.logsoftmax = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = x.reshape((-1, 784))
    # x should have 784 and not 28 x 28
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    x = self.linear3(x)
    x = F.relu(x)
    x = self.linear4(x)
    x = F.relu(x)
    x = self.linear5(x)
    x = F.relu(x)
    x = self.linear6(x)
    x = F.relu(x)
    x = self.linear7(x)
    x = self.logsoftmax(x)
    return x


    