#%% 
# Beispiel from https://nextjournal.com/gkoehler/pytorch-mnist

import matplotlib.pyplot as plt
# PyTorch has dynamic execution graphs, meaning the computation graph is created on the fly.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from nets.Bobnet import BobNet
from nets.lenet import LeNet5, LeNet6
from nets.test_Nets import LeNet

# Here the number of epochs defines how many times we'll loop over the complete trai ning dataset,
# while learning_rate and momentum are hyperparameters for the optimizer we'll be using later on.
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# We'll use a batch_size of 64 for training and size 1000 for testing on this dataset. 
# The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean and standard deviation of the MNIST dataset

# ─── Load Datasets ────────────────────────────────────────────────────────────

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)

# Now let's take a look at some examples. We'll use the test_loader for this.
#%%
examples = enumerate(test_loader) #The enumerate() function adds a counter to an iterable and returns it (the enumerate object).
batch_idx, (example_data, example_targets) = next(examples) # Returns next item from iterator.

colormap = 'plasma'

fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap=colormap, interpolation='none')
  plt.title("Test File : {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])  
fig.show()

print("Example Date: {}".format(example_data.shape))
print("Example Targets: {}".format(example_targets.shape))
# Building the Network
#  We'll use two 2-D convolutional layers 
# followed by two fully-connected (or linear) layers. 
# As activation function we'll choose rectified linear units (ReLUs in short)
#  and as a means of regularization we'll use two dropout layers.


# %%  
# ─── Training The Model ───────────────────────────────────────────────────────


# network = LeNet5()
network = LeNet(3)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


# #On the x-axis we want to display the number of training examples the network has seen during training. 
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), r'C:\dev\result_cnn_mnist\model.pth')
            torch.save(optimizer.state_dict(), r'C:\dev\result_cnn_mnist\optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
#%%
test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

print( "Train Done")
#%%
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Number of training examples seen')
plt.ylabel('Negative log likelihood loss')
#%%
with torch.no_grad():
    output = network(example_data)

# %%
fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap=colormap, interpolation='none')
    plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
fig.show()

# %%
fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap=colormap, interpolation='none')
    plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
fig.show()
# %%
#TODO save in Datei Accuracy, Execution Time of Training
# TODO read Datei und Plot 