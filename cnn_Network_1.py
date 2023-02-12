
# Beispiel from https://nextjournal.com/gkoehler/pytorch-mnist

#%% 

# PyTorch has dynamic execution graphs, meaning the computation graph is created on the fly.
import torch
import torchvision
import os

# Here the number of epochs defines how many times we'll loop over the complete training dataset,
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
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape

import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])  
fig.show()

# Building the Network
#  We'll use two 2-D convolutional layers 
# followed by two fully-connected (or linear) layers. 
# As activation function we'll choose rectified linear units (ReLUs in short)
#  and as a means of regularization we'll use two dropout layers.
# %% 
# import the necessary packages for building the Network layers
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn.functional as F
import torch.optim as optim

#%%
#numChannels: The number of channels in the input images (1 for grayscale or 3for RGB)
numChannels = 1
#classes: Total number of unique class labels in our dataset
classes = 10

###########################################################################
# filter_size:  the height and width of the 2D convolution filter matrix
filter_size  = 5
# filters_num_1 : total number of filters in first 2D convolution layer 
filters_num_1 = 20
# filters_num_2 : total number of filters in second 2D convolution layer 
filters_num_2 = 50
# in_features (int) – size of each input sample in torch.nn.Linear
in_features = 800
# out_features (int) – size of each output sample in torch.nn.Linear
out_features = 500

# %%
class Net(Module):
    def __init__(self, numChannels, classes, filter_size, filters_num_1, filters_num_2):
	    super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=filters_num_1,kernel_size=filter_size)
        # A ReLU activation function is then applied, 
        # followed by a 2×2 max-pooling layer with a 2×2 stride
        #  to reduce the spatial dimensions of our input image.
        self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=2, stride=2)
        #2 layer     
		# initialize second set of CONV => RELU => POOL layers
        # 20 filters
		self.conv2 = Conv2d(in_channels= filters_num_1, out_channels=filters_num_2,kernel_size=filter_size)
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=2, stride=2)   

        # Fully connected layers       
		# initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=800, out_features=500)
		self.relu3 = ReLU()
		# initialize our softmax classifier
        
		self.fc2 = Linear(in_features=500, out_features=classes) 
		self.logSoftmax = LogSoftmax(dim=1)
                
    def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
                
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
                
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.relu3(x)
                
		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)
		# return the output predictions
		return output
      

# %%
