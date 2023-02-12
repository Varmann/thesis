import torch
import torch.nn as nn

# TODO : Gleche Auflösung für alle Kernels, durch entsprechende Conv2D einstellung
# -> Conv2D kernel size variieren, OHNE dass Auflösung reduziert wird

# Idee: filter_number verändern, nicht 20 sondern z.B. 100

# Idee: mehr layers, LeNet mit über 2 layers

# Idee: Maxpool weg lassen oder andere einstellungen (siehe Dokumentation!)



class LeNet(nn.Module):
    def __init__(self, filter_size):
        super(LeNet, self).__init__()
        print("CNN is initialised with Kernel Size of ", filter_size)
        # ─── Parameters ───────────────────────────────────────────────

        #numChannels: The number of channels in the input images (1 for grayscale or 3for RGB)
        numChannels = 1
        # filter_size:  the height and width of the 2D convolution filter matrix
        ###filter_size  = 5
        # filters_num_1 : total number of filters in first 2D convolution layer 
        filters_num_1 = 20  
        # filters_num_2 : total number of filters in second 2D convolution layer 
        filters_num_2 = filters_num_1 + 30
        # in_features (int) – size of each input sample in torch.nn.Linear
        next_layer_size = filters_num_2*((((28-filter_size+1)//2)-filter_size+1)//2)**2
        linear_in_features = next_layer_size 
        # out_features (int) – size of each output sample in torch.nn.Linear
        linear_out_features = linear_in_features//2 
        # classes: Total number of unique class labels in our dataset
        classes = 10

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(numChannels, filters_num_1, kernel_size=filter_size)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(filters_num_1, filters_num_2, kernel_size=filter_size)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=linear_in_features, out_features=linear_out_features)
        self.relu3 = nn.ReLU()
		# initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=linear_out_features, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

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
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
                
		# pass the output to our softmax classifier to get our output
		# predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
		# return the output predictions
        return output
