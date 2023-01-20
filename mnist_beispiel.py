
Pytorch Tutorial

Pytorch is a popular deep learning framework and it's easy to get started.

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10



     

First, we read the mnist data, preprocess them and encapsulate them into dataloader form.

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)



     

Then, we define the model, object function and optimizer that we use to classify.

class SimpleNet(nn.Module):
# TODO:define model




    
model = SimpleNet()

# TODO:define loss function and optimiter
criterion = 
optimizer = 



     

Next, we can start to train and evaluate!

# train and evaluate
for epoch in range(NUM_EPOCHS):
    for images, labels in tqdm(train_loader):
        # TODO:forward + backward + optimize
        
        
        
        
        
    # evaluate
    # TODO:calculate the accuracy using traning and testing dataset
    
    
    
    



     
Q5:

Please print the training and testing accuracy.
