
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

   
Images_Number  = example_data.shape[0]
print(Images_Number)

#%% 
# Importing Image module from PIL package 
from PIL import Image 
import matplotlib.pyplot as plt
t2 =0
import numpy as np
for i in range(3):
    Image_Save_Path = r"C:\Users\Vardan\Desktop\Save_Mnist_Images"
    print(Image_Save_Path)
    Image_Save_Name ="Image_" +str(i+1) + "_Number_" +  str(format(example_targets[i])) + ".jpg"
    print(Image_Save_Name)
    image = Image.new("RGB",[28,28])
    plt.imshow(image, cmap='Blues', interpolation='none')
 
#%%
print(str(example_targets[i]))
print(example_targets[i])
print(str(format(example_targets[i])))


#%%
fig = plt.figure()
plt.tight_layout()
plt.imshow(example_data[i][0], cmap='Blues', interpolation='none')
plt.title("Ground Truth: {}".format(example_targets[i]))
plt.xticks([])
plt.yticks([])  





# %%
arr = np.array(example_data[i][0]).astype(dtype='uint8')
img = Image.fromarray(arr, 'RGB')
img.save(Image_Save_Path+Image_Save_Name)