# Beispiel from https://nextjournal.com/gkoehler/pytorch-mnist

#%% Cell 1 - Dataset initiieren

# PyTorch has dynamic execution graphs, meaning the computation graph is created on the fly.
import torch
import torchvision
import os
import random 
import matplotlib.pyplot as plt

# Here the number of epochs defines how many times we'll loop over the complete training dataset,
# while learning_rate and momentum are hyperparameters for the optimizer we'll be using later on.
n_epochs = 3
batch_size_train = 64
batch_size_test = 10
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
    batch_size=batch_size_train, shuffle=False)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=False)

test_loader_without_normalize = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                ])),
    batch_size=batch_size_test, shuffle=False)

 #%% Cell 2 - random sample laden und auswerten

examples = enumerate(test_loader) #The enumerate() function adds a counter to an iterable and returns it (the enumerate object).
batch_idx, (example_data, example_targets) = next(examples) # Returns next item from iterator.
examples_length = len(example_data)

print(example_data.shape)

colormap = 'Blues'
fig = plt.figure()
for i in range(9):
  random_int = random.randint(0,examples_length)
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[random_int][0], cmap=colormap, interpolation='none')
  plt.title("Test Data : {}".format(example_targets[random_int]))
  plt.xticks([])
  plt.yticks([])  
#fig.show()

#%%
# histogram of one image from trainloader with normalization
# data range is ~ -0.4 to ~ 2.8
test_loader_iter = iter(test_loader)
example_data, example_targets = next(test_loader_iter)
first_image = example_data[0,:,:,:]  # 1 x 28 x 28 = 1 * 28 * 28 = 784
first_image = example_data[0,0,:,:]  #     28 x 28 
print(first_image.shape)
plt.hist(x=first_image.flatten(), bins=10)


# histogram of first image without normalization
# data range is [0 - 1]
test_loader_iter_without_normalize = iter(test_loader_without_normalize)
example_data_without_normalize, example_targets_without_normalize = next(test_loader_iter_without_normalize)
first_image_without_normalize = example_data_without_normalize[0]
print(first_image_without_normalize.shape)
plt.hist(x=first_image_without_normalize.flatten(), bins=10)











#%%
                      # shape = 10 x 1 x 28 x 28
first_image = example_data[1,:,:,:]   # shape = 1 x 28 x 28
first_image_flattened = first_image.flatten()
first_image_flattened = first_image.reshape((-1))
first_image_flattened = first_image.reshape((784))
plt.hist(x=first_image_flattened, bins=10)

 
#%%
print("Example X type : {}".format(type(example_data)))
print("Example X Shape: {}".format(example_data.shape))
print("Example Y type : {}".format(type(example_targets)))
print("Example Y Shape: {}".format(example_targets.shape))

#%%
random_int = random.randint(0,examples_length)
plt.imshow(example_data[random_int][0], cmap=colormap, interpolation='none')
plt.title(" Random Test Data : {}".format(example_targets[random_int]))
#%%
print("Random Data Type : {}".format(type(example_data[random_int][0])))
print("Random Data Shape : {}".format(example_data[random_int][0].shape))
print("Random Data Max Value: {}".format(torch.max(example_data[random_int][0])))
print("Random Data Min Value: {}".format(torch.min(example_data[random_int][0])))


# %%
import numpy as np 
random_int = 0
np_array = np.array(first_image_without_normalize.reshape((28,28)))*255
print("Type : ",type(np_array), "Shape : ", np_array.shape)
plt.imshow(np_array, cmap=colormap, interpolation='none',)
plt.title(" Matrix Multiplikation : {}".format(example_targets[random_int]))
plt.colorbar()
#%%
print("Random Data Max Value: {}".format(np.max(np_array)))
print("Random Data Min Value: {}".format(np.min(np_array)))

# %%
array_transpose = np.transpose(np_array)
plt.imshow(array_transpose, cmap=colormap, interpolation='none')
plt.title(" Matrix Transpose : {}".format(example_targets[random_int]))
# %%
array_shift = np.roll(np_array, 7, axis=1)
plt.imshow(array_shift, cmap=colormap, interpolation='none')
plt.title(" Matrix Schift  : {}".format(example_targets[random_int]))
# %%
