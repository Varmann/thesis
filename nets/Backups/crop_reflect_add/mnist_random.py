# %%
# Beispiel from https://nextjournal.com/gkoehler/pytorch-mnist


#%% Imports
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random
import numpy as np
import my_image


#%% Load Dataset Mnist

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
    torchvision.datasets.MNIST(
        "/files/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_train,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/files/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_test,
    shuffle=True,
)

start = time.time()

examples = enumerate(
    test_loader
)  # The enumerate() function adds a counter to an iterable and returns it (the enumerate object).
batch_idx, (example_data, example_targets) = next(
    examples
)  # Returns next item from iterator.


# %% Load random Image from Dataset
dataset_lenght = len(train_loader.dataset)
random_int  = random.randint(0, dataset_lenght)
print(random_int)
#plt.imshow(train_loader.dataset.test_data[random_int])
# %%

#%% Import
#%% Constants
Tile_width = 3
Tile_padding  = 1
Crop_width = Tile_width + 2 * Tile_padding


#%% Read Image
original_image = train_loader.dataset.test_data[random_int]

#%% Convert Image to numpy array
image_np_array = np.array(original_image)
plt.imshow(image_np_array)
#plt.axis('off')

#%% Height and Widht of the Image

shape = image_np_array.shape
Height = shape[0]
Width = shape[1]

#%% Make Image through reflection bigger, image to be divisible with Crop_width
d  = Height // Crop_width 
r  = Width // Crop_width 

up = Tile_padding
down  = ( (d+1)*Crop_width - Height )
left = Tile_padding
right = ( (r+1)*Crop_width - Width )

#%%
#image_reflected = my_image.reflect(image_np_array, up, down, left , right)
image_reflected = my_image.reflect_mnist(image_np_array, 0, down+Tile_padding, 0 , right+Tile_padding)
print("Reflected Image")
plt.imshow(image_reflected)
#plt.axis('off')


# %% New Reflected Image Width and Height

shape_reflected = image_reflected.shape
Height_reflected = shape_reflected[0]
Width_reflected = shape_reflected[1]

#%% Crop Image

croped_images = []

x = Width_reflected // Tile_width 
y = Height_reflected // Tile_width

fig, axes= plt.subplots(y, x, figsize= (16,8))
#[ax_i.set_axis_off() for ax_i in axes.ravel()]
for i in range (0,y):   
    for j in range (0,x):
        x_pad = Tile_padding
        y_pad = Tile_padding
        if(i==0):
            y_pad = 0
        if(j==0):
            x_pad = 0
        croped_images.append(
                my_image.crop(
                    image_reflected ,
                    i*Tile_width - y_pad,Crop_width,
                    j*Tile_width - x_pad,Crop_width

                    #i*Tile_width, Crop_width,
                    #j*Tile_width,Crop_width
                            )
                            ) 
        
        axes[i,j].imshow(croped_images[x*i+j])    


 
print ( "Croped Image Part size : " ,Crop_width ," x " ,Crop_width )
print ( "Tile width : " , Tile_width  )
print ("Tile Padding : " ,Tile_padding  )
print( "Croped Images : ", len(croped_images) )


#%% Crop real parts from reflected images
print("Show crop parts of reflected image")
print ( "Part size : " , Tile_width ," x " , Tile_width )
print("Parts without padding")
x = Width_reflected // Tile_width 
y = Height_reflected // Tile_width

fig, axes= plt.subplots(y, x, figsize= (16,8))
[ax_i.set_axis_off() for ax_i in axes.ravel()]
for i in range (0,y):   
    for j in range (0,x):
        x_pad = Tile_padding
        y_pad = Tile_padding
        if(i==0):
            y_pad = 0
        if(j==0):
            x_pad = 0
        img_crop = my_image.crop(croped_images[x*i+j], y_pad,Tile_width,x_pad,Tile_width)
        axes[i,j].imshow(img_crop)    



# %% Add image parts
print("Croped parts of reflected Image added together")
for i in range (0,y):   
    y_pad = Tile_padding
    if(i==0):
        y_pad = 0
    image_add  = my_image.crop(croped_images[x*i], y_pad,Tile_width,0,Tile_width)
    #image_add  = my_image.crop(croped_images[x*i], Tile_padding,Tile_width,Tile_padding,Tile_width)
    for j in range (1,x):
        x_pad = Tile_padding       
        if(j==0):
            x_pad = 0  
        img_crop = my_image.crop(croped_images[x*i+j], y_pad,Tile_width,x_pad,Tile_width)
        image_add = np.concatenate((image_add,img_crop), axis=1)    
    if(i==0):
        new_image = image_add    
    else:
        new_image = np.concatenate((new_image,image_add), axis=0)

plt.imshow(new_image)

# %%
print("Croped Image from croped parts of reflected image")
new_image_original =  my_image.crop(new_image, 0,Height,0,Width)
plt.imshow(new_image_original)
# %%
