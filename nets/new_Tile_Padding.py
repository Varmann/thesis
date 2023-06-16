#%% Imports

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import my_image
import random

#%% Constants
Tile_Width = 200 
Tile_Padding  = 50
Crop_width = Tile_Width + 2 * Tile_Padding


#%% Read Image

image_path =  r"C:/Users/vmanukyan/Desktop/Notizen_BA/Test_Images/"
image_name = r"bird.jpg"
#image_name = r"grid.jpg"
img = my_image.read(image_path+image_name)


#%% Convert Image to numpy array
print("Original Image")
image_np_array = np.array(img)
plt.imshow(image_np_array)
#plt.axis('off')

#%%
#Original image Shape
shape = image_np_array.shape
image_Height = shape[0]
image_Width = shape[1]

#%%

# Crop Image
croped_images ,number_crops = my_image.crop_with_padding(image_np_array, Tile_Width, Tile_Padding)
print ( "Croped Image Part size : " ,Crop_width ," x " ,Crop_width )
print ( "Tile width : " , Tile_Width  )
print ("Tile Padding : " ,Tile_Padding  )
print( "Croped Images : ", len(croped_images) )

#h = (image_Width// Tile_Width)+1
#v = (image_Height// Tile_Width)+1
h = number_crops[0] #horizontal
v = number_crops[1] #vertical

fig, axes= plt.subplots(v, h, figsize= (16,8))
#[ax_i.set_axis_off() for ax_i in axes.ravel()]
for i in range (0,v):   
    for j in range (0,h):        
        axes[i,j].imshow(croped_images[h*i+j])   


#%% Crop real parts from reflected images
print("Show crop parts of reflected image")
print ( "Part size : " , Tile_Width ," x " , Tile_Width )
print("Parts without padding")

fig, axes= plt.subplots(v, h, figsize= (16,8))
[ax_i.set_axis_off() for ax_i in axes.ravel()]
for i in range (0,v):   
    for j in range (0,h):        
        img_crop = my_image.crop(croped_images[h*i+j], Tile_Padding,Tile_Width,Tile_Padding,Tile_Width)
        axes[i,j].imshow(img_crop)    


# %% Add image parts
print("Croped parts of reflected Image added together")
for i in range (0,v):   
    image_add  = my_image.crop(croped_images[h*i], Tile_Padding,Tile_Width,Tile_Padding,Tile_Width)
    for j in range (1,h):
        img_crop = my_image.crop(croped_images[h*i+j], Tile_Padding,Tile_Width,Tile_Padding,Tile_Width)
        image_add = np.concatenate((image_add,img_crop), axis=1)    
    if(i==0):
        new_image = image_add    
    else:
        new_image = np.concatenate((new_image,image_add), axis=0)

plt.imshow(new_image)

# %%
crop_orig_image  = my_image.crop(new_image,0,image_Height,0, image_Width)

plt.imshow(crop_orig_image)
# %%

#Random Crop 

y = random.randint(0,(image_Height - Crop_width))
print(y) 
x = random.randint(0,(image_Width - Crop_width))
print(x)

random_crop = my_image.crop(image_np_array ,y,Crop_width,x,Crop_width)
print(random_crop.shape)

plt.imshow(random_crop)
# %%
