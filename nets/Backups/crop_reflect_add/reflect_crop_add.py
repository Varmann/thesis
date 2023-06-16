#%% Imports

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import my_image

#%% Constants
Tile_width = 200 
Tile_padding  = 25
Crop_width = Tile_width + 2 * Tile_padding


#%% Read Image

image_path =  r"C:/Users/vmanukyan/Desktop/Notizen_BA/Test_Images/"
#image_name = r"bird.jpg"
image_name = r"grid.jpg"
img = my_image.read(image_path+image_name)


#%% Convert Image to numpy array
image_np_array = np.array(img)
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
image_reflected = my_image.reflect(image_np_array, up, down, left , right)

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


 
print ( "Croped Image size : " ,Crop_width ," x " ,Crop_width )
print ( "Tile width : " , Tile_width  )
print ("Tile Padding : " ,Tile_padding  )
print( "Croped Images : ", len(croped_images) )


#%% Crop real parts from reflected images

x = Width_reflected // Tile_width 
y = Height_reflected // Tile_width

fig, axes= plt.subplots(y, x, figsize= (16,8))
[ax_i.set_axis_off() for ax_i in axes.ravel()]
for i in range (0,y):   
    for j in range (0,x):
        img_crop = my_image.crop(croped_images[x*i+j], Tile_padding,Tile_width,Tile_padding,Tile_width)
        axes[i,j].imshow(img_crop)    

# %%
for i in range (0,y):   
    image_add  = my_image.crop(croped_images[x*i], Tile_padding,Tile_width,Tile_padding,Tile_width)
    for j in range (1,x):
         img_crop = my_image.crop(croped_images[x*i+j], Tile_padding,Tile_width,Tile_padding,Tile_width)
         image_add = np.concatenate((image_add,img_crop), axis=1)
    if(i==0):
        new_image = image_add    
    else:
        new_image = np.concatenate((new_image,image_add), axis=0)

plt.imshow(new_image)
# %%
