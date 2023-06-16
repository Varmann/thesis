#%%
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import my_image

#%%
image_path =  r"C:/Users/vmanukyan/Desktop/Notizen_BA/Test_Images/"
image_name = r"bird.jpg"

img = my_image.read(image_path+image_name)
plt.imshow(img)
plt.axis('off')

################ Numpy array
image_np_array = np.array(img)
# Original image
image_width = 1000
image_height  = 500
croped_images = []

crop_width = 200
crop_height = 200
tile_padding = 50


#%%

# Image extended
extend_width = crop_width - tile_padding
extend_height = crop_height - tile_padding
image_np_array = np.array(img)
image_np_array_1 = np.array(img)
image_np_array_2 = np.array(img)



#%%
a = np.pad(image_np_array, ((0,0),(0,0),(0,0)), mode='reflect')
a = np.pad(a, ((0,0),(0,0),(0,0)), mode='reflect')
plt.imshow(a)

#%%

image_width = 1200
image_height  = 600
croped_images = []

crop_width = 200
crop_height = 200
tile_padding = 50
x = (image_width-crop_width) // tile_padding
y = (image_height- crop_height) // tile_padding

fig, axes= plt.subplots(y, x, figsize= (16,8))
[ax_i.set_axis_off() for ax_i in axes.ravel()]
for i in range (0,y):   
    for j in range (0,x):
        #axs[i,j].imshow(image.crop_image(img ,i*teil_image, crop_height,j*teil_image,crop_width))  
        croped_images.append(my_image.crop(image_np_array ,i*tile_padding, crop_height,j*tile_padding,crop_width)) 
        axes[i,j].imshow(croped_images[x*i+j])    


 
print ( "Croped Image size : " ,crop_width ," x " ,crop_height )
print ("Tile Padding : " ,tile_padding  )
print( "Croped Images : ", len(croped_images) )

#%%
for Index in range(0,150):    
    image_np_array = np.insert(image_np_array[0:500], 999+Index ,[0,0,0], axis = 1)
print(image_np_array.shape)
plt.imshow(image_np_array)


for Index in range(0,150):    
    image_np_array = np.insert(image_np_array[0:500], 0+Index ,[0,0,0], axis = 1)

plt.imshow(image_np_array)
print(image_np_array.shape)
#%%

concat_array = image_np_array[0:150]
concat_array_reverse = concat_array[::-1]
image_np_array = np.concatenate((concat_array_reverse,image_np_array), casting="same_kind")  

print(image_np_array.shape)    
plt.imshow(image_np_array)


# %%
concat_array = image_np_array[501:650]
concat_array_reverse = concat_array[::-1]
image_np_array = np.concatenate((image_np_array,concat_array_reverse), casting="same_kind")  

print(image_np_array.shape)    
plt.imshow(image_np_array)

# %%

image_width = 1300
image_height  = 800
croped_images = []

crop_width = 200
crop_height = 200
tile_padding = 50
x = (image_width-crop_width) // tile_padding
y = (image_height- crop_height) // tile_padding

fig, axes= plt.subplots(y, x, figsize= (16,8))
[ax_i.set_axis_off() for ax_i in axes.ravel()]
for i in range (0,y):   
    for j in range (0,x):
        #axs[i,j].imshow(image.crop_image(img ,i*teil_image, crop_height,j*teil_image,crop_width))  
        croped_images.append(my_image.crop(image_np_array ,i*tile_padding, crop_height,j*tile_padding,crop_width)) 
        axes[i,j].imshow(croped_images[x*i+j])    


 
print ( "Croped Image size : " ,crop_width ," x " ,crop_height )
print ("Tile Padding : " ,tile_padding  )
print( "Croped Images : ", len(croped_images) )

# %%


# %%
image_width = 1000
image_height  = 500
croped_images = []

crop_width = 200
crop_height = 200
tile_padding = 50
x = (image_width-crop_width) // tile_padding
y = (image_height- crop_height) // tile_padding

fig, axes= plt.subplots(y, x, figsize= (16,8))
[ax_i.set_axis_off() for ax_i in axes.ravel()]
for i in range (0,y):   
    for j in range (0,x):
        #axs[i,j].imshow(image.crop_image(img ,i*teil_image, crop_height,j*teil_image,crop_width))  
        croped_images.append(my_image.crop(img ,i*tile_padding, crop_height,j*tile_padding,crop_width)) 
        axes[i,j].imshow(croped_images[x*i+j])    


 
print ( "Croped Image size : " ,crop_width ," x " ,crop_height )
print ("Tile Padding : " ,tile_padding  )
print( "Croped Images : ", len(croped_images) )


# %%