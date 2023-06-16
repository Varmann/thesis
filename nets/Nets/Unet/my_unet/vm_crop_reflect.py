#%%
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random


def plot_img_and_mask(img, mask ):    
    fig, ax = plt.subplots(1, 2)
    plt.style.use('grayscale')
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Mask')
    ax[1].imshow(mask) 
    plt.show()


def crop_with_padding(image_np_array:np.ndarray, Tile_Width :int, Tile_Padding:int):
    if(Tile_Width <= 0):
        raise TypeError("Tile_Width must be postive number !")
    if(Tile_Padding <= 0 ):
        raise TypeError("Tile_Padding must be postive number !")
    
    if(Tile_Width < Tile_Padding):
         raise TypeError("Tile width must be bigger than Tile padding !")

    ######################################################################################
    # Find Crop Width 
    # Match the image Height and Width to Crop_Width
    Crop_width = Tile_Width + 2 * Tile_Padding
    #Original image Shape
    shape = image_np_array.shape
    image_Height = shape[0]
    image_Width = shape[1]

    # Making Image through reflection bigger, image to be divisible with Crop_width
    d  = image_Height // Tile_Width
    d_m = image_Height  % Tile_Width
    r  = image_Width  // Tile_Width
    r_m = image_Width  % Tile_Width

    v  = d if d_m == 0 else d+1
    h  = r if r_m == 0 else r+1

    up = Tile_Padding
    down  = v* Tile_Width - image_Height + Tile_Padding
    left = Tile_Padding
    right = h*Tile_Width - image_Width +Tile_Padding

    # Reflect
    if(image_np_array.ndim == 2) :
         image_reflected = np.pad(image_np_array, ((up, down),(left, right)), mode='reflect')
    # 3rd Dimension is the Gray/RGB values, do not reflect.
    elif(image_np_array.ndim == 3) :
        image_reflected = np.pad(image_np_array, ((up, down),(left , right),(0,0)), mode='reflect')    
    else:
        raise TypeError("Array must be 2(Gray) or 3(RGB) dimensional!")
    
    ######################################################################################
    # Crop Image
    croped_images = []
    shape = image_reflected.shape
    image_Height = shape[0]
    image_Width = shape[1]

    number_crops  = [h,v]

    for i in range (0,v):   
        for j in range (0,h):
            # Crop image
            y =  i*Tile_Width
            x =  j*Tile_Width
            image_teil = image_reflected[y:y+Crop_width, x:x+Crop_width]
            croped_images.append( image_teil ) 
    # Return Array of Croped Images
    return croped_images , number_crops

# crop image 
# y - Height, x - width
# image_teil = image[y:y+h, x:x+w]
def crop(image:np.ndarray, y:int,height:int, x:int, width:int) :
    if(image.ndim != 2) :
        raise TypeError("Array must be 2 dimensional!")
    if(y < 0):
        raise TypeError("To crop the image the start Pixel must be postive number !")
    if(height < 0):
        raise TypeError("The crop height must be postive number !")
    if(x < 0):
        raise TypeError("To crop image the start Pixel must be postive number !")
    if(width < 0):
        raise TypeError("The crop width must be postive number !")
    image_teil = image[y:y+height, x:x+width]
    return image_teil


def random_crop_rotate90(image, mask,crop_size, Row_min, Row_max, Column_min ,Column_max):    
    #print("********** Image and Mask shapes *********")
    #print(image.shape, mask.shape)
    #print("******************************************")
    
    # image size 1920x1080         
    Row_random = random.randint(Row_min , Row_max)   
    Column_random = random.randint(Column_min , Column_max)
    #print("Random Column ,Row")
    #print(Row_random,Column_random)
    
    y = Row_random
    x = Column_random
    croped_image  = image[y:y+crop_size, x:x+crop_size]
    croped_mask  = mask[y:y+crop_size, x:x+crop_size]
    
    # Random Rotate at 90 degree
    #k_Random = random.randint(0 , 3)         
    k_Random = 0
    #print("Random Degree", k_Random * 90)

    img = np.rot90(croped_image, k_Random)
    msk = np.rot90(croped_mask, k_Random)
    
    return img, msk , [Row_random,Column_random, k_Random]


# read image
def read_image(image_path: str | Path) -> np.array:
    """returns an image array """
    from PIL import Image
    # 8 Bit Grafik hat Werte zwischen 0 und 255 (Integer). Wenn man durch 255 teilt, erhält man Float Werte zwischen 0 und 1
    image_array = np.asarray(Image.open(image_path)) / 255
    return image_array 

# Constants
Tile_Width = 200 
Tile_Padding  = 50
Crop_width = Tile_Width + 2 * Tile_Padding
# Row
Row_min = 350
Row_max  = 350 + 1400 
#Column 
Column_min = 230 
Column_max  = 230 + 600 

import random
image_path =  r"C:/Users/vmanukyan/Desktop/Fluidized/Croped/HD/Raw/"
image_name = r"frame1.png"

mask_path =  r"C:/Users/vmanukyan/Desktop/Fluidized/Croped/HD/Segmented/"
mask_name = r"frame1-1.png"
#image_name = r"grid.jpg"
img = np.asarray(Image.open(image_path+image_name))
msk = np.asarray(Image.open(mask_path+mask_name))
#plt.imshow(img)


#crop image                
y = Row_min
x = Column_min
image  = img[y:y+1400, x:x+600]
mask  =  msk[y:y+1400, x:x+600] 

#Reflect / Crop 
croped_images ,number_crops = crop_with_padding(image, Tile_Width, Tile_Padding)       
h = number_crops[0] #horizontal
v = number_crops[1] #vertical
#Add together
for i in range (0,v):   
    image_add  = crop(croped_images[h*i], Tile_Padding,Tile_Width,Tile_Padding,Tile_Width)
    for j in range (1,h):
        img_crop = crop(croped_images[h*i+j], Tile_Padding,Tile_Width,Tile_Padding,Tile_Width)
        image_add = np.concatenate((image_add,img_crop), axis=1)    
    if(i==0):
        new_image = image_add    
    else:
        new_image = np.concatenate((new_image,image_add), axis=0)

# Crop real part of mask
result  = Image.fromarray(crop(new_image,0,1400,0, 600))


#plot_img_and_mask(image,new_image)

#plt.imshow(result)



#%%
print ( "Croped Image Part size : " ,Crop_width ," x " ,Crop_width )
print ( "Tile width : " , Tile_Width  )
print ("Tile Padding : " ,Tile_Padding  )
print( "Croped Images : ", len(croped_images) )

#h = (image_Width// Tile_Width)+1
#v = (image_Height// Tile_Width)+1
h = number_crops[0] #horizontal
v = number_crops[1] #vertical

fig, axes= plt.subplots(v, h, figsize= (h,v))
[ax_i.set_axis_off() for ax_i in axes.ravel()]
for i in range (0,v):  
    for j in range (0,h):        
        axes[i,j].imshow(croped_images[h*i+j])   


#%% Crop real parts from reflected images
print("Show crop parts of reflected image")
print ( "Part size : " , Tile_Width ," x " , Tile_Width )
print("Parts without padding")

fig, axes= plt.subplots(v, h, figsize= (h,v))
[ax_i.set_axis_off() for ax_i in axes.ravel()]
for i in range (0,v):   
    for j in range (0,h):        
        img_crop = crop(croped_images[h*i+j], Tile_Padding,Tile_Width,Tile_Padding,Tile_Width)
        axes[i,j].imshow(img_crop)    

#%%
#Add together
for i in range (0,v):   
    image_add  = croped_images[h*i]
    for j in range (1,h):
        img_crop = croped_images[h*i+j]
        image_add = np.concatenate((image_add,img_crop), axis=1)    
    if(i==0):
        new_image = image_add    
    else:
        new_image = np.concatenate((new_image,image_add), axis=0)

plt.imshow(new_image)
#%%
#Add together
for i in range (0,v):   
    image_add  = crop(croped_images[h*i], Tile_Padding,Tile_Width,Tile_Padding,Tile_Width)
    for j in range (1,h):
        img_crop = crop(croped_images[h*i+j], Tile_Padding,Tile_Width,Tile_Padding,Tile_Width)
        image_add = np.concatenate((image_add,img_crop), axis=1)    
    if(i==0):
        new_image = image_add    
    else:
        new_image = np.concatenate((new_image,image_add), axis=0)

plot_img_and_mask(image, crop(new_image,0,1400,0,600))

#%%
plot_img_and_mask(img, msk )

#%%
crop_size = 300
# Random Row 
Row_min = 350
Row_max  = 350 + 1400 - crop_size
## Random Column 
Column_min = 230 
Column_max  = 230 +  600 - crop_size 

print(Row_min, Row_max, Column_min ,Column_max)

#%%
croped_image, croped_mask , Randoms =  random_crop_rotate90(img, msk, crop_size, Row_min, Row_max, Column_min ,Column_max)       
text  = str(Randoms[0]) + "," +str(Randoms[1]) + "," + str(Randoms[2]) 
print(text)
plot_img_and_mask(croped_image, croped_mask)
croped_image = croped_image[None , : ]

# %%
plt.imshow(image)
# %%
# Crop image with padding 
def crop_without_padding(image_np_array:np.ndarray, Tile_Width :int):
    if(Tile_Width <= 0):
        raise TypeError("Tile_Width must be postive number !")
    ######################################################################################
    # Find Crop Width 
    # Match the image Height and Width to Crop_Width
    Crop_width = Tile_Width
    #Original image Shape
    shape = image_np_array.shape
    image_Height = shape[0]
    image_Width = shape[1]

    # Making Image through reflection bigger, image to be divisible with Crop_width
    d  = image_Height // Tile_Width
    d_m = image_Height  % Tile_Width
    r  = image_Width  // Tile_Width
    r_m = image_Width  % Tile_Width

    v  = d if d_m == 0 else d+1
    h  = r if r_m == 0 else r+1

    up = 0
    down  = v* Tile_Width - image_Height
    left = 0
    right = h*Tile_Width - image_Width

    # Reflect
    if(image_np_array.ndim == 2) :
         image_reflected = np.pad(image_np_array, ((up, down),(left, right)), mode='reflect')
    # 3rd Dimension is the Gray/RGB values, do not reflect.
    elif(image_np_array.ndim == 3) :
        image_reflected = np.pad(image_np_array, ((up, down),(left , right),(0,0)), mode='reflect')    
    else:
        raise TypeError("Array must be 2(Gray) or 3(RGB) dimensional!")
    
    ######################################################################################
    # Crop Image
    croped_images = []
    shape = image_reflected.shape
    image_Height = shape[0]
    image_Width = shape[1]

    number_crops  = [h,v]

    for i in range (0,v):   
        for j in range (0,h):
            # Crop image
            y =  i*Tile_Width
            x =  j*Tile_Width
            image_teil = image_reflected[y:y+Crop_width, x:x+Crop_width]
            croped_images.append( image_teil ) 
    # Return Array of Croped Images
    return croped_images , number_crops
