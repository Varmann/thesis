import matplotlib.pyplot as plt
import numpy as np
import random

# def plot_img_and_mask(img, mask):    
#     classes = mask.max() + 1
#     fig, ax = plt.subplots(1, classes + 1)
#     plt.style.use('grayscale')
#     ax[0].set_title('Input image')
#     ax[0].imshow(img)
#     for i in range(classes):
#         ax[i + 1].set_title(f'Mask (class {i + 1})')
#         ax[i + 1].imshow(mask == i)
#     plt.xticks([]), plt.yticks([])   
#     plt.show()


def plot_img_and_mask(img, mask):
    fig, ax = plt.subplots(1, 2 ,facecolor = "lightgrey")
    [ax_i.set_axis_off() for ax_i in ax.ravel()]
    plt.style.use('grayscale')        
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Mask') 
    ax[1].imshow(mask)     
   
    plt.show()

def plot_img_and_mask_save(img, mask ,filename):   
    fig, ax = plt.subplots(1, 2,facecolor = "lightgrey", dpi = 200)
    [ax_i.set_axis_off() for ax_i in ax.ravel()]   
    plt.style.use('grayscale')
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Mask')
    ax[1].imshow(mask) 
    plt.savefig(filename,facecolor='lightgrey')    
    plt.show()


def plot_img_and_mask_save_3(img, mask_padding , mask_no_padding, filename):   
    fig, ax = plt.subplots(1, 3,facecolor = "lightgrey", dpi = 200)
    [ax_i.set_axis_off() for ax_i in ax.ravel()]   
    plt.style.use('grayscale')
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Mask padding')
    ax[1].imshow(mask_padding) 
    ax[2].set_title('Mask no padding')
    ax[2].imshow(mask_no_padding)
    plt.savefig(filename,facecolor='lightgrey')    
    plt.show()


# Crop image with padding 
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
    croped_image  = crop(image,y,crop_size, x,crop_size)
    croped_mask  = crop(mask,y,crop_size, x,crop_size)
    
    #plot_img_and_mask(croped_image, croped_mask)

    # Random Rotate at 90 degree
    k_Random = random.randint(0 , 3)         
    #print("Random Degree", k_Random * 90)

    img = np.rot90(croped_image, k_Random)
    msk = np.rot90(croped_mask, k_Random)
    
    if(croped_image.shape[0] != crop_size | croped_image.shape[1] != crop_size ):
        raise TypeError("Error : Croped Image Shape is not equal to Crop size !" ,croped_image.shape )
    elif(croped_mask.shape[0] != crop_size | croped_mask.shape[1] != crop_size ):
        raise TypeError("Error : Croped Mask Shape is not equal to Crop size Error !" ,croped_mask.shape )   
    return img, msk




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
