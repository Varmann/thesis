from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

# read image
def read(image_path: str | Path) -> np.array:
    """returns an image array """
    from PIL import Image
    # 8 Bit Grafik hat Werte zwischen 0 und 255 (Integer). Wenn man durch 255 teilt, erhält man Float Werte zwischen 0 und 1
    image_array = np.asarray(Image.open(image_path)) / 255
    return image_array


# Crop Image with Tile_Width and Tile_Padding
# Returns List of croped images and the numbers of horizontal und vertical crops.
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
    r  = image_Width // Tile_Width

    up = Tile_Padding
    down  = (d+1)* Tile_Width - image_Height + Tile_Padding
    left = Tile_Padding
    right = (r+1)*Tile_Width - image_Width + Tile_Padding

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

    v = (image_Height// Tile_Width) + 1
    h = (image_Width// Tile_Width) + 1 
  
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
def crop(image, y,height, x, width) :
    image_teil = image[y:y+height, x:x+width]
    return image_teil


# Pads with the reflection of the vector 
# mirrored on the first and last values of the vector along each axis.
def reflect(img_np_array:np.ndarray,  up:int , down:int, left:int , right:int,):
    # 3rd Dimension is the Gray/RGB values, do not reflect.
    if(img_np_array.ndim == 3) :
        img_reflected = np.pad(img_np_array, ((up, down),(left, right),(0,0)), mode='reflect')
        return  img_reflected
    else:
        raise TypeError("Array must be 3 dimensional!")
   

# Pads with the reflection of the vector 
# mirrored on the first and last values of the vector along each axis.
def reflect_mnist(img_np_array:np.ndarray,  up:int , down:int, right:int, left:int):
    # 3rd Dimension is the Gray/RGB values, do not reflect.
    if(img_np_array.ndim == 2) :
        img_reflected = np.pad(img_np_array, ((up, down),(left, right)), mode='reflect')
        return  img_reflected
    else:
        raise TypeError("Array must be 2 dimensional!")



