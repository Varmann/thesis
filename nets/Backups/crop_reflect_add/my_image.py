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


# crop image 
# image_teil = image[y:y+h, x:x+w]
def crop(image, y,h, x, w) :
    image_teil = image[y:y+h, x:x+w]
    return image_teil


# Pads with the reflection of the vector 
# mirrored on the first and last values of the vector along each axis.
def reflect(img_np_array:np.ndarray,  up:int , down:int, right:int, left:int):
    # 3rd Dimension is the Gray/RGB values, do not reflect.
    if(img_np_array.ndim == 3) :
        img_reflected = np.pad(img_np_array, ((up, down),(right, left),(0,0)), mode='reflect')
        return  img_reflected
    else:
        raise TypeError("Array must be 3 dimensional!")
   
    