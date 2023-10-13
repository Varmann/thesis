import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

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
    ax[1].imshow(mask,vmin=0, vmax=1)
    plt.show()


def plot_img_imgmsk_mask_predicts_save(img, imgmsk,mask , predict,  predict_no_padding, filename, show = False):   
    fig, ax = plt.subplots(1, 5,facecolor = "lightgrey", dpi = 600)
    [ax_i.set_axis_off() for ax_i in ax.ravel()]   
    plt.style.use('grayscale')
    ### 
    ax[0].set_title('Image')
    ax[0].imshow(img,vmin=0, vmax=255)
    ###
    ax[1].set_title('Mask')
    ax[1].imshow(mask, vmin=0, vmax=1) 
    ##
    ax[2].set_title('Image+Predict')
    ax[2].imshow(imgmsk,vmin=0, vmax=255)    
    ###
    ax[3].set_title('Predict')
    #ax[3].imshow(predict)
    ax[3].imshow(predict, vmin=0, vmax=1)
    ###
    ax[4].set_title('No Padding')
    #ax[3].imshow(predict_no_padding)
    ax[4].imshow(predict_no_padding, vmin=0, vmax=1)
    ###
    plt.savefig(filename,facecolor='lightgrey',bbox_inches='tight')    
    if(show):
        plt.show()


def plot_img_imgmsk_predicts_save(img, imgmsk, predict,  predict_no_padding, filename, show = False):   
    fig, ax = plt.subplots(1, 4,facecolor = "lightgrey", dpi = 600)
    [ax_i.set_axis_off() for ax_i in ax.ravel()]   
    plt.style.use('grayscale')
    ### 
    ax[0].set_title('Input image')
    ax[0].imshow(img,vmin=0, vmax=255)
    #
    ax[1].set_title('Image+Predict')
    ax[1].imshow(imgmsk,vmin=0, vmax=255)
    ###
    ax[2].set_title('Predict')
    ax[2].imshow(predict, vmin=0, vmax=1)
    ###s
    ax[3].set_title('No Padding')
    ax[3].imshow(predict_no_padding, vmin=0, vmax=1)
    ###
    plt.savefig(filename,facecolor='lightgrey',bbox_inches='tight')    
    if(show):
        plt.show()


def plot_reflected_save(input_img,img, img_refl, refl_padd,  predict_padd, filename, show = False):   
    fig, ax = plt.subplots(1, 5,facecolor = "lightgrey", dpi = 600)
    [ax_i.set_axis_off() for ax_i in ax.ravel()]   
    plt.style.use('grayscale')
    #
    ax[0].set_title('Input')
    ax[0].imshow(input_img)
    ### 
    ax[1].set_title('Crop')
    ax[1].imshow(img)
    #
    ax[2].set_title('Reflected')
    ax[2].imshow(img_refl)
    ###
    ax[3].set_title('Padding')
    ax[3].imshow(refl_padd)
    ###s
    ax[4].set_title('Predict')
    ax[4].imshow(predict_padd)
    ###
    plt.savefig(filename,facecolor='lightgrey',bbox_inches='tight')    
    if(show):
        plt.show()


# Crop image with padding 
def crop_with_padding(image_np_array:np.ndarray, Tile_Width :int, Tile_Padding:int):
    """ Reflect image UP and LEFT with Tile_Padding ,
     DOWN and RIGHT to be able to crop with the size of
    [Tile_Width + 2* Tile_Padding] minimum as many times as with the size of Tile_Width
    """
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
    return image_reflected, croped_images , number_crops

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


def random_crop_rotate90(image_np_array: np.ndarray, mask_np_array: np.ndarray, crop_size: int, Row_min: int, Row_max: int, Column_min: int, Column_max: int):    
    """Between Row_min/_max and Column_min/_max
     Random Crop of [crop_size x crop_size] and
     Random Rotate of multiple of 90 Degree.
    """
    # image size 1920x1080         
    Row_random = random.randint(Row_min , Row_max)   
    Column_random = random.randint(Column_min , Column_max)
    #print("Random Column ,Row")
    #print(Row_random,Column_random)
    
    y = Row_random
    x = Column_random
    croped_image  = crop(image_np_array,y,crop_size, x,crop_size)
    croped_mask  = crop(mask_np_array,y,crop_size, x,crop_size)
    
    #plot_img_and_mask(croped_image, croped_mask)

    # Random Rotate at 90 degree
    k_Random = random.randint(0 , 3)         
    #print("Random Degree", k_Random * 90)

    img = np.rot90(croped_image, k_Random)
    msk = np.rot90(croped_mask, k_Random)
    
    if(croped_image.shape[0] != crop_size | croped_image.shape[1] != crop_size ):
        raise TypeError("Error : Croped Image Size is not equal to Crop size !" ,croped_image.shape )
    elif(croped_mask.shape[0] != crop_size | croped_mask.shape[1] != crop_size ):
        raise TypeError("Error : Croped Mask Size is not equal to Crop size Error !" ,croped_mask.shape )   
    return img, msk


def crop_without_padding(image_np_array: np.ndarray, Tile_Width :int):
    """ Reflect image at Edges to crop with [Tile_Width x Tile_Width] and Crop    
    """
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




def pil_imgs_random_crop_rotate90_flip(pil_image: Image, pil_mask: Image,crop_size: int, Row_min: int, Row_max: int, Column_min: int ,Column_max: int):
    """Between Row_min/_max and Column_min/_max
     1. Random Crop of [crop_size x crop_size] and
     2. Random Rotate of multiple of 90 Degree.
     3. Random Flip vertical or horizontal.
    """
    # image size 1920x1080         
    Row_random = random.randint(Row_min , Row_max)   
    Column_random = random.randint(Column_min , Column_max)
    #print("Random Column ,Row")
    #print(Row_random,Column_random)

    #im.crop((left, top, right, bottom))
    left    =  Column_random
    top     =  Row_random
    right   =  Column_random + crop_size
    bottom  =  Row_random + crop_size
    croped_image  = pil_image.crop((left, top, right, bottom))
    croped_mask   = pil_mask.crop((left, top, right, bottom))
    
    #plot_img_and_mask(croped_image, croped_mask)

    # Random Rotate at 90 degree
    k_Random = random.randint(0 , 3)        
    angle  = k_Random * 90  
    #print("Random Degree", k_Random * 90)

    img_rotate = croped_image.rotate(angle)
    msk_rotate = croped_mask.rotate(angle)
    
    j_Random = random.randint(0 , 2)   
    if(j_Random == 0):
        img = img_rotate
        msk = msk_rotate
    elif (j_Random == 1):
        img = img_rotate.transpose(Image.FLIP_LEFT_RIGHT)
        msk = msk_rotate.transpose(Image.FLIP_LEFT_RIGHT)
    elif(j_Random == 2):
        img = img_rotate.transpose(Image.FLIP_TOP_BOTTOM)
        msk = msk_rotate.transpose(Image.FLIP_TOP_BOTTOM)
    else :
        raise TypeError("Random int is not 0,1 or2 !",j_Random)
        
    # fig, ax = plt.subplots(1, 5,facecolor = "lightgrey", dpi = 200)
    # [ax_i.set_axis_off() for ax_i in ax.ravel()]   
    # plt.style.use('grayscale')
    # ### 
    # ax[0].imshow(croped_image)
    # ###
    # ax[1].set_title(str(angle))
    # ax[1].imshow(img_rotate) 
    # ###
    # ax[2].imshow(msk_rotate,vmin=0, vmax=1)
    # ###
    # ax[3].set_title(str(j_Random))
    # ax[3].imshow(img)
    # ax[4].imshow(msk, vmin=0, vmax=1)
    # #
    # plt.show()


    if(img.size  != (crop_size,crop_size)):
        raise TypeError("Error : Croped Image Size is not equal to Crop size !" ,croped_image.size )
    elif(msk.size  != (crop_size,crop_size)):
        raise TypeError("Error : Croped Mask Size is not equal to Crop size Error !" ,croped_mask.size )   
    return img, msk




def pil_images_combine(img_list:np.ndarray, img_width,img_height, h:int , v:int , space:int , bachground = "white" ):
    """Combine Image list to one image.img_width
     1. img_list: List of Images.
     2. img_width : Image width of the image in list.
     3. img_height : Image heigth of the image in list.
     4. h: number of images to be combined horizontally.
     5. v: number of images to be combined vertically.
     6. space: pixel value for space between the part images.
     7. bachground : bachground color .
    """
    new_width = h*img_width + (h+1)*space
    new_height= v*img_height + (v+1)*space
    new_image = Image.new(mode="L",size=(new_width,new_height), color=bachground )
    for i in range (0,v):   
        for j in range (0,h):
            y_pos = space + j*img_height + j*space
            x_pos = space + i*img_width + i*space
            img = Image.fromarray(img_list[h*i+j])
            new_image.paste(img,(y_pos,x_pos))
    return new_image