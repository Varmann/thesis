import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from defaults import *



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
    return image_reflected, croped_images , number_crops




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
    new_image = Image.new(mode="RGB",size=(new_width,new_height), color=bachground )
    for i in range (0,v):   
        for j in range (0,h):
            y_pos = space + j*img_height + j*space
            x_pos = space + i*img_width + i*space
            img = Image.fromarray(img_list[h*i+j])
            new_image.paste(img,(y_pos,x_pos))
    return new_image


def plot_img_and_mask(img, mask):
    fig, ax = plt.subplots(1, 2 ,facecolor = "lightgrey")
    [ax_i.set_axis_off() for ax_i in ax.ravel()]
    plt.style.use('grayscale')        
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Mask') 
    ax[1].imshow(mask,vmin=0, vmax=1)
    plt.show()

#def  plot_images(index_Input_Image:int,input_image, image_roi,image_reflected,croped_images_padding,predicted_masks, predicted_mask, result_mask, img_roi_mask, img_roi_mask_edge,mask_edges, predicted_mask_no_padding ):


def plot_images(
    index_Input_Image: int,
    input_image:np.ndarray,
    image_roi:np.ndarray,
    image_reflected_padding:np.ndarray,    
    croped_images_padding:np.ndarray,
    predicted_masks:np.ndarray,
    predicted_mask:np.ndarray,
    result_mask:Image,
    img_roi_mask:Image,
    img_roi_mask_edge:Image,
    mask_edges:np.ndarray,
    image_reflected_no_padd:np.ndarray,    
    predicted_mask_no_padding:np.ndarray,
    number_crops_padding:[int,int],
    number_crops_no_padd:[int,int],
    reflect_color:str,
    reflect_linestyle:str,
    facecolor_plot:str,
    backcolor:str,
    padding_color:str,
    padd_linestyle_1:str,
    padd_linestyle_2:str,
    combine_space:int,
    TILE_WIDTH_color:str,
    TILE_WIDTH_linestyle:str, 
    titelcolor:str,
    titelfontsize:int,
    titelfontname:str,
    bounding_box_color:str,
    bounding_box = False,
    show = True,
    save = True,
):
        # backcolor = "cyan"
        # padding_color = "yellow"
        # TILE_WIDTH_color = "blue"
        # titelcolor = "red"
        # titelfontsize = 7
        # titelfontname = "cursive"
        # bounding_box_color = "blue"
        fig, ax = plt.subplots(2, 6,figsize = (8,4), facecolor = facecolor_plot, dpi = 600)
        if(bounding_box):           
            [ax_i.set_xticks([])  for ax_i in ax.ravel()]  
            [ax_i.set_yticks([])  for ax_i in ax.ravel()]  
            #[ax_i.spines["left"].set_visible(True)  for ax_i in ax.ravel()]  
            [ax_i.spines["left"].set_color(bounding_box_color) for ax_i in ax.ravel()]  
            [ax_i.spines["top"].set_color(bounding_box_color) for ax_i in ax.ravel()]  
            [ax_i.spines["right"].set_color(bounding_box_color) for ax_i in ax.ravel()]  
            [ax_i.spines["bottom"].set_color(bounding_box_color) for ax_i in ax.ravel()]  
        else:
            [ax_i.set_axis_off() for ax_i in ax.ravel()]  
        ### 1
        ax[0][0].set_title('Image',color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)
        ax[0][0].imshow(Image.fromarray(image_roi).convert("RGB"))
        ###
        ax[0][1].set_title('Reflect', color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)             
        x = [TILE_PADDING,TILE_PADDING,(TILE_PADDING+image_roi.shape[1]),(TILE_PADDING+image_roi.shape[1]),TILE_PADDING]
        y = [TILE_PADDING,(TILE_PADDING+image_roi.shape[0]),(TILE_PADDING+image_roi.shape[0]),TILE_PADDING,TILE_PADDING]
        ax[0][1].plot(x,y, color= reflect_color,linestyle = reflect_linestyle, linewidth = 0.3)
        ax[0][1].imshow(Image.fromarray(image_reflected_padding).convert("RGB"))      

        ax[0][2].set_title('1. Padding', color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)  
        ax[0][3].set_title('2. Padding', color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)               
        # Draw/Plot Padding Rectagles on Reflected Image.
        h_padding = number_crops_padding[0]  # horizontal
        v_padding = number_crops_padding[1]  # vertical
        for k in range (0,3):   
            for i in range (0,v_padding):   
                for j in range (0,h_padding):
                    # Crop image
                    line_width = 2                            
                    space_x = line_width
                    space_y = line_width
                    if (i==0):
                        space_x = line_width if(j==0) else 0
                    if (j== (h_padding-1)):
                        space_x = -line_width
                    if (i== (v_padding-1)):
                        space_y= -line_width 
                    
                    if k==0:   
                        # Rectangle 200x200
                        x1 = TILE_PADDING+j*TILE_WIDTH 
                        y1 = TILE_PADDING+i*TILE_WIDTH  
                        if(i==0): 
                            x = [x1,x1,x1+TILE_WIDTH,x1+TILE_WIDTH,x1]
                            y = [y1,y1+TILE_WIDTH,y1+TILE_WIDTH,y1,y1]
                            ax[0][2].plot(x, y, color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)
                            ax[0][3].plot(x, y, color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)
                        else:
                            if(j==0): 
                                ax[0][2].plot([x1,x1],[y1,y1+TILE_WIDTH], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                                ax[0][2].plot([x1,x1+TILE_WIDTH],[y1+TILE_WIDTH,y1+TILE_WIDTH], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                                ax[0][2].plot([x1+TILE_WIDTH,x1+TILE_WIDTH],[y1+TILE_WIDTH,y1], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                            
                                ax[0][3].plot([x1,x1],[y1,y1+TILE_WIDTH], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                                ax[0][3].plot([x1,x1+TILE_WIDTH],[y1+TILE_WIDTH,y1+TILE_WIDTH], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                                ax[0][3].plot([x1+TILE_WIDTH,x1+TILE_WIDTH],[y1+TILE_WIDTH,y1], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                            else:
                                ax[0][2].plot([x1,x1+TILE_WIDTH],[y1+TILE_WIDTH,y1+TILE_WIDTH], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                                ax[0][2].plot([x1+TILE_WIDTH,x1+TILE_WIDTH],[y1+TILE_WIDTH,y1], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)
                                
                                ax[0][3].plot([x1,x1+TILE_WIDTH],[y1+TILE_WIDTH,y1+TILE_WIDTH], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                                ax[0][3].plot([x1+TILE_WIDTH,x1+TILE_WIDTH],[y1+TILE_WIDTH,y1], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)
          
                        # Rectangle 200x200
                        x1 = TILE_PADDING+j*CROP_SIZE + (j+1)*combine_space
                        y1 = TILE_PADDING+i*CROP_SIZE + (i+1)*combine_space
                        x = [x1,x1,x1+TILE_WIDTH,x1+TILE_WIDTH,x1]
                        y = [y1,y1+TILE_WIDTH,y1+TILE_WIDTH,y1,y1]

                        # Predicted Masks Combined + TILE_WIDTH Rectangles
                        ax[0][5].plot(x,y, color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.3)                    
                        
                        
                        # Rectangle 200x200
                        x1 = TILE_PADDING+j*TILE_WIDTH
                        y1 = TILE_PADDING+i*TILE_WIDTH
                        x = [x1,x1,x1+TILE_WIDTH,x1+TILE_WIDTH,x1]
                        y = [y1,y1+TILE_WIDTH,y1+TILE_WIDTH,y1,y1]

                        ## Predicted Mask + Padding
                        ax[1][0].plot(list(np.asarray(x) -TILE_PADDING), list(np.asarray(y) -TILE_PADDING), color= TILE_WIDTH_color,linestyle = "solid", linewidth = 0.3)
                    else:  
                        # Rectangle 300x300                              
                        x1 = j*TILE_WIDTH + space_x
                        y1 = i*TILE_WIDTH  + space_y  
                        x = [x1,x1,x1+CROP_SIZE,x1+CROP_SIZE,x1]
                        y = [y1,y1+CROP_SIZE,y1+CROP_SIZE,y1,y1]

                    if k==1:                                
                        if ((j+1)%2==0) and ((i+1)%2==0):
                            ax[0][2].plot(x, y, color= padding_color,linestyle = padd_linestyle_2, linewidth = 0.5)
                        if ((j+1)%2==1) and ((i+1)%2==0):
                            ax[0][3].plot(x, y, color= padding_color,linestyle = padd_linestyle_2, linewidth = 0.5)    
                    elif k==2:
                        if ((j+1)%2==1) and ((i+1)%2==1):
                            ax[0][2].plot(x, y, color= padding_color ,linestyle = padd_linestyle_1, linewidth = 0.5)
                        if ((j+1)%2==0) and ((i+1)%2==1):
                            ax[0][3].plot(x, y, color= padding_color ,linestyle = padd_linestyle_1, linewidth = 0.5)

        ax[0][2].imshow(Image.fromarray(image_reflected_padding).convert("RGB"))
        ax[0][3].imshow(Image.fromarray(image_reflected_padding).convert("RGB"))
        # ###
        ax[0][4].set_title('Crop with Padding', color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)
        croped_images_padding_combined = pil_images_combine(croped_images_padding ,(TILE_WIDTH + 2*TILE_PADDING),(TILE_WIDTH + 2*TILE_PADDING), h_padding, v_padding , space = combine_space, bachground = padding_color )
        ax[0][4].imshow(croped_images_padding_combined.convert("RGB"))
        ###
        ax[0][5].set_title('Predict with Padding', color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)                
        predicted_masks_combined = pil_images_combine(predicted_masks ,(TILE_WIDTH + 2*TILE_PADDING),(TILE_WIDTH + 2*TILE_PADDING), h_padding, v_padding , space = combine_space , bachground = backcolor )        
        ax[0][5].imshow(predicted_masks_combined.convert("RGB"))
        ### 2
        ax[1][0].set_title('Predicted Mask',color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)
        ax[1][0].imshow(Image.fromarray(predicted_mask).convert("RGB"))
        ###
        ax[1][1].set_title('Mask', color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)
        ax[1][1].imshow(result_mask.convert("RGB"))
        ###
        ax[1][2].set_title('Image + Mask',color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)
        ax[1][2].imshow(img_roi_mask.convert("RGB"))
        ###
        ax[1][3].set_title('Image + Mask Edges',color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)
        ax[1][3].imshow(Image.fromarray(img_roi_mask_edge).convert("RGB"))
        ###
        ax[1][4].set_title('Crop without Padding',color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)     
        ax[1][4].plot(  [0,image_roi.shape[1],image_roi.shape[1]],[image_roi.shape[0],image_roi.shape[0],0]  , color= reflect_color,linestyle = reflect_linestyle, linewidth = 0.3) 
        # Draw/Plot Padding Rectagles on Reflected Image.
        h_no_padd = number_crops_no_padd[0]  # horizontal
        v_no_padd  = number_crops_no_padd[1]  # vertical   
        for i in range (0,v_no_padd ):   
            for j in range (0,h_no_padd ):
                # Rectangle 300x300  
                x1 = j*CROP_SIZE
                y1 = i*CROP_SIZE
                if(i==0): 
                    x = [x1,x1,x1+CROP_SIZE,x1+CROP_SIZE,x1]
                    y = [y1,y1+CROP_SIZE,y1+CROP_SIZE,y1,y1]
                    ax[1][4].plot(x, y, color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)
                else:
                    if(j==0): 
                        ax[1][4].plot([x1,x1],[y1,y1+CROP_SIZE], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                        ax[1][4].plot([x1,x1+CROP_SIZE],[y1+CROP_SIZE,y1+CROP_SIZE], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                        ax[1][4].plot([x1+CROP_SIZE,x1+CROP_SIZE],[y1+CROP_SIZE,y1], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                    else:
                        ax[1][4].plot([x1,x1+CROP_SIZE],[y1+CROP_SIZE,y1+CROP_SIZE], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)   
                        ax[1][4].plot([x1+CROP_SIZE,x1+CROP_SIZE],[y1+CROP_SIZE,y1], color= TILE_WIDTH_color,linestyle = TILE_WIDTH_linestyle, linewidth = 0.5)
          
        ax[1][4].imshow(Image.fromarray(image_reflected_no_padd).convert("RGB"))
        # ax[1][4].set_title('Mask Edges',color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)        
        # ax[1][4].imshow(mask_edges.convert("RGB"))
        ###
        ax[1][5].set_title('Predict without padding',color = titelcolor, fontsize=titelfontsize, fontname = titelfontname)
        ax[1][5].imshow(Image.fromarray(predicted_mask_no_padding).convert("RGB"))
        ###
        if(save):
            plt.savefig(SAVE_PLOTS_FILES_PATH[index_Input_Image],facecolor=facecolor_plot,bbox_inches='tight')  
            plt.close()
        if(show):  
            if(save):
                saved_plot_image = Image.open(SAVE_PLOTS_FILES_PATH[index_Input_Image]).convert("RGB")  
                saved_plot_image.show() 
                #saved_plot_image.close() 
            else:
                img_buf = io.BytesIO() 
                plt.savefig(img_buf, format='png',facecolor=facecolor_plot,bbox_inches='tight')                  
                pil_image_from_buffer = Image.open(img_buf).convert("RGB")
                pil_image_from_buffer.show()
                #pil_image_from_buffer.close() 
                pil_image_from_buffer.save(SAVE_PLOTS_FILES_PATH[k])
                img_buf.close()
                     


          
