#%%
import cv2
import glob
import matplotlib.pyplot as plt

image_read_path1 =  r"C:/Users/vmanukyan/Desktop/Fluidized/Croped/HD/Raw/"
images_raw = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob(image_read_path1+'/*.png')]

print(type(images_raw))
print(len(images_raw))

image_read_path2 =  r"C:/Users/vmanukyan/Desktop/Fluidized/Croped/HD/Segmented/"
images_segmented = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob(image_read_path2+'/*.png')]

print(type(images_segmented))
print(len(images_segmented))

#%%
crop_size = 300

# %%
plt.imshow(images_segmented[3])
# %%
import random
x_min = 230
x_max  = 820 - crop_size

x_random = random.randint(x_min , x_max)

y_min = 600
y_max  = 1750 - crop_size
y_random = random.randint(y_min , y_max)

print(x_random , y_random)

# %%
image_save1 = r'C:/Users/vmanukyan/Desktop/Fluidized/Croped/'+ str(crop_size)+r'/Raw/'
image_save2 = r'C:/Users/vmanukyan/Desktop/Fluidized/Croped/'+ str(crop_size)+r'/Segmented/'

for index in range(0,7):
    for i in range(0, 10):
        x_random = random.randint(x_min , x_max)
        y_random = random.randint(y_min , y_max)
        print(x_random , y_random)
        x = x_random
        y = y_random
        croped_raw  = images_raw[index][y:y+crop_size, x:x+crop_size]
        croped_segmented  = images_segmented[index][y:y+crop_size, x:x+crop_size]
        ind  = str(index+1)+'_'+str(i+1)
        filename1 = 'image_' + ind + '.png' 
        cv2.imwrite(image_save1+filename1, croped_raw)
        filename2 = 'image_' + ind + '_mask'+ '.png' 
        cv2.imwrite(image_save2+ filename2, croped_segmented)

# %%
image_read_path1 =  image_save1
images_croped_raw = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob(image_read_path1+'/*.png')]

print(type(images_croped_raw))
print(len(images_croped_raw))

image_read_path2 =  image_save2
images_croped_segmented = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob(image_read_path2+'/*.png')]

print(type(images_croped_segmented))
print(len(images_croped_segmented))


# %%


