#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#%%
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc   =  nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True)
                                )
  
        self.down1 =  nn.Sequential(
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True)
                                )
    
        self.down2 =  nn.Sequential(
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True)
                                )
   
        self.down3 =  nn.Sequential(
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True)
                                )
  
        self.down  =  nn.Sequential(
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                                nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True)
                                )
        
        self.up1_1 =  Up(1024,bilinear)        
        self.up1   =  nn.Sequential(
                            #nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2)),                    
                            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(inplace=True)
                            )

        self.up2_1 =  Up(512,bilinear)  
        self.up2   =  nn.Sequential(
                            #nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2)),
                            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(inplace=True)
                            )
        
        self.up3_1 =  Up(256,bilinear)  
        self.up3   =  nn.Sequential(
                            #nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2)),
                            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(inplace=True),
                            )

        self.up4_1 =  Up(1024,bilinear)  
        self.up4   =  nn.Sequential(
                            #nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),   
                            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(inplace=True),
                            )



    def forward(self, x):
        print(x.shape)

        x1 = self.inc(x)
        print(x1.shape)

        x2 = self.down1(x1)
        print(x2.shape)

        x3 = self.down2(x2)
        print(x3.shape)

        x4 = self.down3(x3)
        print(x4.shape)

        x5 = self.down4(x4)
        print(x5.shape)

        x = self.up1_1(x5, x4)
        x = self.up1(x)
        print(x.shape)

        x = self.up2_1(x, x3)
        x = self.up2(x)
        print(x.shape)

        x = self.up3_1(x, x2)
        x = self.up3(x, x2)
        print(x.shape)

        x = self.up4_1(x, x1)
        x = self.up4(x, x1)
        print(x.shape)

        logits = self.outc(x)
        print(x.shape)

        time.sleep(2)
        return logits
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels,  bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)      

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x

#%%
# read image
def read(image_path: str | Path) -> np.array:
    """returns an image array """
    from PIL import Image
    # 8 Bit Grafik hat Werte zwischen 0 und 255 (Integer). Wenn man durch 255 teilt, erhält man Float Werte zwischen 0 und 1
    image_array = np.asarray(Image.open(image_path))
    return image_array


#%%
image_path =  r"C:/Users/vmanukyan/Desktop/Notizen_BA/Test_Images/"
#image_name = r"572.png"
#image_name = r"572_gray.png"
#image_name = r"auto.jpg"
#image_name = r"bird.jpg"
#image_name = r"fluidized.png"
image_name = r"hund1_300.jpg"
#image_name = r"hund1_572.jpg"
img = read(image_path+image_name)

print(img.shape)

image = torch.from_numpy(img)
#
#image = torch.zeros((200,200,3))

#%%
new_image = image[None, None ,:]

net= UNet(1,2,False)

new_image2  = new_image.float()/255

# new_image = image[None, :]
# net= UNet(3,2,False)
# new_image2  = (torch.permute(new_image,(0,3,1,2)).float())/255
# print(new_image2.shape)


#print(net)
#%%
out = net(new_image2)
#print(out.shape)

# %%
bild = out[0][0].detach().numpy()
plt.imshow(bild)
# %%
bild = out[0][1].detach().numpy()
plt.imshow(bild)
# %%