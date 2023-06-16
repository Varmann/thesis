#%%
import torch
import torchvision
import matplotlib.pyplot as plt


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/files/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                #torchvision.transforms.RandomRotation(30),
                #torchvision.transforms.RandomAdjustSharpness(1),
                #torchvision.transforms.RandomVerticalFlip()
                #TODO: RandomCrop
                # hier kann die augmentation passieren
                torchvision.transforms.RandomCrop(15)
            ]
        ),
    ),
    batch_size=1,
    shuffle=True,
)
train_loader_iterator = iter(train_loader)
 
#%%
image, cls = next(train_loader_iterator)
plt.imshow(image.squeeze())
# %%
