import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.utils import *

from defaults import *

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    try:
        mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    except Exception as e:
        raise e
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            z = tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            )
            unique = list(z)

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # TODO bei resize Performance untersuchen
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_files = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_files = list(self.images_dir.glob(name + '.*'))

        assert len(img_files) == 1, f'Either no image or multiple images found for the ID {name}: {img_files}'
        assert len(mask_files) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_files}'
        mask = load_image(mask_files[0])
        image = load_image(img_files[0])

        assert image.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {image.size} and {mask.size}'
        

         # TODO Random Crop 300x300 oder andere Augmentation(Spiegeln, 90Grad drehen).
        # Jedes Bild nur einmal, stattdessen sehr viele epochs.       
        # Row max for Random Row         
        Row_max  = ROW_MIN + HEIGHT - CROP_SIZE
        #Column max for Random Column 
        Column_max  = COL_MIN + WIDTH - CROP_SIZE 

        croped_image, croped_mask = pil_imgs_random_crop_rotate90_flip(image, mask, CROP_SIZE, ROW_MIN, Row_max, COL_MIN ,Column_max) 

        image = self.preprocess(self.mask_values, croped_image, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, croped_mask, self.scale, is_mask=True)   

        # fig, ax = plt.subplots(1, 4,facecolor = "lightgrey", dpi = 200)
        # [ax_i.set_axis_off() for ax_i in ax.ravel()]   
        # plt.style.use('grayscale')
        # ### 
        # ax[0].set_title('Croped Image')
        # ax[0].imshow(np.array(croped_image))
        # ###
        # ax[1].set_title('Croped Mask')
        # ax[1].imshow(np.array(croped_mask)) 
        # ###
        # ax[2].set_title('Preprocess')
        # ax[2].imshow((np.array(image))[0,:,:])
        # ###
        # ax[3].set_title('Preprocess')
        # ax[3].imshow(np.array(mask), vmin=0, vmax=1)
        # ###    
        # plt.show()
        # print("done plot")

        return {
                'image': torch.as_tensor(image.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous()
            }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')



