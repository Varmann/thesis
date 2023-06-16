from pathlib import Path

#### Default values for Unet Model
# Minimum probability value to consider a mask pixel white
MASK_THRESHOLD = 0.5 
# checkpoint default path
MODEL_CHECKPOINT = Path(__file__).parent.resolve() / "checkpoints" / "checkpoint.pth"
# Default classes number
N_CLASSES_DV = 2
# Image Scaling dont change for the given Konfiguration
IMAGE_SCALING  = 1.0

### Padding and Crop Parameter
TILE_WIDTH = 200
TILE_PADDING = 50
CROP_SIZE = TILE_WIDTH + 2 * TILE_PADDING

# Row
ROW_MIN = 350
HEIGHT = 1400

# Column
COL_MIN = 230
WIDTH = 600


# Files Paths
fpath = Path(__file__).parent.resolve()

data_path = fpath / "data"
imgs_path = data_path / "imgs"
predicted_imgs_path = data_path / "imgs_predicted"
predicted_imgs_path.mkdir(exist_ok=True)

image_file_paths = [str(p) for p in imgs_path.iterdir() if p.is_file()]
predicted_images_file_paths = [
    str(predicted_imgs_path / (p.stem + "_OUT.png"))
    for p in imgs_path.iterdir()
    if p.is_file()
]

imgs_masks_path = data_path / "imgs_masks"
imgs_masks_file_path = [
    str(imgs_masks_path / (p.stem + "_OUT.png"))
    for p in imgs_path.iterdir()
    if p.is_file()
]

imgs_masks_path2 = data_path / "imgs_masks//Without_Padding"
imgs_masks_file_path2 = [
    str(imgs_masks_path2 / (p.stem + "_OUT.png"))
    for p in imgs_path.iterdir()
    if p.is_file()
]
