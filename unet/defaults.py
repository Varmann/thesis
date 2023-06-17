from pathlib import Path

#### Default values for Unet Model 
# Minimum probability value to consider a mask pixel white
MASK_THRESHOLD = 0.5 
# checkpoint default path
MODEL_CHECKPOINT = Path(__file__).parent.resolve() / "checkpoints" / "checkpoint.pth"
# Default classes number
N_CLASSES = 2
# Image Scaling dont change for the given Konfiguration
IMAGE_SCALE = 1.0
# Default Values of train model arguments
EPOCHS = 20
BATCH_SIZE = 3 
LEARNING_RATE = 1e-5
# Percent of the data that is used as validation (0-100)
VAL_PERCENT = 20.0


### Padding and Crop Parameter
TILE_WIDTH = 200
TILE_PADDING = 50
CROP_SIZE = TILE_WIDTH + 2 * TILE_PADDING

# Input Image 
# Row
ROW_MIN = 350
HEIGHT = 1400
# Column
COL_MIN = 230
WIDTH = 600


# Files Paths
DIR_CHECKPOINTS = Path(__file__).parent.resolve() / "checkpoints"
DIR_CHECKPOINTS.mkdir(exist_ok=True)

DIR_IMG = Path(__file__).parent.resolve() / "data" / "imgs"
DIR_MASKS = Path(__file__).parent.resolve() / "data" / "masks"
DIR_PREDICTED = Path(__file__).parent.resolve() / "data" / "imgs_predicted"
DIR_PREDICTED.mkdir(exist_ok=True)

IMGS_MASKS_FILES_PATH  = [
    str(DIR_MASKS / (p.stem + "_OUT.png"))
    for p in DIR_MASKS.iterdir()
    if p.is_file()
]

IMAGE_FILE_PATHS = [str(p) for p in DIR_IMG.iterdir() if p.is_file()]
PREDICTED_IMAGES_FILES_PATH = [
    str(DIR_PREDICTED / (p.stem + "_OUT.png"))
    for p in DIR_IMG.iterdir()
    if p.is_file()
]



SAVE_PLOTS_PATH = Path(__file__).parent.resolve() / "data" / "Save_plots"
SAVE_PLOTS_PATH.mkdir(exist_ok=True)
SAVE_PLOTS_FILES_PATH = [
    str(SAVE_PLOTS_PATH / (p.stem + "_OUT.png"))
    for p in DIR_IMG.iterdir()
    if p.is_file()
]



