from pathlib import Path
import io
import os

# Default Values of train model arguments
EPOCHS = 100
BATCH_SIZE = 50
LEARNING_RATE = 1e-5
# Percent of the data that is used as validation (0-100)
VAL_PERCENT = 10.0
# Default classes number
N_CLASSES = 2
# Image Scaling dont change for the given Konfiguration
IMAGE_SCALE = 1.0

#### Default values for Unet Model
# Minimum probability value to consider a mask pixel white
MASK_THRESHOLD = 0.5
# checkpoint default path
MODEL_CHECKPOINT = Path(__file__).parent.resolve() / "checkpoints" / "checkpoint.pth"
# Load Model from last Training
TRAIN_LOAD_LAST_MODEL = False

### Padding and Crop Parameter
## CROP_SIZE corresponds to CNN input layer Parameter . Don't change this parameter.
CROP_SIZE = 300
# TILE_PADDING can be changed. Default is 50.
TILE_PADDING = 30
TILE_WIDTH = CROP_SIZE - 2*TILE_PADDING

# Input Image: Height = 1500, Widht = 750
###############################################
# Input Image and Input Mask Region of Interest 
# Region of the air bubbles.
# Row
ROW_MIN = 50
HEIGHT = 1250+150
#Column
COL_MIN = 60
WIDTH = 620


#### Entire Image 
# ROW_MIN = 0
# HEIGHT = 1500
# #Column
# COL_MIN = 0
# WIDTH = 750



# ROW_MIN = 350
# HEIGHT = 1400
# Column
# COL_MIN = 230
# WIDTH = 600


SAVE_IMAGES_MASKS = True

# Files Paths
DIR_CHECKPOINTS = Path(__file__).parent.resolve() / "checkpoints"
#DIR_CHECKPOINTS.mkdir(exist_ok=True)

images_not_trained = True

DIR_IMG = Path(__file__).parent.resolve() / "data" / "imgs"
DIR_MASKS = Path(__file__).parent.resolve() / "data" / "masks"
DIR_PREDICTED = Path(__file__).parent.resolve() / "data" / "imgs_predicted"
SAVE_PLOTS_PATH = Path(__file__).parent.resolve() / "data" / "Save_plots"
SAVE_ROI_MASK_PATH = Path(__file__).parent.resolve() / "data" / "roi_mask_predicted"
SAVE_ROI_MASK_EDGE_PATH = Path(__file__).parent.resolve() / "data" / "roi_mask_edge_predicted"

if(images_not_trained) :
    DIR_IMG = Path(__file__).parent.resolve() / "data" / "imgs_not_seen" / "imgs"
    DIR_MASKS = Path(__file__).parent.resolve() / "data" / "imgs_not_seen" /"masks"
    DIR_PREDICTED = Path(__file__).parent.resolve() / "data" / "imgs_not_seen" /"imgs_predicted"
    SAVE_PLOTS_PATH = Path(__file__).parent.resolve() / "data" /"imgs_not_seen" /"Save_plots"
    SAVE_ROI_MASK_PATH = Path(__file__).parent.resolve() / "data" / "imgs_not_seen"/"roi_mask_predicted"
    SAVE_ROI_MASK_EDGE_PATH = Path(__file__).parent.resolve() / "data" /"imgs_not_seen" /"roi_mask_edge_predicted"


#DIR_PREDICTED.mkdir(exist_ok=True)

IMAGE_FILES_PATH = [str(p) for p in DIR_IMG.iterdir() if p.is_file()]


n = len(IMAGE_FILES_PATH)
IMAGE_FILES_PATH = []
IMGS_MASK_FILES_PATH = []
PREDICTED_IMAGES_FILES_PATH = []
SAVE_PLOTS_FILES_PATH = []
SAVE_ROI_MASK_FILES_PATH = []
SAVE_ROI_MASK_EDGE_FILES_PATH = []

for i in range(0,n):
   IMAGE_FILES_PATH.append( str(DIR_IMG) + "\\image_" + str(i+1) + ".png")
   IMGS_MASK_FILES_PATH.append( str(DIR_MASKS) + "\\image_" + str(i+1) + "_mask.png")
   PREDICTED_IMAGES_FILES_PATH.append( str(DIR_PREDICTED) + "\\image_" + str(i+1) + "_OUT.png")
   SAVE_PLOTS_FILES_PATH.append( str(SAVE_PLOTS_PATH) + "\\plot_" + str(i+1) + "_OUT.png")
   SAVE_ROI_MASK_FILES_PATH.append( str(SAVE_ROI_MASK_PATH) + "\\ROI_MASK_" + str(i+1) + "_Predicted.png")
   SAVE_ROI_MASK_EDGE_FILES_PATH.append( str(SAVE_ROI_MASK_EDGE_PATH) + "\\ROI_MASK_EDGE" + str(i+1) + "_Predicted.png")


# SAVE_PLOTS_FILES_PATH = [
#     str(SAVE_PLOTS_PATH / (p.stem + "_OUT.png"))
#     for p in DIR_IMG.iterdir()
#     if p.is_file()
# ]
