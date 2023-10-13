from pathlib import Path

# Default Values of train model arguments
EPOCHS = 50
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
# Percent of the data that is used as validation (0-100)
VAL_PERCENT = 20.0
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
TILE_WIDTH = 200
TILE_PADDING = 50
CROP_SIZE = TILE_WIDTH + 2 * TILE_PADDING

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


# Files Paths
DIR_CHECKPOINTS = Path(__file__).parent.resolve() / "checkpoints"
#DIR_CHECKPOINTS.mkdir(exist_ok=True)

images_not_trained = True

DIR_IMG = Path(__file__).parent.resolve() / "data" / "imgs"
DIR_MASKS = Path(__file__).parent.resolve() / "data" / "masks"
DIR_PREDICTED = Path(__file__).parent.resolve() / "data" / "imgs_predicted"
SAVE_PLOTS_PATH_IMAGE = Path(__file__).parent.resolve() / "data" / "Save_plots_Image"
SAVE_PLOTS_PATH_PREDICT = Path(__file__).parent.resolve() / "data" / "Save_plots_Predict"

if(images_not_trained) :
    DIR_IMG = Path(__file__).parent.resolve() / "data" / "imgs_not_seen" / "imgs"
    DIR_MASKS = Path(__file__).parent.resolve() / "data" / "imgs_not_seen" /"masks"
    DIR_PREDICTED = Path(__file__).parent.resolve() / "data" / "imgs_not_seen" /"imgs_predicted"
    SAVE_PLOTS_PATH_IMAGE = Path(__file__).parent.resolve() / "data" /"imgs_not_seen" /"Save_plots_Image"
    SAVE_PLOTS_PATH_PREDICT = Path(__file__).parent.resolve() / "data" /"imgs_not_seen" /"Save_plots_Predict"


#DIR_PREDICTED.mkdir(exist_ok=True)

IMAGE_FILES_PATH = [str(p) for p in DIR_IMG.iterdir() if p.is_file()]


n = len(IMAGE_FILES_PATH)
IMAGE_FILES_PATH = []
IMGS_MASK_FILES_PATH = []
PREDICTED_IMAGES_FILES_PATH = []
SAVE_PLOTS_IMAGE_FILES_PATH = []
SAVE_PLOTS_PREDICT_FILES_PATH = []

for i in range(0,n):
   IMAGE_FILES_PATH.append( str(DIR_IMG) + "\\image_" + str(i+1) + ".png")
   IMGS_MASK_FILES_PATH.append( str(DIR_MASKS) + "\\image_" + str(i+1) + "_mask.png")
   PREDICTED_IMAGES_FILES_PATH.append( str(DIR_PREDICTED) + "\\image_" + str(i+1) + "_OUT.png")
   SAVE_PLOTS_IMAGE_FILES_PATH.append( str(SAVE_PLOTS_PATH_IMAGE) + "\\plot_Image_" + str(i+1) + "_OUT.png")
   SAVE_PLOTS_PREDICT_FILES_PATH.append( str(SAVE_PLOTS_PATH_PREDICT) + "\\plot_Predict_" + str(i+1) + "_OUT.png")



# SAVE_PLOTS_FILES_PATH = [
#     str(SAVE_PLOTS_PATH / (p.stem + "_OUT.png"))
#     for p in DIR_IMG.iterdir()
#     if p.is_file()
# ]
