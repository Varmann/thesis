# %%
from pathlib import Path
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image , ImageFilter
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import *
from defaults import *

import matplotlib.pyplot as plt

# %%
def predict_img(net, full_img, device, scale_factor=1.0, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(
            output, (full_img.size[1], full_img.size[0]), mode="bilinear"
        )
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model",
        "-m",
        default=MODEL_CHECKPOINT,
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="INPUT",
        default=IMAGE_FILES_PATH,
        nargs="+",
        help="Filenames of input images",
        required=False,
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="OUTPUT",
        default=PREDICTED_IMAGES_FILES_PATH,
        nargs="+",
        help="Filenames of output images",
    )
    parser.add_argument(
        "--viz",
        "-v",
        action="store_true",
        help="Visualize the images as they are processed",
    )
    parser.add_argument(
        "--no-save", "-n", action="store_true", help="Do not save the output masks"
    )
    parser.add_argument(
        "--mask-threshold",
        "-t",
        type=float,
        default=MASK_THRESHOLD,
        help="Minimum probability value to consider a mask pixel white",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        choices=[1.0],
        help="Scale factor for the input images",
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--classes", "-c", type=int, default=N_CLASSES, help="Number of classes"
    )
    (args, unknown) = parser.parse_known_args()
    return (args, unknown)


def get_output_filenames(args):
    def _generate_name(fn):
        return f"{os.path.splitext(fn)[0]}_OUT.png"

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros(
            (mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8
        )
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 1:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == "__main__":
    args, unknown = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")       

    model_args = " ".join(
        f"{k}={v}" for k, v in vars(args).items() if k not in ["input", "output", "model"]
    )
    logging.info(f"Model args -> {model_args}")


    in_files = args.input
    # out_files = get_output_filenames(args)
    out_files = PREDICTED_IMAGES_FILES_PATH
    # print(out_files)

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model -> {(args.model.name)}")
    logging.info(f"Using device -> {device}")

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    for index_Input_Image, filename_Input_Image in enumerate(in_files):
        logging.info(f'Predicting image -> {Path(filename_Input_Image).name} ...')
        input_image = np.array(Image.open(filename_Input_Image).convert("L"))

        #####################################################################################
        logging.info("Croping with padding")
        # crop image
        image_roi = input_image[ROW_MIN : ROW_MIN + HEIGHT, COL_MIN : COL_MIN + WIDTH]
        #plt.imshow(image)
        # Reflect / Crop
        image_reflected, croped_images, number_crops = crop_with_padding(image_roi, TILE_WIDTH, TILE_PADDING)
        predicted_masks = {}
        h = number_crops[0]  # horizontal
        v = number_crops[1]  # vertical

        croped_imgs_combined = pil_images_combine(croped_images ,(TILE_WIDTH + 2*TILE_PADDING),(TILE_WIDTH + 2*TILE_PADDING), h, v , space = 10, bachground = "white" )
        #imgs_combined.show()
        # Predict every part      
        for i in range(0, v):
            for j in range(0, h):
                mask1 = predict_img(
                    net=net,
                    full_img=Image.fromarray(croped_images[h * i + j]),
                    scale_factor=args.scale,
                    out_threshold=args.mask_threshold,
                    device=device,
                )
                temp_mask1 = mask_to_image(mask1, mask_values)
                predicted_masks[h * i + j] = np.array(temp_mask1)
        
        
        predicted_masks_combined = pil_images_combine(predicted_masks ,(TILE_WIDTH + 2*TILE_PADDING),(TILE_WIDTH + 2*TILE_PADDING), h, v , space = 10 , bachground = "lightgray" )
        #masks_combined.show()

        # Add together
        mask_add_h = []
        mask_add_v = []
        for i in range(0, v):
            # Add horizontal
            mask_add_h = crop(
                predicted_masks[h * i], TILE_PADDING, TILE_WIDTH, TILE_PADDING, TILE_WIDTH
            )
            for j in range(1, h):
                mask_crop = crop(
                    predicted_masks[h * i + j], TILE_PADDING, TILE_WIDTH, TILE_PADDING, TILE_WIDTH
                )
                mask_add_h = np.concatenate((mask_add_h, mask_crop), axis=1)
            # Add vertical
            if i == 0:
                img_add_v = mask_add_h
            else:
                img_add_v = np.concatenate((img_add_v, mask_add_h), axis=0)

        predicted_mask = img_add_v
        ###
        result_mask = Image.fromarray(crop(predicted_mask, 0, HEIGHT, 0, WIDTH)).convert('L')
        ## Edges of Segment in predicted mask.
        mask_edges = result_mask.filter(ImageFilter.FIND_EDGES)
        mask_edges = mask_edges.filter(ImageFilter.MaxFilter)
        
        output_rgb_1 = image_roi[..., None].repeat(3, axis=-1)
        mask_array = np.array(result_mask.convert("L"))

        for i in range(3):
            #Red
            if i==0:
                val = 255
            #Green
            if i==1:
                val = 0
            #Blue
            if i==1:
                val = 0
            #val = 255 if i == 0 else 0
            output_rgb_1[..., i] = np.where(mask_array > 0, val, 0)
        
        
        background = Image.fromarray(image_roi)
        foreground = Image.fromarray(output_rgb_1)
        foreground.paste(background, (0, 0), background)      

        img_roi_mask = foreground.copy()
        img_roi_mask.convert("RGB").save(SAVE_ROI_MASK_FILES_PATH[index_Input_Image])
        logging.info(f"{Path(SAVE_ROI_MASK_FILES_PATH[index_Input_Image]).name}  Image and Predicted Mask saved to -> {Path(SAVE_ROI_MASK_FILES_PATH[index_Input_Image])}  ")

       
        img_roi_mask_edge = image_roi[..., None].repeat(3, axis=-1)
        mask_edges_2 = result_mask.filter(ImageFilter.FIND_EDGES)
        mask_edges_2 = mask_edges_2.filter(ImageFilter.MaxFilter)
        mask_array = np.array(mask_edges_2)

        for i in range(3):
            val = 255 if i == 0 else 0
            img_roi_mask_edge[..., i] = np.where(mask_array > 0, val, img_roi_mask_edge[..., i])
        

        Image.fromarray(img_roi_mask_edge).convert("RGB").save(SAVE_ROI_MASK_EDGE_FILES_PATH[index_Input_Image])
        logging.info(f"{Path(SAVE_ROI_MASK_EDGE_FILES_PATH[index_Input_Image]).name}  Image and Predicted Mask Edge saved to -> {Path(SAVE_ROI_MASK_EDGE_FILES_PATH[index_Input_Image])}  ")

        if not args.no_save:
            out_filename = out_files[index_Input_Image]
            result_mask.save(out_filename)
            logging.info(f" {Path(out_filename).name}  Mask saved to -> {Path(out_filename)}  ")

        if args.viz:
            logging.info(
                f"Visualizing results for image -> {Path(filename_Input_Image).name}, close to continue..."
            )
            plot_img_and_mask(image_roi, predicted_mask)

        #####################################################################################
        logging.info("Croping without padding")
        # Reflect / Crop  without padding
        croped_images2, number_crops2 = crop_without_padding(image_roi, (TILE_WIDTH + 2 * TILE_PADDING))
        pred_masks_2 = {}
        h2 = number_crops2[0]  # horizontal
        v2 = number_crops2[1]  # vertical
        # Predict every part
        for i in range(0, v2):
            for j in range(0, h2):
                mask2 = predict_img(
                    net=net,
                    full_img=Image.fromarray(croped_images2[h2 * i + j]),
                    scale_factor=args.scale,
                    out_threshold=args.mask_threshold,
                    device=device,
                )
                temp_mask2 = mask_to_image(mask2, mask_values)
                pred_masks_2[h2 * i + j] = np.array(temp_mask2)

        # Add together
        img_add_h_2 = []
        img_add_v_2 = []
        for i in range(0, v2):
            # Add horizontal
            img_add_h_2 = pred_masks_2[h2 * i]
            for j in range(1, h2):
                imgs_crop_2 = pred_masks_2[h2 * i + j]
                img_add_h_2 = np.concatenate((img_add_h_2, imgs_crop_2), axis=1)
            # Add vertical
            if i == 0:
                img_add_v_2 = img_add_h_2
            else:
                img_add_v_2 = np.concatenate((img_add_v_2, img_add_h_2), axis=0)

        predicted_mask_no_padding = crop(img_add_v_2, 0, HEIGHT, 0, WIDTH)
        ###
        result_mask2 = Image.fromarray(predicted_mask_no_padding)
        
        if True:
            logging.info(
                f"{Path(SAVE_PLOTS_FILES_PATH[index_Input_Image]).name} Image and Masks plots saved to -> {Path(SAVE_PLOTS_FILES_PATH[index_Input_Image])} "
            )
            logging.info(f'Visualizing Masks for image -> {Path(filename_Input_Image).name}')
            
            
            if  (True):            
   
                backcolor = "blue"
                titelfontsize = 6
                titelfontname = 'Arial'
                fig, ax = plt.subplots(2, 6,figsize = (8,4), facecolor = backcolor, dpi = 600)
                [ax_i.set_axis_off() for ax_i in ax.ravel()]                   
                #fig.tight_layout()                
                #plt.style.use('grayscale')
                ### 1
                ax[0][0].set_title('Input',fontsize=titelfontsize, fontname = titelfontname)
                ax[0][0].imshow(Image.fromarray(input_image).convert("RGB"))
                ###
                ax[0][1].set_title('ROI', fontsize=titelfontsize, fontname = titelfontname)
                ax[0][1].imshow(Image.fromarray(image_roi).convert("RGB")) 
                ##
                ax[0][2].set_title('Reflect', fontsize=titelfontsize, fontname = titelfontname)    
                line_width = 0.3
                # ax[0][2].plot([50,50],[0,image_reflected.shape[0]], color= "red",linestyle = "solid", linewidth = line_width)
                # ax[0][2].plot([0,image_reflected.shape[1]],[50,50], color= "red",linestyle = "solid", linewidth = line_width)
                # ax[0][2].plot([50+image_roi.shape[1],50+image_roi.shape[1]],[0,image_reflected.shape[0]], color= "red",linestyle = "solid", linewidth = line_width)
                # ax[0][2].plot([0,image_reflected.shape[1]],[50+image_roi.shape[0],50+image_roi.shape[0]], color= "red",linestyle = "solid", linewidth = line_width)

                x = [50,50,(50+image_roi.shape[1]),(50+image_roi.shape[1]),50]
                y = [50,(50+image_roi.shape[0]),(50+image_roi.shape[0]),50,50]
                ax[0][2].plot(x,y, color= "red",linestyle = "solid", linewidth = line_width)
                ax[0][2].imshow(Image.fromarray(image_reflected).convert("RGB"))             
                ##
                ax[0][3].set_title('Reflect+Padding', fontsize=titelfontsize, fontname = titelfontname)               

                for k in range (0,3):   
                    for i in range (0,v):   
                        for j in range (0,h):
                            # Crop image
                            line_width = 2                            
                            space_x = line_width
                            space_y = line_width
                            if (i==0):
                                space_x = line_width if(j==0) else 0
                            if (j== (h-1)):
                                space_x = -line_width
                            if (i== (v-1)):
                                space_y= -line_width
                            

                            if k==0:    
                                x1 = 50+j*TILE_WIDTH
                                y1 = 50+i*TILE_WIDTH                         
                                width = TILE_WIDTH
                                x = [x1,x1,x1+width,x1+width,x1]
                                y = [y1,y1+width,y1+width,y1,y1]
                                ax[0][3].plot(x, y, color= "blue",linestyle = "solid", linewidth = 1)
                            else:                                
                                x1 = j*TILE_WIDTH + space_x
                                y1 = i*TILE_WIDTH  + space_y                       

                                width = TILE_WIDTH+2*TILE_PADDING
                                x = [x1,x1,x1+width,x1+width,x1]
                                y = [y1,y1+width,y1+width,y1,y1]

                            if k==1:                                
                                if ((j+1)%2==0) and ((i+1)%2==0):
                                    ax[0][3].plot(x, y, color= "red",linestyle = "solid", linewidth = 1)
                            elif k==2:
                                if ((j+1)%2==1) and ((i+1)%2==1):
                                    ax[0][3].plot(x, y, color= "yellow" ,linestyle = "solid", linewidth = 1)

                ax[0][3].imshow(Image.fromarray(image_reflected).convert("RGB"))
                # ###
                ax[0][4].set_title('Padding', fontsize=titelfontsize, fontname = titelfontname)
                ax[0][4].imshow(croped_imgs_combined.convert("RGB"))
                ###
                ax[0][5].set_title('Predict', fontsize=titelfontsize, fontname = titelfontname)
                ax[0][5].imshow(predicted_masks_combined.convert("RGB"))
                ### 2
                ax[1][0].set_title('Predicted Mask', fontsize=titelfontsize, fontname = titelfontname)
                ax[1][0].imshow(Image.fromarray(predicted_mask).convert("RGB"))
                ###
                ax[1][1].set_title('Mask', fontsize=titelfontsize, fontname = titelfontname)
                ax[1][1].imshow(result_mask.convert("RGB"))
                ###
                ax[1][2].set_title('ROI + Mask', fontsize=titelfontsize, fontname = titelfontname)
                ax[1][2].imshow(img_roi_mask.convert("RGB"))
                ###
                ax[1][3].set_title('ROI + Mask Edges', fontsize=titelfontsize, fontname = titelfontname)
                ax[1][3].imshow(Image.fromarray(img_roi_mask_edge).convert("RGB"))
                ###
                ax[1][4].set_title('Mask Edges', fontsize=titelfontsize, fontname = titelfontname)
                ax[1][4].imshow(mask_edges.convert("RGB"))
                ###
                ax[1][5].set_title('NoPadding', fontsize=titelfontsize, fontname = titelfontname)
                ax[1][5].imshow(Image.fromarray(predicted_mask_no_padding).convert("RGB"))
                ###
                plt.savefig(SAVE_PLOTS_FILES_PATH[index_Input_Image],facecolor=backcolor,bbox_inches='tight')  
                plt.show()
                
                # img_buf = io.BytesIO()
                # plt.savefig(img_buf, format='png',facecolor=backcolor,bbox_inches='tight')
                # im = Image.open(img_buf).convert("RGB")
                # #im.show()
                # im.save(SAVE_PLOTS_FILES_PATH[k])
                # img_buf.close()

                # plt.show()
                # #plt.pause(5)
                # plt.close()
          

            else:
                mask_img = crop(np.array(Image.open(IMGS_MASK_FILES_PATH[k])),ROW_MIN,HEIGHT,COL_MIN,WIDTH)


# %%
