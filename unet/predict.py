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

    for k, filename in enumerate(in_files):
        logging.info(f'Predicting image -> {Path(filename).name} ...')
        img = np.array(Image.open(filename))

        #####################################################################################
        logging.info("Croping with padding")
        # crop image
        image = img[ROW_MIN : ROW_MIN + HEIGHT, COL_MIN : COL_MIN + WIDTH]
        #plt.imshow(image)
        # Reflect / Crop
        image_reflected, croped_images, number_crops = crop_with_padding(image, TILE_WIDTH, TILE_PADDING)
        masks = {}
        h = number_crops[0]  # horizontal
        v = number_crops[1]  # vertical

        imgs_combined = pil_images_combine(croped_images ,(TILE_WIDTH + 2*TILE_PADDING),(TILE_WIDTH + 2*TILE_PADDING), h, v , space = 10, bachground = "white" )
        #imgs_combined.show()
        # Predict every part      
        for i in range(0, v):
            for j in range(0, h):
                mask = predict_img(
                    net=net,
                    full_img=Image.fromarray(croped_images[h * i + j]),
                    scale_factor=args.scale,
                    out_threshold=args.mask_threshold,
                    device=device,
                )
                temp_mask = mask_to_image(mask, mask_values)
                masks[h * i + j] = np.array(temp_mask)
        
        
        masks_combined = pil_images_combine(masks ,(TILE_WIDTH + 2*TILE_PADDING),(TILE_WIDTH + 2*TILE_PADDING), h, v , space = 10 , bachground = "lightgray" )
        #masks_combined.show()
        # img_list =[image,image_reflected,np.array(imgs_combined),np.array(masks_combined)]
        # all_images_combined = pil_images_combine(img_list ,imgs_combined.size[0],imgs_combined.size[1], 4, 1 , space = 10 , bachground = "black" )
        # all_images_combined.show()
        # all_images_combined = Image.new(mode="L",size=(masks_combined.size[0]- 200,masks_combined.size[1]), color="black")
        # all_images_combined.paste(Image.fromarray(image),(200,300))
        # all_images_combined.paste(Image.fromarray(image_reflected),(masks_combined.size[0]-200+50,200))
        # all_images_combined.paste(imgs_combined,(2*masks_combined.size[0]-200+100,0))
        # all_images_combined.paste(masks_combined,(3*masks_combined.size[0]-200+150,0))
        # #all_images_combined.show()
        # all_images_combined.save(SAVE_PLOTS_IMAGE_FILES_PATH[k])

        # logging.info("Showing Image, Reflected Image , Image Padding, Predict Padding")
        # fig2, ax2 = plt.subplots(1, 1,facecolor = "lightgrey", dpi = 600)        
        # plt.style.use('grayscale')
        # ax2.imshow(all_images_combined)
        # plt.show()
        new_input_image_pil = Image.new(mode="L",size= (imgs_combined.size[0],imgs_combined.size[1]), color="black")
        new_input_image_pil.paste(Image.fromarray(img),(200,200))
        new_image_pil = Image.new(mode="L",size= (imgs_combined.size[0],imgs_combined.size[1]), color="black")
        new_image_pil.paste(Image.fromarray(image),(300,400))
        new_image_reflected_pil = Image.new(mode="L",size= (imgs_combined.size[0],imgs_combined.size[1]), color="black")
        new_image_reflected_pil.paste(Image.fromarray(image_reflected),(200,300))
        logging.info("Showing Image, Reflected Image , Image Padding, Predict Padding")
        plot_reflected_save(new_input_image_pil,new_image_pil,new_image_reflected_pil,imgs_combined,masks_combined,SAVE_PLOTS_IMAGE_FILES_PATH[k], show = True)   

        # Add together
        img_add_h = []
        img_add_v = []
        for i in range(0, v):
            # Add horizontal
            img_add_h = crop(
                masks[h * i], TILE_PADDING, TILE_WIDTH, TILE_PADDING, TILE_WIDTH
            )
            for j in range(1, h):
                img_crop = crop(
                    masks[h * i + j], TILE_PADDING, TILE_WIDTH, TILE_PADDING, TILE_WIDTH
                )
                img_add_h = np.concatenate((img_add_h, img_crop), axis=1)
            # Add vertical
            if i == 0:
                img_add_v = img_add_h
            else:
                img_add_v = np.concatenate((img_add_v, img_add_h), axis=0)

        predicted_mask = img_add_v[0 : HEIGHT, 0 : WIDTH]
        ###
        result = Image.fromarray(predicted_mask)
        ## Edges of Segment in predicted mask.
        mask_edges = result.filter(ImageFilter.FIND_EDGES)
        mask_edges = mask_edges.filter(ImageFilter.MaxFilter)
        print(image.shape)
        background = Image.fromarray(image)
        
        output_rgb = image[..., None].repeat(3, axis=-1)
        mask_array = np.array(mask_edges)
        for i in range(3):
            val = 255 if i == 0 else 0
            output_rgb[..., i] = np.where(mask_array > 0, val, 0)
        
        
        plt.imshow(output_rgb)


        #mask_edges.paste(background, (0, 0), background)

        # pixels = list(result.getdata())
        # result.show()
        if not args.no_save:
            out_filename = out_files[k]
            result.save(out_filename)
            logging.info(f" {Path(out_filename).name}  Mask saved to -> {Path(out_filename)}  ")

        if args.viz:
            logging.info(
                f"Visualizing results for image -> {Path(filename).name}, close to continue..."
            )
            plot_img_and_mask(image, predicted_mask)

        #####################################################################################
        logging.info("Croping without padding")
        # Reflect / Crop  without padding
        croped_images2, number_crops2 = crop_without_padding(
            image, (TILE_WIDTH + 2 * TILE_PADDING)
        )
        masks2 = {}
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
                masks2[h2 * i + j] = np.array(temp_mask2)

        # Add together
        img_add_h_2 = []
        img_add_v_2 = []
        for i in range(0, v2):
            # Add horizontal
            img_add_h_2 = masks2[h2 * i]
            for j in range(1, h2):
                imgs_crop_2 = masks2[h2 * i + j]
                img_add_h_2 = np.concatenate((img_add_h_2, imgs_crop_2), axis=1)
            # Add vertical
            if i == 0:
                img_add_v_2 = img_add_h_2
            else:
                img_add_v_2 = np.concatenate((img_add_v_2, img_add_h_2), axis=0)

        predicted_mask_no_padding = img_add_v_2
        ###
        result2 = Image.fromarray(crop(predicted_mask_no_padding, 0, HEIGHT, 0, WIDTH))
        
        if True:
            logging.info(
                f" {Path(SAVE_PLOTS_PREDICT_FILES_PATH[k]).name} Image and Masks plots saved to -> {Path(SAVE_PLOTS_PREDICT_FILES_PATH[k])} "
            )
            logging.info(f'Visualizing Masks for image -> {Path(filename).name}')
            # plot_img_and_mask(image, result)
            mask_img = crop(
                np.array(Image.open(IMGS_MASK_FILES_PATH[k])),
                ROW_MIN,
                HEIGHT,
                COL_MIN,
                WIDTH,
            )
            predicted_mask_no_padding = crop(
                predicted_mask_no_padding, 0, HEIGHT, 0, WIDTH
            )


            if  (images_not_trained):
                plot_img_imgmsk_predicts_save(
                    image,
                    output_rgb,
                    predicted_mask,
                    predicted_mask_no_padding,
                    SAVE_PLOTS_PREDICT_FILES_PATH[k], 
                    show = True
                )                
            else:
                plot_img_imgmsk_mask_predicts_save(
                    image,
                    output_rgb,
                    mask_img,
                    predicted_mask,
                    predicted_mask_no_padding,
                    SAVE_PLOTS_PREDICT_FILES_PATH[k], 
                    show = False
                )


# %%
