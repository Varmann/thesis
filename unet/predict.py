# %%
from pathlib import Path
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import (
    plot_img_and_mask,
    plot_img_and_mask_save,
    plot_img_and_mask_save_3,
    crop,
    crop_with_padding,
    crop_without_padding,
)


# %%
def predict_img(
    net, full_img, device, scale_factor=1.0, out_threshold=0.5
):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        # plt.imshow(output[0,0,:,:])
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
        default=model_checkpoint_dv,
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="INPUT",
        default=image_file_paths,
        nargs="+",
        help="Filenames of input images",
        required=False,
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="OUTPUT",
        default=predicted_images_file_paths,
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
        default=mask_threshold_dv,
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
        "--classes", "-c", type=int, default=n_classes_dv, help="Number of classes"
    )
    (args, unknown) = parser.parse_known_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items() if k not in ['input', 'output']))

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
    #### Default values
    # Minimum probability value to consider a mask pixel white
    mask_threshold_dv = float(0.5)
    # checkpoint default path
    model_checkpoint_dv = Path(__file__).parent.resolve() / "checkpoints" / "checkpoint.pth"
    # Default classes number
    n_classes_dv = 2

    ### Constants
    Tile_Width = 200
    Tile_Padding = 50
    Crop_width = Tile_Width + 2 * Tile_Padding
    # Row
    Row_min = 350
    Row_max = 450 + 1400
    # Column
    Column_min = 230
    Column_max = 230 + 600

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

    args, unknown = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    in_files = args.input
    # out_files = get_output_filenames(args)
    out_files = predicted_images_file_paths
    # print(out_files)

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {(args.model.name)}")
    logging.info(f"Using device {device}")

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    for k, filename in enumerate(in_files):
        logging.info(f'Predicting image " {Path(filename).name} " ...')
        img = np.asarray(Image.open(filename))

        #####################################################################################
        logging.info("Croping with padding")
        # crop image
        y = Row_min
        x = Column_min
        image = img[y : y + 1400, x : x + 600]

        # Reflect / Crop
        croped_images, number_crops = crop_with_padding(image, Tile_Width, Tile_Padding)
        masks = {}
        h = number_crops[0]  # horizontal
        v = number_crops[1]  # vertical
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

        # Add together
        for i in range(0, v):
            image_add = crop(
                masks[h * i], Tile_Padding, Tile_Width, Tile_Padding, Tile_Width
            )
            for j in range(1, h):
                img_crop = crop(
                    masks[h * i + j], Tile_Padding, Tile_Width, Tile_Padding, Tile_Width
                )
                image_add = np.concatenate((image_add, img_crop), axis=1)
            if i == 0:
                new_image = image_add
            else:
                new_image = np.concatenate((new_image, image_add), axis=0)

        ###
        result = Image.fromarray(new_image)

        if not args.no_save:
            out_filename = out_files[k]

            result.save(out_filename)
            logging.info(f"Mask saved to {Path(out_filename).name}")

        if args.viz:
            logging.info(
                f"Visualizing results for image {Path(filename).name}, close to continue..."
            )
            plot_img_and_mask(image, result)

        # if vizualise_predict:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     #plot_img_and_mask(image, result)
        #     plot_img_and_mask_save(image, result, imgs_masks_file_path[k])

        #####################################################################################
        logging.info("Croping without padding")
        # Reflect / Crop  without padding
        croped_images2, number_crops2 = crop_without_padding(
            image, (Tile_Width + 2 * Tile_Padding)
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
        for i in range(0, v2):
            image_add2 = masks2[h2 * i]
            for j in range(1, h2):
                img_crop2 = masks2[h2 * i + j]
                image_add2 = np.concatenate((image_add2, img_crop2), axis=1)
            if i == 0:
                new_image2 = image_add2
            else:
                new_image2 = np.concatenate((new_image2, image_add2), axis=0)

        ###
        result2 = Image.fromarray(crop(new_image2, 0, 1400, 0, 600))

        if True:
            logging.info(
                f"Image and Masks saved to {Path(imgs_masks_file_path2[k]).name}"
            )
            logging.info(f'Visualizing Masks for image "{Path(filename).name}"')
            # plot_img_and_mask(image, result)
            plot_img_and_mask_save_3(image, result, result2, imgs_masks_file_path2[k])


# %%
