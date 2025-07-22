from PIL import Image
import numpy as np
import os
import argparse

'''
This script is used to extract the bright and dark part of the CT images.
parameters:
    img_folder: The path to the folder where Original BMP images are stored
    mask_folder: The path to the folder where Binary mask PNG images are stored
    output_folder: The path to the folder where different target region images will be saved
'''

def parser_init():
    parser = argparse.ArgumentParser(description='Extract bright and dark part of CT images')
    parser.add_argument('--sample_folder', type=str,
                        default=".//datasets//H351-1-0001", help='The path to the folder where original images are stored')
    parser.add_argument('--mask_folder', type=str,
                        default=".//datasets//H351-1-0001_pred", help='The path to the folder where binary mask images are stored')
    parser.add_argument('--output_folder', type=str,
                        default=".//datasets//H351-1-0001_output", help='The path to the folder where different target region images will be saved')
    return parser

def main():
    parser = parser_init()
    args = parser.parse_args()

    # folder path
    img_folder = args.sample_folder
    mask_folder = args.mask_folder
    output_folder = args.output_folder

    bright_part_save_folder = os.path.join(output_folder, "bright_part")
    dark_part_save_folder = os.path.join(output_folder, "dark_part")
    os.makedirs(bright_part_save_folder, exist_ok=True)
    os.makedirs(dark_part_save_folder, exist_ok=True)

    files_list = os.listdir(img_folder)

    # Category 1 and 2's Gray value
    bright_part = 85
    dark_part = 170

    # load original PNG image list and corresponding mask image list
    for img_files in files_list:
        if img_files.endswith(".png"):
            img_path = os.path.join(img_folder, img_files)
            mask_files = img_files.split('.png')[0] + '_pred.png'
            mask_path = os.path.join(mask_folder, mask_files)

            # Open PNG image and corresponding Binary mask image
            img_image = Image.open(img_path)
            mask_image = Image.open(mask_path)

            # Transform PNG image and Binary mask image to numpy array
            png_array = np.array(img_image)
            mask_array = np.array(mask_image)

            # find the target region in Binary mask image
            bright_target_mask = (mask_array == bright_part)
            dark_target_mask = (mask_array == dark_part)

            # extract the target region in PNG image
            bright_target_png_array = np.zeros_like(png_array)
            bright_target_png_array[bright_target_mask] = png_array[bright_target_mask]
            dark_target_png_array = np.zeros_like(png_array)
            dark_target_png_array[dark_target_mask] = png_array[dark_target_mask]

            # transform numpy array to PIL image
            bright_target_bmp_image = Image.fromarray(bright_target_png_array)
            dark_target_bmp_image = Image.fromarray(dark_target_png_array)

            # change output name and save path
            bright_target_output_path = os.path.join(bright_part_save_folder, img_files.replace('bmp', 'png'))
            dark_part_output_path = os.path.join(dark_part_save_folder, img_files.replace('bmp', 'png'))

            # save different target region in PNG image
            bright_target_bmp_image.save(bright_target_output_path)
            dark_target_bmp_image.save(dark_part_output_path)


if __name__ == '__main__':
    main()