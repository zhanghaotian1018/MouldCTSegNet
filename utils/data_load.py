import os
from glob import glob

import cv2
import numpy as np
from torch.utils.data import Dataset


class MouldCTDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 augment=None):
        self.augment = augment
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images_name = sorted(glob(self.image_dir + os.sep + '*.png'))
        self.mask_name = sorted(glob(self.mask_dir + os.sep + '*.png'))

    def __len__(self):
        return len(self.images_name)

    @staticmethod
    def preprocess(input, is_mask):
        img = np.asarray(input)

        if is_mask:
            # get mask
            b = img[:, :, 0]
            g = img[:, :, 1]
            r = img[:, :, 2]

            # bright --> green
            idx1_1 = r == 0
            idx2_1 = g == 128
            idx3_1 = b == 0

            # dark --> red
            idx1_2 = r == 128
            idx2_2 = g == 0
            idx3_2 = b == 0

            algoMask = np.zeros((r.shape[0], r.shape[1]), dtype=np.uint8)
            idx1 = idx1_1 & idx2_1 & idx3_1
            idx2 = idx1_2 & idx2_2 & idx3_2
            algoMask[idx1] = 1
            algoMask[idx2] = 2

            return algoMask[np.newaxis, ...]

        else:
            img = img.transpose((2, 0, 1))  # C H W

            return img

    def __getitem__(self, idx):
        image = cv2.imread(self.images_name[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mid_output_shape = (224, 224)
        image = cv2.resize(image, mid_output_shape, interpolation=cv2.INTER_NEAREST)

        # do normalization for each channel
        mean = 0.14071481
        std = 0.19077588
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = (image - mean) / std

        mask = cv2.imread(self.mask_name[idx])
        mask = cv2.resize(mask, mid_output_shape, interpolation=cv2.INTER_NEAREST)

        image = self.preprocess(input=image, is_mask=False)
        mask = self.preprocess(input=mask, is_mask=True)

        image = image.astype(np.float32)

        sample = self.augment(image=image.transpose((1, 2, 0)), mask=mask.transpose((1, 2, 0)))
        image, mask = sample['image'].transpose((2, 0, 1)), sample['mask'].transpose((2, 0, 1))

        sample = {
            'image': image.astype(np.float32),
            'mask': mask.astype(np.int64)
        }

        return sample

