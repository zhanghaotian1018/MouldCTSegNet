import argparse
import logging
import os
from glob import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config

parser = argparse.ArgumentParser()

# change dataset to mould CT dataset
parser.add_argument('--root_path', type=str,
                    default=r'.//datasets//H351-1-0001', help='root dir for data')
parser.add_argument('--output_dir', type=str,
                    default=r'.//datasets//H351-1-0001_pred', help='output dir')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/MouldCTSegNet.yaml', metavar="FILE",
                    help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)


def pred_images(model,
                device,
                pre_dir,
                save_dir):
    global new_h, new_w
    colorKey = {
        'bright': [0, 128, 0],  # 绿色
        'dark': [128, 0, 0],  # 红色
        'background': [0, 0, 0]
    }
    test_imgs = sorted(glob(pre_dir + os.sep + '*.png'))

    for idx, test_img in enumerate(test_imgs):
        basename = os.path.basename(test_img).split('.png')[0]

        img = cv2.imread(test_img)

        # ----------------------------------------------------------------
        oldh, oldw = img.shape[:2]
        scale = 224.0 * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        target_size = (newh, neww)
        img = torch.from_numpy(img)
        img_array = resize(img.unsqueeze(0).permute((0, 3, 1, 2)), target_size)
        img_array = img_array.squeeze(0)
        # Pad
        h, w = img_array.shape[-2:]  # size of input image
        # Input of Image Encoder: 1024
        padh = 224 - h
        padw = 224 - w
        img_array = F.pad(img_array, (0, padw, 0, padh))  # size=[1, 3, 1024, 1024]

        img_array = np.asarray(img_array)
        # mean and std of samples
        mean = 0.14071481
        std = 0.19077588
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
        img_array = (img_array - mean) / std

        img = torch.from_numpy(img_array.astype(np.float32))
        img = img[np.newaxis, ...]
        img = img.to(device)

        model.eval()

        with torch.no_grad():
            output = model(img)

        output = nn.Softmax2d()(output)
        output = output.argmax(dim=1)
        output = output.data.cpu().numpy()
        output = np.squeeze(output)

        # translate output to colorful mask
        mask = np.zeros((output.shape[0], output.shape[1], 3))

        mask[output == 0] = colorKey['background']
        mask[output == 1] = colorKey['bright']
        mask[output == 2] = colorKey['dark']

        mask = mask.astype(np.uint8)
        mask = torch.Tensor(mask)
        masks = F.interpolate(
            mask.permute((2, 0, 1)).unsqueeze(0),
            (224, 224),
            mode="bilinear",
            align_corners=False
        )
        masks = masks[..., : target_size[0], : target_size[1]]
        masks = F.interpolate(masks, (oldh, oldw), mode="bilinear", align_corners=False)
        masks = np.asarray(masks.squeeze(0).permute((1, 2, 0)))

        b = masks[:, :, 0]
        g = masks[:, :, 1]
        r = masks[:, :, 2]
        # bright --> green
        idx1_r = r == 0
        idx1_g = g >= 64
        idx1_b = b < 64
        # dark --> red
        idx2_r = r == 0
        idx2_g = g < 64
        idx2_b = b >= 64
        algoMask = np.zeros((r.shape[0], r.shape[1]), dtype=np.uint8)
        idx1 = idx1_r & idx1_g & idx1_b
        idx2 = idx2_r & idx2_g & idx2_b
        algoMask[idx1] = 85  # bright
        algoMask[idx2] = 170  # dark

        # save binary mask
        save_dir_name = os.path.join(save_dir, basename + '_pred.png')
        cv2.imwrite(save_dir_name, algoMask)

        logging.info(f'Mask saved to {save_dir_name}')
    logging.info(f'pred finished')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    pre_dir = args.root_path
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # SwinUet model
    model_dir = r'./checkpoint'

    # model
    model_path = model_dir + os.sep + 'MouldCTSegNet.pth'

    # SwinUnet
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes)
    model.to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded successfully')

    pred_images(model=model,
                device=device,
                pre_dir=pre_dir,
                save_dir=save_dir)
