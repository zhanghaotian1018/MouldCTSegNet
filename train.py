import argparse
import logging
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import time
import albumentations as albu
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.vision_transformer import SwinUnet as ViT_seg
from utils.data_load import MouldCTDataset
from utils.utils import DiceLoss
from config import get_config

parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_epochs', type=int,
                    default=350, help='maximum epoch number to train')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/MouldCTSegNet_train.yaml', metavar="FILE",
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

if __name__ == "__main__":
    os.makedirs(config.MODEL.MODEL_DIR, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')

    cudnn.benchmark = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)

    SAVE_DIR = config.MODEL.SAVE_DIR + '/' + str(
        time.localtime().tm_mon) + '_' + str(
        time.localtime().tm_mday) + '_' + str(
        time.localtime().tm_hour)

    print(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    logging.info('Start training')

    img_augmentation = albu.Compose([
        albu.ShiftScaleRotate(scale_limit=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.RandomBrightnessContrast(0.2, 0.2),
        ])
    ])

    train_dataset = MouldCTDataset(image_dir=config.MODEL.TRAIN_IMAGE_DIR,
                                   mask_dir=config.MODEL.TRAIN_MASK_DIR,
                                   augment=img_augmentation)
    val_dataset = MouldCTDataset(image_dir=config.MODEL.VAL_IMAGE_DIR,
                                 mask_dir=config.MODEL.VAL_MASK_DIR,
                                 augment=img_augmentation)
    logging.info('DeepLearning training started!')

    batch_size = args.batch_size * args.n_gpu
    num_workers = os.cpu_count()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    base_lr = args.base_lr
    num_classes = args.num_classes

    optimizer = optim.SGD(net.parameters(),
                          lr=base_lr,
                          momentum=0.9,
                          weight_decay=0.0001)
    ce_loss = nn.CrossEntropyLoss().to(device)
    dice_loss = DiceLoss(num_classes)

    writer = SummaryWriter(SAVE_DIR + '/log')
    train_iter_num = 0
    train_max_iterations = args.max_epochs * len(train_dataloader)
    val_iter_num = 0
    val_max_iterations = args.max_epochs * len(val_dataloader)

    best_loss = float('inf')
    max_epochs = args.max_epochs
    model = net
    MixedLossFunction = True
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_epoch_avg_total_loss = []
        train_epoch_avg_ce_loss = []
        train_epoch_avg_dice_loss = []
        train_epoch_avg_dice_score = []
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{max_epochs}', unit='img') as pbar:
            for i_batch, sample_batch in enumerate(train_dataloader):
                images, masks = sample_batch['image'], sample_batch['mask']
                images = images.to(device)
                masks = masks.to(device)
                pred = model(images)

                loss_ce = ce_loss(pred, torch.squeeze(masks, 1))
                train_epoch_avg_ce_loss.append(loss_ce.item())

                loss_dice = dice_loss(pred, torch.squeeze(masks, 1), softmax=True)
                train_epoch_avg_dice_loss.append(loss_dice.item())

                dice_score = 1.0 - loss_dice
                train_epoch_avg_dice_score.append(dice_score.item())

                if MixedLossFunction:
                    loss = 0.4 * loss_ce + 0.6 * loss_dice
                else:
                    loss = loss_ce

                train_epoch_avg_total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = base_lr * (1.0 - train_iter_num / train_max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                train_iter_num = train_iter_num + 1
                writer.add_scalar('train/step_lr', lr_, train_iter_num)
                writer.add_scalar('train/step_total_loss', loss, train_iter_num)
                writer.add_scalar('train/step_loss_ce', loss_ce, train_iter_num)
                writer.add_scalar('train/step_loss_dice', loss_dice, train_iter_num)
                writer.add_scalar('train/step_dice_score', dice_score, train_iter_num)

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss(batch)': loss.item()})

                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (train_iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

                if train_iter_num % 20 == 0:
                    image = images[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, train_iter_num)
                    outputs = torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, train_iter_num)
                    labs = masks[1, ...] * 50
                    writer.add_image('train/GroundTruth', labs, train_iter_num, dataformats='CHW')
        writer.add_scalar('train/epoch_avg_total_loss', np.mean(train_epoch_avg_total_loss), epoch)
        writer.add_scalar('train/epoch_avg_ce_loss', np.mean(train_epoch_avg_ce_loss), epoch)
        writer.add_scalar('train/epoch_avg_dice_loss', np.mean(train_epoch_avg_dice_loss), epoch)
        writer.add_scalar('train/epoch_avg_dice_score', np.mean(train_epoch_avg_dice_score), epoch)


        model.eval()
        with torch.no_grad():
            val_epoch_avg_total_loss = []
            val_epoch_avg_ce_loss = []
            val_epoch_avg_dice_loss = []
            val_epoch_avg_dice_score = []
            with tqdm(total=len(val_dataset), desc=f'Epoch {epoch}/{max_epochs}', unit='img') as pbar:
                for i_batch, sample_batch in enumerate(val_dataloader):
                    images, masks = sample_batch['image'], sample_batch['mask']
                    images = images.to(device)
                    masks = masks.to(device)
                    pred = model(images)

                    loss_ce = ce_loss(pred, torch.squeeze(masks, 1))
                    val_epoch_avg_ce_loss.append(loss_ce.item())

                    loss_dice = dice_loss(pred, torch.squeeze(masks, 1), softmax=True)
                    val_epoch_avg_dice_loss.append(loss_dice.item())

                    dice_score = 1.0 - loss_dice
                    val_epoch_avg_dice_score.append(dice_score.item())

                    if MixedLossFunction:
                        loss = 0.4 * loss_ce + 0.6 * loss_dice
                    else:
                        loss = loss_ce
                    val_epoch_avg_total_loss.append(loss.item())

                    val_iter_num = val_iter_num + 1
                    writer.add_scalar('val/step/total_loss', loss, val_iter_num)
                    writer.add_scalar('val/step/loss_ce', loss_ce, val_iter_num)
                    writer.add_scalar('val/step/loss_dice', loss_dice, val_iter_num)
                    writer.add_scalar('val/step/dice_score', dice_score, val_iter_num)

                    logging.info(
                        'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (val_iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

                    if val_iter_num % 20 == 0:
                        image = images[1, 0:1, :, :]
                        image = (image - image.min()) / (image.max() - image.min())
                        writer.add_image('val/Image', image, val_iter_num)
                        outputs = torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=True)
                        writer.add_image('val/Prediction', outputs[1, ...] * 50, val_iter_num)
                        labs = masks[1, ...] * 50
                        writer.add_image('val/GroundTruth', labs, val_iter_num)

                    pbar.update(images.shape[0])
                    pbar.set_postfix(**{'loss(batch)': loss.item()})

            writer.add_scalar('val/epoch_avg_total_loss', np.mean(val_epoch_avg_total_loss), epoch)
            writer.add_scalar('val/epoch_avg_ce_loss', np.mean(val_epoch_avg_ce_loss), epoch)
            writer.add_scalar('val/epoch_avg_dice_loss', np.mean(val_epoch_avg_dice_loss), epoch)
            writer.add_scalar('val/epoch_avg_dice_score', np.mean(val_epoch_avg_dice_score), epoch)

            if np.mean(val_epoch_avg_total_loss) < best_loss:
                best_loss = np.mean(val_epoch_avg_total_loss)
                torch.save(model.state_dict(), SAVE_DIR + os.sep + f'MouldCTSegNet_epoch{epoch}.pth')
                logging.info(f'epoch{epoch} Model saved successfully!')
            if epoch == max_epochs:
                torch.save(model.state_dict(), SAVE_DIR + os.sep + f'MouldCTSegNet_Last_epoch.pth')
                print('Last Model saved successfully!')


    writer.close()
    logging.info('Train Finished!')
