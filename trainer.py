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


class TrainingConfig:
    """Configuration class for training parameters"""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._setup_arguments()
    
    def _setup_arguments(self):
        """Setup command line arguments"""
        self.parser.add_argument('--num_classes', type=int, default=3, 
                                help='output channel of network')
        self.parser.add_argument('--batch_size', type=int, default=24, 
                                help='batch_size per gpu')
        self.parser.add_argument('--output_dir', type=str, help='output dir')
        self.parser.add_argument('--max_epochs', type=int, default=350, 
                                help='maximum epoch number to train')
        self.parser.add_argument('--n_gpu', type=int, default=1, 
                                help='total gpu')
        self.parser.add_argument('--deterministic', type=int, default=1,
                                help='whether use deterministic training')
        self.parser.add_argument('--base_lr', type=float, default=0.01,
                                help='segmentation network learning rate')
        self.parser.add_argument('--img_size', type=int, default=224, 
                                help='input patch size of network input')
        self.parser.add_argument('--seed', type=int, default=1234, 
                                help='random seed')
        self.parser.add_argument('--cfg', type=str, 
                                default='./configs/MouldCTSegNet_train.yaml', 
                                metavar="FILE", help='path to config file')
        self.parser.add_argument("--opts", help="Modify config options", 
                                default=None, nargs='+')
        self.parser.add_argument('--zip', action='store_true', 
                                help='use zipped dataset instead of folder dataset')
        self.parser.add_argument('--cache-mode', type=str, default='part', 
                                choices=['no', 'full', 'part'],
                                help='cache mode for dataset')
        self.parser.add_argument('--resume', help='resume from checkpoint')
        self.parser.add_argument('--accumulation-steps', type=int, 
                                help="gradient accumulation steps")
        self.parser.add_argument('--use-checkpoint', action='store_true',
                                help="use gradient checkpointing to save memory")
        self.parser.add_argument('--amp-opt-level', type=str, default='O1', 
                                choices=['O0', 'O1', 'O2'],
                                help='mixed precision opt level')
        self.parser.add_argument('--tag', help='tag of experiment')
        self.parser.add_argument('--eval', action='store_true', 
                                help='Perform evaluation only')
        self.parser.add_argument('--throughput', action='store_true', 
                                help='Test throughput only')
    
    def get_config(self):
        """Parse arguments and return configuration"""
        args = self.parser.parse_args()
        config = get_config(args)
        return args, config


class DataManager:
    """Manages data loading and augmentation"""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self._setup_augmentation()
    
    def _setup_augmentation(self):
        """Setup data augmentation pipeline"""
        self.img_augmentation = albu.Compose([
            albu.ShiftScaleRotate(scale_limit=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.OneOf([
                albu.RandomBrightnessContrast(0.2, 0.2),
            ])
        ])
    
    def create_dataloaders(self):
        """Create train and validation dataloaders"""
        # Create datasets
        train_dataset = MouldCTDataset(
            image_dir=self.config.MODEL.TRAIN_IMAGE_DIR,
            mask_dir=self.config.MODEL.TRAIN_MASK_DIR,
            augment=self.img_augmentation
        )
        val_dataset = MouldCTDataset(
            image_dir=self.config.MODEL.VAL_IMAGE_DIR,
            mask_dir=self.config.MODEL.VAL_MASK_DIR,
            augment=self.img_augmentation
        )
        
        # Calculate batch size and number of workers
        batch_size = self.args.batch_size * self.args.n_gpu
        num_workers = os.cpu_count()
        
        # Create dataloaders
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
        
        return train_dataloader, val_dataloader


class ModelManager:
    """Manages model initialization and setup"""
    
    def __init__(self, config, args, device):
        self.config = config
        self.args = args
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
    
    def setup_model(self):
        """Initialize model and load pretrained weights"""
        self.model = ViT_seg(self.config, img_size=self.args.img_size, 
                            num_classes=self.args.num_classes).cuda()
        self.model.load_from(self.config)
        return self.model
    
    def setup_optimizer(self):
        """Setup optimizer and loss functions"""
        # Setup optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.args.base_lr,
            momentum=0.9,
            weight_decay=0.0001
        )
        
        # Setup loss functions
        ce_loss = nn.CrossEntropyLoss().to(self.device)
        dice_loss = DiceLoss(self.args.num_classes)
        
        return self.optimizer, ce_loss, dice_loss


class TrainingMonitor:
    """Manages training monitoring and logging"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.writer = None
        self.best_loss = float('inf')
    
    def setup_logging(self):
        """Setup TensorBoard logging"""
        self.writer = SummaryWriter(self.save_dir + '/log')
        return self.writer
    
    def update_best_loss(self, current_loss, epoch, model):
        """Update best loss and save model if improved"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            torch.save(model.state_dict(), 
                      self.save_dir + os.sep + f'MouldCTSegNet_epoch{epoch}.pth')
            logging.info(f'epoch{epoch} Model saved successfully!')
            return True
        return False
    
    def save_final_model(self, model, epoch):
        """Save final model after training"""
        torch.save(model.state_dict(), 
                  self.save_dir + os.sep + f'MouldCTSegNet_Last_epoch.pth')
        logging.info('Final Model saved successfully!')


class Trainer:
    """Main training class that orchestrates the training process"""
    
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        
        # Initialize managers
        self.data_manager = DataManager(config, args)
        self.model_manager = ModelManager(config, args, device)
        
        # Setup save directory
        self.save_dir = self._create_save_dir()
        self.monitor = TrainingMonitor(self.save_dir)
        
        # Training state
        self.train_iter_num = 0
        self.val_iter_num = 0
        self.MixedLossFunction = True
    
    def _create_save_dir(self):
        """Create directory for saving models and logs"""
        save_dir = self.config.MODEL.SAVE_DIR + '/' + str(
            time.localtime().tm_mon) + '_' + str(
            time.localtime().tm_mday) + '_' + str(
            time.localtime().tm_hour)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    
    def setup_training(self):
        """Setup all components for training"""
        logging.info('Starting training setup...')
        
        # Setup model
        model = self.model_manager.setup_model()
        
        # Setup data
        train_dataloader, val_dataloader = self.data_manager.create_dataloaders()
        
        # Setup optimizer and loss
        optimizer, ce_loss, dice_loss = self.model_manager.setup_optimizer()
        
        # Setup monitoring
        writer = self.monitor.setup_logging()
        
        # Calculate iteration numbers
        train_max_iterations = self.args.max_epochs * len(train_dataloader)
        val_max_iterations = self.args.max_epochs * len(val_dataloader)
        
        return {
            'model': model,
            'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader,
            'optimizer': optimizer,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
            'writer': writer,
            'train_max_iterations': train_max_iterations,
            'val_max_iterations': val_max_iterations
        }
    
    def train_epoch(self, epoch, model, train_dataloader, optimizer, 
                   ce_loss, dice_loss, writer, train_max_iterations):
        """Train for one epoch"""
        model.train()
        train_epoch_avg_total_loss = []
        train_epoch_avg_ce_loss = []
        train_epoch_avg_dice_loss = []
        train_epoch_avg_dice_score = []
        
        with tqdm(total=len(train_dataloader.dataset), 
                 desc=f'Epoch {epoch}/{self.args.max_epochs}', 
                 unit='img') as pbar:
            
            for i_batch, sample_batch in enumerate(train_dataloader):
                # Move data to device
                images, masks = sample_batch['image'], sample_batch['mask']
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                pred = model(images)
                
                # Calculate losses
                loss_ce = ce_loss(pred, torch.squeeze(masks, 1))
                loss_dice = dice_loss(pred, torch.squeeze(masks, 1), softmax=True)
                dice_score = 1.0 - loss_dice
                
                # Combined loss
                if self.MixedLossFunction:
                    loss = 0.4 * loss_ce + 0.6 * loss_dice
                else:
                    loss = loss_ce
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update learning rate
                lr_ = self.args.base_lr * (1.0 - self.train_iter_num / train_max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                
                # Update metrics
                train_epoch_avg_total_loss.append(loss.item())
                train_epoch_avg_ce_loss.append(loss_ce.item())
                train_epoch_avg_dice_loss.append(loss_dice.item())
                train_epoch_avg_dice_score.append(dice_score.item())
                
                # Update iteration counter
                self.train_iter_num += 1
                
                # Log step metrics
                self._log_step_metrics(writer, 'train', lr_, loss, loss_ce, 
                                      loss_dice, dice_score, self.train_iter_num)
                
                # Update progress bar
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss(batch)': loss.item()})
                
                # Log training info
                logging.info(f'iteration {self.train_iter_num} : loss : {loss.item():.6f}, '
                           f'loss_ce: {loss_ce.item():.6f}, loss_dice: {loss_dice.item():.6f}')
                
                # Log images every 20 steps
                if self.train_iter_num % 20 == 0:
                    self._log_images(writer, 'train', images, pred, masks, self.train_iter_num)
        
        # Log epoch metrics
        self._log_epoch_metrics(writer, 'train', 
                               np.mean(train_epoch_avg_total_loss),
                               np.mean(train_epoch_avg_ce_loss),
                               np.mean(train_epoch_avg_dice_loss),
                               np.mean(train_epoch_avg_dice_score),
                               epoch)
        
        return np.mean(train_epoch_avg_total_loss)
    
    def validate_epoch(self, epoch, model, val_dataloader, ce_loss, dice_loss, writer):
        """Validate for one epoch"""
        model.eval()
        val_epoch_avg_total_loss = []
        val_epoch_avg_ce_loss = []
        val_epoch_avg_dice_loss = []
        val_epoch_avg_dice_score = []
        
        with torch.no_grad():
            with tqdm(total=len(val_dataloader.dataset), 
                     desc=f'Validation Epoch {epoch}/{self.args.max_epochs}', 
                     unit='img') as pbar:
                
                for i_batch, sample_batch in enumerate(val_dataloader):
                    # Move data to device
                    images, masks = sample_batch['image'], sample_batch['mask']
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    # Forward pass
                    pred = model(images)
                    
                    # Calculate losses
                    loss_ce = ce_loss(pred, torch.squeeze(masks, 1))
                    loss_dice = dice_loss(pred, torch.squeeze(masks, 1), softmax=True)
                    dice_score = 1.0 - loss_dice
                    
                    # Combined loss
                    if self.MixedLossFunction:
                        loss = 0.4 * loss_ce + 0.6 * loss_dice
                    else:
                        loss = loss_ce
                    
                    # Update metrics
                    val_epoch_avg_total_loss.append(loss.item())
                    val_epoch_avg_ce_loss.append(loss_ce.item())
                    val_epoch_avg_dice_loss.append(loss_dice.item())
                    val_epoch_avg_dice_score.append(dice_score.item())
                    
                    # Update iteration counter
                    self.val_iter_num += 1
                    
                    # Log step metrics
                    self._log_step_metrics(writer, 'val', None, loss, loss_ce, 
                                          loss_dice, dice_score, self.val_iter_num)
                    
                    # Log validation info
                    logging.info(f'val iteration {self.val_iter_num} : loss : {loss.item():.6f}, '
                               f'loss_ce: {loss_ce.item():.6f}, loss_dice: {loss_dice.item():.6f}')
                    
                    # Log images every 20 steps
                    if self.val_iter_num % 20 == 0:
                        self._log_images(writer, 'val', images, pred, masks, self.val_iter_num)
                    
                    # Update progress bar
                    pbar.update(images.shape[0])
                    pbar.set_postfix(**{'loss(batch)': loss.item()})
        
        # Log epoch metrics
        self._log_epoch_metrics(writer, 'val',
                               np.mean(val_epoch_avg_total_loss),
                               np.mean(val_epoch_avg_ce_loss),
                               np.mean(val_epoch_avg_dice_loss),
                               np.mean(val_epoch_avg_dice_score),
                               epoch)
        
        return np.mean(val_epoch_avg_total_loss)
    
    def _log_step_metrics(self, writer, phase, lr, loss, loss_ce, loss_dice, dice_score, iteration):
        """Log metrics for each training/validation step"""
        if lr is not None:
            writer.add_scalar(f'{phase}/step_lr', lr, iteration)
        writer.add_scalar(f'{phase}/step_total_loss', loss, iteration)
        writer.add_scalar(f'{phase}/step_loss_ce', loss_ce, iteration)
        writer.add_scalar(f'{phase}/step_loss_dice', loss_dice, iteration)
        writer.add_scalar(f'{phase}/step_dice_score', dice_score, iteration)
    
    def _log_epoch_metrics(self, writer, phase, total_loss, ce_loss, dice_loss, dice_score, epoch):
        """Log metrics for each epoch"""
        writer.add_scalar(f'{phase}/epoch_avg_total_loss', total_loss, epoch)
        writer.add_scalar(f'{phase}/epoch_avg_ce_loss', ce_loss, epoch)
        writer.add_scalar(f'{phase}/epoch_avg_dice_loss', dice_loss, epoch)
        writer.add_scalar(f'{phase}/epoch_avg_dice_score', dice_score, epoch)
    
    def _log_images(self, writer, phase, images, pred, masks, iteration):
        """Log sample images, predictions and ground truth"""
        image = images[1, 0:1, :, :]
        image = (image - image.min()) / (image.max() - image.min())
        writer.add_image(f'{phase}/Image', image, iteration)
        
        outputs = torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=True)
        writer.add_image(f'{phase}/Prediction', outputs[1, ...] * 50, iteration)
        
        labs = masks[1, ...] * 50
        writer.add_image(f'{phase}/GroundTruth', labs, iteration, 
                        dataformats='CHW' if phase == 'train' else None)
    
    def train(self):
        """Main training loop"""
        logging.info('DeepLearning training started!')
        
        # Setup training components
        components = self.setup_training()
        
        # Extract components
        model = components['model']
        train_dataloader = components['train_dataloader']
        val_dataloader = components['val_dataloader']
        optimizer = components['optimizer']
        ce_loss = components['ce_loss']
        dice_loss = components['dice_loss']
        writer = components['writer']
        train_max_iterations = components['train_max_iterations']
        
        # Training loop
        for epoch in range(1, self.args.max_epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(epoch, model, train_dataloader, optimizer,
                                        ce_loss, dice_loss, writer, train_max_iterations)
            
            # Validate for one epoch
            val_loss = self.validate_epoch(epoch, model, val_dataloader, 
                                         ce_loss, dice_loss, writer)
            
            # Update best model
            self.monitor.update_best_loss(val_loss, epoch, model)
            
            # Save final model at last epoch
            if epoch == self.args.max_epochs:
                self.monitor.save_final_model(model, epoch)
        
        # Close writer and log completion
        writer.close()
        logging.info('Train Finished!')


def main():
    """Main function to run the training pipeline"""
    # Setup configuration
    config_manager = TrainingConfig()
    args, config = config_manager.get_config()
    
    # Create model directory
    os.makedirs(config.MODEL.MODEL_DIR, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')
    
    # Enable cuDNN benchmark for better performance
    cudnn.benchmark = True
    
    # Initialize and run trainer
    trainer = Trainer(args, config, device)
    trainer.train()


if __name__ == "__main__":
    main()