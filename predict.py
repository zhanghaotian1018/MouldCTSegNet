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


class PredictionConfig:
    """Configuration class for prediction parameters"""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._setup_arguments()
    
    def _setup_arguments(self):
        """Setup command line arguments for prediction"""
        self.parser.add_argument('--root_path', type=str,
                                default=r'./datasets/H351-1-0001', 
                                help='root directory for input data')
        self.parser.add_argument('--output_dir', type=str,
                                default=r'./datasets/H351-1-0001_pred', 
                                help='output directory for predictions')
        self.parser.add_argument('--batch_size', type=int,
                                default=1, help='batch_size per gpu')
        self.parser.add_argument('--num_classes', type=int,
                                default=3, help='output channel of network')
        self.parser.add_argument('--img_size', type=int,
                                default=224, help='input image size for network')
        self.parser.add_argument('--model_path', type=str,
                                default=r'./checkpoint/MouldCTSegNet_best.pth',
                                help='path to trained model checkpoint')
        self.parser.add_argument('--cfg', type=str, 
                                default='./configs/MouldCTSegNet_predict.yaml', 
                                metavar="FILE", help='path to config file')
        self.parser.add_argument("--opts", help="Modify config options", 
                                default=None, nargs='+')
        self.parser.add_argument('--cache-mode', type=str, default='part', 
                                choices=['no', 'full', 'part'],
                                help='cache mode for dataset')
        self.parser.add_argument('--resume', type=str, default=None,
                                help='path to checkpoint to resume from')
        self.parser.add_argument('--amp-opt-level', type=str, default='O1', 
                                choices=['O0', 'O1', 'O2'],
                                help='mixed precision opt level')
    
    def get_config(self):
        """Parse arguments and return configuration"""
        args = self.parser.parse_args()
        config = get_config(args)
        return args, config


class ModelManager:
    """Manages model loading and setup for prediction"""
    
    def __init__(self, config, args, device):
        self.config = config
        self.args = args
        self.device = device
        self.model = None
    
    def load_model(self):
        """Load trained model from checkpoint"""
        # Initialize model
        self.model = ViT_seg(self.config, img_size=self.args.img_size, 
                           num_classes=self.args.num_classes)
        self.model.to(self.device)
        
        # Load model weights
        if not os.path.isfile(self.args.model_path):
            raise FileNotFoundError(f"Model file not found: {self.args.model_path}")
        
        state_dict = torch.load(self.args.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        logging.info('Model loaded successfully')
        return self.model


class ImageProcessor:
    """Handles image preprocessing and postprocessing"""
    
    def __init__(self, img_size=224):
        self.img_size = img_size
        # Color mapping for different classes
        self.color_key = {
            'bright': [0, 128, 0],    # Green
            'dark': [128, 0, 0],      # Red  
            'background': [0, 0, 0]    # Black
        }
        # Image normalization parameters
        self.mean = 0.14071481
        self.std = 0.19077588
    
    def preprocess_image(self, image_path):
        """
        Preprocess input image for model prediction
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed tensor and original image dimensions
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Store original dimensions
        old_height, old_width = img.shape[:2]
        
        # Resize image while maintaining aspect ratio
        scale = self.img_size * 1.0 / max(old_height, old_width)
        new_height, new_width = old_height * scale, old_width * scale
        new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
        target_size = (new_height, new_width)
        
        # Convert to tensor and resize
        img_tensor = torch.from_numpy(img)
        img_array = resize(img_tensor.unsqueeze(0).permute((0, 3, 1, 2)), target_size)
        img_array = img_array.squeeze(0)
        
        # Pad to target size
        height, width = img_array.shape[-2:]
        pad_height = self.img_size - height
        pad_width = self.img_size - width
        img_array = F.pad(img_array, (0, pad_width, 0, pad_height))
        
        # Normalize image
        img_array = np.asarray(img_array)
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
        img_array = (img_array - self.mean) / self.std
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array.astype(np.float32))
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor, (old_height, old_width), target_size
    
    def postprocess_prediction(self, prediction, original_size, target_size):
        """
        Convert model prediction to color mask and resize to original dimensions
        
        Args:
            prediction: Model output tensor
            original_size: Original image dimensions (height, width)
            target_size: Target size after preprocessing
            
        Returns:
            Color mask as numpy array
        """
        # Apply softmax and get class predictions
        output = nn.Softmax2d()(prediction)
        output = output.argmax(dim=1)
        output = output.data.cpu().numpy()
        output = np.squeeze(output)
        
        # Create color mask
        mask = np.zeros((output.shape[0], output.shape[1], 3))
        mask[output == 0] = self.color_key['background']
        mask[output == 1] = self.color_key['bright']
        mask[output == 2] = self.color_key['dark']
        mask = mask.astype(np.uint8)
        
        # Resize mask to original dimensions
        mask_tensor = torch.Tensor(mask)
        mask_resized = F.interpolate(
            mask_tensor.permute((2, 0, 1)).unsqueeze(0),
            self.img_size,
            mode="bilinear",
            align_corners=False
        )
        
        # Crop to target size and resize to original dimensions
        mask_resized = mask_resized[..., :target_size[0], :target_size[1]]
        mask_resized = F.interpolate(mask_resized, original_size, 
                                   mode="bilinear", align_corners=False)
        mask_final = np.asarray(mask_resized.squeeze(0).permute((1, 2, 0)))
        
        return mask_final
    
    def create_binary_mask(self, color_mask):
        """
        Convert color mask to binary mask for specific classes
        
        Args:
            color_mask: Color mask from postprocessing
            
        Returns:
            Binary mask with class labels
        """
        # Extract color channels
        blue = color_mask[:, :, 0]
        green = color_mask[:, :, 1]
        red = color_mask[:, :, 2]
        
        # Create binary mask
        binary_mask = np.zeros((red.shape[0], red.shape[1]), dtype=np.uint8)
        
        # Identify bright regions (green)
        bright_mask = (red == 0) & (green >= 64) & (blue < 64)
        # Identify dark regions (red)  
        dark_mask = (red == 0) & (green < 64) & (blue >= 64)
        
        # Assign class labels
        binary_mask[bright_mask] = 85   # Bright class
        binary_mask[dark_mask] = 170    # Dark class
        
        return binary_mask


class Predictor:
    """Main prediction class that orchestrates the prediction process"""
    
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        
        # Initialize managers
        self.model_manager = ModelManager(config, args, device)
        self.image_processor = ImageProcessor(img_size=args.img_size)
        
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
    
    def setup_prediction(self):
        """Setup all components for prediction"""
        logging.info('Starting prediction setup...')
        
        # Load model
        model = self.model_manager.load_model()
        model.eval()
        
        return model
    
    def predict_single_image(self, model, image_path):
        """
        Perform prediction on a single image
        
        Args:
            model: Loaded model
            image_path: Path to input image
            
        Returns:
            Prediction results
        """
        # Preprocess image
        img_tensor, original_size, target_size = self.image_processor.preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)
        
        # Perform prediction
        with torch.no_grad():
            prediction = model(img_tensor)
        
        # Postprocess prediction
        color_mask = self.image_processor.postprocess_prediction(prediction, original_size, target_size)
        binary_mask = self.image_processor.create_binary_mask(color_mask)
        
        return binary_mask, os.path.basename(image_path)
    
    def predict_batch(self):
        """Perform batch prediction on all images in input directory"""
        logging.info('Starting batch prediction...')
        
        # Setup model
        model = self.setup_prediction()
        
        # Get list of input images
        input_images = sorted(glob(os.path.join(self.args.root_path, '*.png')))
        
        if not input_images:
            raise FileNotFoundError(f"No PNG images found in {self.args.root_path}")
        
        logging.info(f'Found {len(input_images)} images to process')
        
        # Process each image
        for image_path in input_images:
            try:
                # Predict single image
                binary_mask, basename = self.predict_single_image(model, image_path)
                
                # Save prediction
                output_path = os.path.join(self.args.output_dir, f'{basename.split(".png")[0]}_pred.png')
                cv2.imwrite(output_path, binary_mask)
                
                logging.info(f'Prediction saved: {output_path}')
                
            except Exception as e:
                logging.error(f'Error processing image {image_path}: {str(e)}')
                continue
        
        logging.info('Batch prediction completed!')


def main():
    """Main function to run the prediction pipeline"""
    # Setup configuration
    config_manager = PredictionConfig()
    args, config = config_manager.get_config()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')
    
    # Initialize and run predictor
    predictor = Predictor(args, config, device)
    predictor.predict_batch()


if __name__ == '__main__':
    main()