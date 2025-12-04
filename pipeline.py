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
from PIL import Image
import vtk
from vtkmodules.vtkInteractionWidgets import vtkButtonWidget

# 导入网络模块（需要确保网络模块可用）
try:
    from networks.vision_transformer import SwinUnet as ViT_seg
    from config import get_config
except ImportError:
    # 如果无法导入，提供备用方案
    logging.warning("无法导入网络模块，确保网络模块可用")

class MouldCTSegPipeline:
    """端到端的模具CT分割和三维重建管道"""
    
    def __init__(self):
        self.setup_logging()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f'使用设备: {self.device}')
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
    
    def setup_arguments(self):
        """设置命令行参数"""
        parser = argparse.ArgumentParser(description='模具CT分割和三维重建端到端管道')
        
        # 预测参数
        parser.add_argument('--root_path', type=str,
                          default=r'./datasets/H351-1-0001', 
                          help='输入数据的根目录')
        parser.add_argument('--output_dir', type=str,
                          default=r'./datasets/H351-1-0001_pred', 
                          help='预测结果输出目录')
        parser.add_argument('--batch_size', type=int,
                          default=1, help='每个GPU的批次大小')
        parser.add_argument('--num_classes', type=int,
                          default=3, help='网络输出通道数')
        parser.add_argument('--img_size', type=int,
                          default=224, help='网络输入图像大小')
        parser.add_argument('--model_path', type=str,
                          default=r'./checkpoint/MouldCTSegNet_best.pth',
                          help='训练模型检查点路径')
        parser.add_argument('--cfg', type=str, 
                          default='./configs/MouldCTSegNet_predict.yaml', 
                          metavar="FILE", help='配置文件路径')
        
        # 提取参数
        parser.add_argument('--extract_output_folder', type=str,
                          default="./datasets/H351-1-0001_output",
                          help='提取结果输出文件夹')
        
        # 三维重建参数
        parser.add_argument('--start_index', type=int, default=0,
                          help='处理的起始图像索引')
        parser.add_argument('--end_index', type=int, default=1000,
                          help='处理的结束图像索引')
        parser.add_argument('--skip_3d_reconstruction', action='store_true',
                          help='跳过三维重建步骤')
        
        return parser.parse_args()
    
    class PredictionConfig:
        """预测配置类"""
        
        def __init__(self, args):
            self.args = args
        
        def get_config(self):
            """获取配置"""
            try:
                # `get_config` expects several attributes (cfg, opts, batch_size, cache_mode, resume, amp_opt_level).
                # Ensure they exist on the args namespace (provide sensible defaults if missing).
                cfg_args = argparse.Namespace(**vars(self.args))
                for attr, default in (
                    ('cfg', getattr(self.args, 'cfg', './configs/MouldCTSegNet_predict.yaml')),
                    ('opts', getattr(self.args, 'opts', None)),
                    ('batch_size', getattr(self.args, 'batch_size', None)),
                    ('cache_mode', getattr(self.args, 'cache_mode', 'part')),
                    ('resume', getattr(self.args, 'resume', None)),
                    ('amp_opt_level', getattr(self.args, 'amp_opt_level', 'O1')),
                ):
                    if not hasattr(cfg_args, attr):
                        setattr(cfg_args, attr, default)

                config = get_config(cfg_args)
                return self.args, config
            except:
                # 如果get_config不可用，返回基本配置
                return self.args, None

    class ModelManager:
        """模型管理类"""
        
        def __init__(self, config, args, device):
            self.config = config
            self.args = args
            self.device = device
            self.model = None
        
        def load_model(self):
            """加载训练模型"""
            # 初始化模型
            self.model = ViT_seg(self.config, img_size=self.args.img_size, 
                               num_classes=self.args.num_classes)
            self.model.to(self.device)
            
            # 加载模型权重
            if not os.path.isfile(self.args.model_path):
                raise FileNotFoundError(f"模型文件未找到: {self.args.model_path}")
            checkpoint = torch.load(self.args.model_path, map_location=self.device)

            # Handle checkpoints that wrap state_dict or have DataParallel prefixes
            if isinstance(checkpoint, dict):
                # common keys: 'state_dict', 'model', or raw mapping
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (from DataParallel)
            new_state = {}
            try:
                for k, v in state_dict.items():
                    new_key = k
                    if k.startswith('module.'):
                        new_key = k[len('module.'):]
                    new_state[new_key] = v
                self.model.load_state_dict(new_state)
            except Exception:
                # Fallback: try to load directly (may raise a helpful error)
                self.model.load_state_dict(state_dict)
            
            logging.info('模型加载成功')
            return self.model

    class ImageProcessor:
        """图像处理器"""
        
        def __init__(self, img_size=224):
            self.img_size = img_size
            self.color_key = {
                'bright': [0, 128, 0],    # 绿色
                'dark': [128, 0, 0],      # 红色  
                'background': [0, 0, 0]    # 黑色
            }
            self.mean = 0.14071481
            self.std = 0.19077588
        
        def preprocess_image(self, image_path):
            """预处理输入图像"""
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            old_height, old_width = img.shape[:2]
            
            # 调整大小并保持宽高比
            scale = self.img_size * 1.0 / max(old_height, old_width)
            new_height, new_width = old_height * scale, old_width * scale
            new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
            target_size = (new_height, new_width)
            
            # 转换为张量并调整大小
            img_tensor = torch.from_numpy(img)
            img_array = resize(img_tensor.unsqueeze(0).permute((0, 3, 1, 2)), target_size)
            img_array = img_array.squeeze(0)
            
            # 填充到目标大小
            height, width = img_array.shape[-2:]
            pad_height = self.img_size - height
            pad_width = self.img_size - width
            img_array = F.pad(img_array, (0, pad_width, 0, pad_height))
            
            # 归一化图像
            img_array = np.asarray(img_array)
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
            img_array = (img_array - self.mean) / self.std
            
            # 转换为张量
            img_tensor = torch.from_numpy(img_array.astype(np.float32))
            img_tensor = img_tensor.unsqueeze(0)
            
            return img_tensor, (old_height, old_width), target_size
        
        def postprocess_prediction(self, prediction, original_size, target_size):
            """后处理预测结果"""
            output = nn.Softmax2d()(prediction)
            output = output.argmax(dim=1)
            output = output.data.cpu().numpy()
            output = np.squeeze(output)
            
            # 创建彩色掩码
            mask = np.zeros((output.shape[0], output.shape[1], 3))
            mask[output == 0] = self.color_key['background']
            mask[output == 1] = self.color_key['bright']
            mask[output == 2] = self.color_key['dark']
            mask = mask.astype(np.uint8)
            
            # 调整掩码大小到原始尺寸
            mask_tensor = torch.Tensor(mask)
            mask_resized = F.interpolate(
                mask_tensor.permute((2, 0, 1)).unsqueeze(0),
                self.img_size,
                mode="bilinear",
                align_corners=False
            )
            
            # 裁剪并调整到原始尺寸
            mask_resized = mask_resized[..., :target_size[0], :target_size[1]]
            mask_resized = F.interpolate(mask_resized, original_size, 
                                       mode="bilinear", align_corners=False)
            mask_final = np.asarray(mask_resized.squeeze(0).permute((1, 2, 0)))
            
            return mask_final
        
        def create_binary_mask(self, color_mask):
            """创建二值掩码"""
            blue = color_mask[:, :, 0]
            green = color_mask[:, :, 1]
            red = color_mask[:, :, 2]
            
            binary_mask = np.zeros((red.shape[0], red.shape[1]), dtype=np.uint8)
            
            # 识别亮区域（绿色）
            bright_mask = (red == 0) & (green >= 64) & (blue < 64)
            # 识别暗区域（红色）  
            dark_mask = (red == 0) & (green < 64) & (blue >= 64)
            
            binary_mask[bright_mask] = 85   # 亮类别
            binary_mask[dark_mask] = 170    # 暗类别
            
            return binary_mask

    class Predictor:
        """预测器类"""
        
        def __init__(self, args, config, device):
            self.args = args
            self.config = config
            self.device = device
            # Reference the nested classes via the outer class name
            self.model_manager = MouldCTSegPipeline.ModelManager(config, args, device)
            self.image_processor = MouldCTSegPipeline.ImageProcessor(img_size=args.img_size)
            
            # 创建输出目录
            os.makedirs(self.args.output_dir, exist_ok=True)
        
        def setup_prediction(self):
            """设置预测"""
            logging.info('开始预测设置...')
            model = self.model_manager.load_model()
            model.eval()
            return model
        
        def predict_single_image(self, model, image_path):
            """预测单张图像"""
            img_tensor, original_size, target_size = self.image_processor.preprocess_image(image_path)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                prediction = model(img_tensor)
            
            color_mask = self.image_processor.postprocess_prediction(prediction, original_size, target_size)
            binary_mask = self.image_processor.create_binary_mask(color_mask)
            
            return binary_mask, os.path.basename(image_path)
        
        def predict_batch(self):
            """批量预测"""
            logging.info('开始批量预测...')
            model = self.setup_prediction()
            input_images = sorted(glob(os.path.join(self.args.root_path, '*.png')))
            
            if not input_images:
                raise FileNotFoundError(f"在 {self.args.root_path} 中未找到PNG图像")
            
            logging.info(f'找到 {len(input_images)} 张图像需要处理')
            
            for image_path in input_images:
                try:
                    binary_mask, basename = self.predict_single_image(model, image_path)
                    output_path = os.path.join(self.args.output_dir, f'{basename.split(".png")[0]}_pred.png')
                    cv2.imwrite(output_path, binary_mask)
                    logging.info(f'预测保存: {output_path}')
                except Exception as e:
                    logging.error(f'处理图像 {image_path} 时出错: {str(e)}')
                    continue
            
            logging.info('批量预测完成!')
            return self.args.output_dir

    def extract_categories(self, args):
        """提取特定类别"""
        logging.info('开始提取亮部和暗部区域...')
        
        img_folder = args.root_path
        mask_folder = args.output_dir
        output_folder = args.extract_output_folder
        
        bright_part_save_folder = os.path.join(output_folder, "bright_part")
        dark_part_save_folder = os.path.join(output_folder, "dark_part")
        os.makedirs(bright_part_save_folder, exist_ok=True)
        os.makedirs(dark_part_save_folder, exist_ok=True)
        
        files_list = os.listdir(img_folder)
        
        # 类别灰度值
        bright_part = 85
        dark_part = 170
        
        for img_files in files_list:
            if img_files.endswith(".png"):
                img_path = os.path.join(img_folder, img_files)
                mask_files = img_files.split('.png')[0] + '_pred.png'
                mask_path = os.path.join(mask_folder, mask_files)
                
                if not os.path.exists(mask_path):
                    logging.warning(f"掩码文件不存在: {mask_path}")
                    continue
                
                # 打开图像和掩码
                img_image = Image.open(img_path)
                mask_image = Image.open(mask_path)
                
                # 转换为numpy数组
                png_array = np.array(img_image)
                mask_array = np.array(mask_image)
                
                # 找到目标区域
                bright_target_mask = (mask_array == bright_part)
                dark_target_mask = (mask_array == dark_part)
                
                # 提取目标区域
                bright_target_png_array = np.zeros_like(png_array)
                bright_target_png_array[bright_target_mask] = png_array[bright_target_mask]
                dark_target_png_array = np.zeros_like(png_array)
                dark_target_png_array[dark_target_mask] = png_array[dark_target_mask]
                
                # 转换为PIL图像并保存
                bright_target_bmp_image = Image.fromarray(bright_target_png_array)
                dark_target_bmp_image = Image.fromarray(dark_target_png_array)
                
                bright_target_output_path = os.path.join(bright_part_save_folder, img_files)
                dark_part_output_path = os.path.join(dark_part_save_folder, img_files)
                
                bright_target_bmp_image.save(bright_target_output_path)
                dark_target_bmp_image.save(dark_part_output_path)
        
        logging.info('类别提取完成!')
        return output_folder

    def read_images(self, files, format):
        """读取图像序列"""
        if format.lower() == 'png':
            reader = vtk.vtkPNGReader()
        else:
            reader = vtk.vtkBMPReader()
            
        image3D = vtk.vtkImageAppend()
        image3D.SetAppendAxis(2)

        for f in files:
            reader.SetFileName(f)
            reader.Update()
            t_img = vtk.vtkImageData()
            t_img.DeepCopy(reader.GetOutput())
            image3D.AddInputData(t_img)

        image3D.Update()
        return image3D.GetOutput()

    def load_files(self, args):
        """加载文件"""
        start_index = args.start_index
        end_index = args.end_index
        output_folder = args.extract_output_folder
        
        bright_png_folder = os.path.join(output_folder, 'bright_part')
        dark_png_folder = os.path.join(output_folder, 'dark_part')

        bright_png_files = glob(bright_png_folder + os.sep + '*.png')[start_index:end_index]
        bright_png_files.sort()
        dark_png_files = glob(dark_png_folder + os.sep + '*.png')[start_index:end_index]
        dark_png_files.sort()

        dark_data_list = self.read_images(dark_png_files, 'png')
        dark_data_list.SetSpacing([1, 1, 1])

        bright_data_list = self.read_images(bright_png_files, 'png')
        bright_data_list.SetSpacing([1, 1, 1])

        # 加载原始图像和掩码
        ori_png_folder = args.root_path
        ori_png_files = glob(ori_png_folder + os.sep + '*.png')[start_index:end_index]
        ori_png_files.sort()
        mask_png_folder = args.output_dir
        mask_png_files = glob(mask_png_folder + os.sep + '*.png')[start_index:end_index]
        mask_png_files.sort()

        ori_data_list = self.read_images(ori_png_files, 'png')
        ori_data_list.SetSpacing([1, 1, 1])

        mask_data_list = self.read_images(mask_png_files, 'png')
        mask_data_list.SetSpacing([1, 1, 1])

        return bright_data_list, dark_data_list, ori_data_list, mask_data_list

    def vtk_init(self):
        """VTK初始化"""
        renderer_left = vtk.vtkRenderer()
        renderer_left.SetBackground(0.8, 0.8, 0.8)
        renderer_left.SetViewport(0.0, 0.0, 0.5, 1.0)

        renderer_right = vtk.vtkRenderer()
        renderer_right.SetBackground(0.6, 0.6, 0.6)
        renderer_right.SetViewport(0.5, 0.0, 1.0, 1.0)

        # 创建渲染窗口
        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(1200, 600)
        render_window.SetWindowName('3D可视化')
        render_window.AddRenderer(renderer_left)
        render_window.AddRenderer(renderer_right)

        # 创建窗口交互器
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        render_window_interactor.SetInteractorStyle(style)

        colors = vtk.vtkNamedColors()

        return renderer_left, renderer_right, render_window, render_window_interactor, colors

    def create_button(self, render_window_interactor):
        """创建按钮"""
        pngReader = vtk.vtkPNGReader()
        pngReader.SetFileName("./utils/Button.png")
        pngReader.Update()
        image = pngReader.GetOutput()

        sz = 150.0
        bds = [0, 0, 0, 0, 0, 0]
        bds[0] = 4096
        bds[1] = bds[0] + sz / 2
        bds[2] = 0
        bds[3] = bds[2] + sz / 2

        buttonRepresentation = vtk.vtkTexturedButtonRepresentation2D()
        buttonRepresentation.SetNumberOfStates(1)
        buttonRepresentation.SetButtonTexture(0, image)
        buttonRepresentation.PlaceWidget(bds)

        mbutton = vtkButtonWidget()
        mbutton.SetInteractor(render_window_interactor)
        mbutton.SetRepresentation(buttonRepresentation)
        
        return mbutton

    def button_callback_factory(self, renderer_left, ori_volume, bright_volume, dark_volume):
        """创建按钮回调函数工厂"""
        mbutton_state = 0
        
        def my_button_callback(obj, event):
            nonlocal mbutton_state
            mbutton_state += 1
            if mbutton_state == 3:
                mbutton_state = 0
            
            if mbutton_state == 0:
                print('显示全部')
                renderer_left.RemoveVolume(dark_volume)
                renderer_left.RemoveVolume(bright_volume)
                renderer_left.AddVolume(ori_volume)
            elif mbutton_state == 1:
                print('显示亮部')
                renderer_left.RemoveVolume(ori_volume)
                renderer_left.RemoveVolume(dark_volume)
                renderer_left.AddVolume(bright_volume)
            else:
                print('显示暗部')
                renderer_left.RemoveVolume(ori_volume)
                renderer_left.RemoveVolume(bright_volume)
                renderer_left.AddVolume(dark_volume)
        
        return my_button_callback

    def show_bright_dark_parts(self, bright_data_output, dark_data_output, ori_data_output):
        """显示亮部和暗部"""
        # 亮部体积数据滤波
        bright_volume_data_filter = vtk.vtkImageGaussianSmooth()
        bright_volume_data_filter.SetInputData(bright_data_output)
        bright_volume_data_filter.SetStandardDeviation(4.0)
        bright_volume_data_filter.Update()

        # 暗部体积数据滤波
        dark_volume_data_filter = vtk.vtkImageGaussianSmooth()
        dark_volume_data_filter.SetInputData(dark_data_output)
        dark_volume_data_filter.SetStandardDeviation(4.0)
        dark_volume_data_filter.Update()

        # 创建映射器
        bright_volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        bright_volume_mapper.SetInputConnection(bright_volume_data_filter.GetOutputPort())

        dark_volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        dark_volume_mapper.SetInputConnection(dark_volume_data_filter.GetOutputPort())

        ori_volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        ori_volume_mapper.SetInputData(ori_data_output)

        # 设置颜色和不透明度传递函数
        bd_color_func = vtk.vtkColorTransferFunction()
        bd_color_func.AddRGBPoint(40, 0.0, 0.0, 0.0)
        bd_color_func.AddRGBPoint(200, 1.0, 1.0, 1.0)
        bd_opacity_func = vtk.vtkPiecewiseFunction()
        bd_opacity_func.AddPoint(40, 0.0)
        bd_opacity_func.AddPoint(255, 1.0)

        # 设置体积属性
        bright_volume_property = vtk.vtkVolumeProperty()
        bright_volume_property.SetColor(bd_color_func)
        bright_volume_property.SetScalarOpacity(bd_opacity_func)
        bright_volume_property.SetInterpolationTypeToLinear()

        dark_volume_property = vtk.vtkVolumeProperty()
        dark_volume_property.SetColor(bd_color_func)
        dark_volume_property.SetScalarOpacity(bd_opacity_func)
        dark_volume_property.SetInterpolationTypeToLinear()

        # 原始图像颜色设置
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(20, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(255, 1.0, 1.0, 1.0)
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(20, 0.0)
        opacity_func.AddPoint(255, 1.0)

        ori_volume_property = vtk.vtkVolumeProperty()
        ori_volume_property.SetColor(color_func)
        ori_volume_property.SetScalarOpacity(opacity_func)
        ori_volume_property.SetInterpolationTypeToLinear()

        # 创建体积对象
        bright_volume = vtk.vtkVolume()
        bright_volume.SetMapper(bright_volume_mapper)
        bright_volume.SetProperty(bright_volume_property)

        dark_volume = vtk.vtkVolume()
        dark_volume.SetMapper(dark_volume_mapper)
        dark_volume.SetProperty(dark_volume_property)

        ori_volume = vtk.vtkVolume()
        ori_volume.SetMapper(ori_volume_mapper)
        ori_volume.SetProperty(ori_volume_property)

        return bright_volume, dark_volume, ori_volume

    def create_tf_lut(self, colorful_flag):
        """创建查找表"""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(300)
        lut.SetTableRange(0, 299)
        lut.Build()

        if colorful_flag == 0:
            for i in range(1):
                lut.SetTableValue(i, [0.0, 0.0, 0.0, 0.0])
            for i in range(1, 299):
                lut.SetTableValue(i, [i/(299-1), i/(299-1), i/(299-1), 0.8])
        else:
            for i in range(40):
                lut.SetTableValue(i, [0.0, 0.0, 0.0, 0.0])
            for i in range(40, 130):
                lut.SetTableValue(i, [0.0, 0.5, 0.0, 0.5])
            for i in range(130, 299):
                lut.SetTableValue(i, [0.5, 0.0, 0.0, 0.5])

        return lut

    def three_d_reconstruction(self, args):
        """三维重建"""
        logging.info('开始三维重建...')
        
        # VTK初始化
        renderer_left, renderer_right, render_window, render_window_interactor, colors = self.vtk_init()

        # 加载文件
        bright_data_output, dark_data_output, ori_data_output, mask_data_output = self.load_files(args)

        # 显示亮部和暗部
        bright_volume, dark_volume, ori_volume = self.show_bright_dark_parts(
            bright_data_output, dark_data_output, ori_data_output
        )
        renderer_left.AddVolume(ori_volume)

        # 创建图像平面小部件
        ori_widget = vtk.vtkImagePlaneWidget()
        ori_widget.TextureVisibilityOff()
        ori_widget.SetInputData(ori_data_output)
        ori_widget.SetPlaneOrientationToZAxes()
        ori_widget.SetInteractor(render_window_interactor)
        ori_widget.SetSliceIndex(1)

        # 重切片掩码图像以匹配原始图像
        mask_image_reslice = vtk.vtkImageReslice()
        mask_image_reslice.SetInputData(mask_data_output)
        mask_image_reslice.SetOutputDimensionality(2)
        mask_image_reslice.SetResliceAxes(ori_widget.GetResliceAxes())
        mask_image_reslice.SetOutputScalarType(vtk.VTK_UNSIGNED_CHAR)
        mask_image_reslice.Update()

        ori_image_reslice = vtk.vtkImageReslice()
        ori_image_reslice.SetInputData(ori_data_output)
        ori_image_reslice.SetOutputDimensionality(2)
        ori_image_reslice.SetResliceAxes(ori_widget.GetResliceAxes())
        ori_image_reslice.SetOutputScalarType(vtk.VTK_UNSIGNED_CHAR)
        ori_image_reslice.Update()

        # 在右侧视口显示原始图像
        ori_plane = vtk.vtkPlaneSource()
        ori_plane_mapper = vtk.vtkPolyDataMapper()
        ori_plane_mapper.SetInputConnection(ori_plane.GetOutputPort())

        ori_lut = self.create_tf_lut(colorful_flag=0)
        ori_plane_texture = vtk.vtkTexture()
        ori_plane_texture.SetInputConnection(ori_image_reslice.GetOutputPort())
        ori_plane_texture.SetLookupTable(ori_lut)
        ori_plane_texture.SetColorModeToMapScalars()

        ori_plane_actor = vtk.vtkActor()
        ori_plane_actor.SetMapper(ori_plane_mapper)
        ori_plane_actor.SetTexture(ori_plane_texture)
        renderer_right.AddActor(ori_plane_actor)

        # 在右侧视口显示掩码图像
        mask_image_plane = vtk.vtkPlaneSource()
        mask_image_mapper = vtk.vtkPolyDataMapper()
        mask_image_mapper.SetInputConnection(mask_image_plane.GetOutputPort())

        mask_lut = self.create_tf_lut(colorful_flag=1)
        mask_color_map = vtk.vtkImageMapToColors()
        mask_color_map.SetInputConnection(mask_image_reslice.GetOutputPort())
        mask_color_map.SetLookupTable(mask_lut)
        mask_color_map.Update()
        mask_texture = vtk.vtkTexture()
        mask_texture.SetInputConnection(mask_color_map.GetOutputPort())

        mask_image_actor = vtk.vtkActor()
        mask_image_actor.SetMapper(mask_image_mapper)
        mask_image_actor.SetTexture(mask_texture)
        mask_image_actor.GetProperty().SetOpacity(0.5)
        renderer_right.AddActor(mask_image_actor)

        # 添加按钮
        mbutton = self.create_button(render_window_interactor)
        button_callback = self.button_callback_factory(renderer_left, ori_volume, bright_volume, dark_volume)
        mbutton.AddObserver("StateChangedEvent", button_callback)
        mbutton.SetEnabled(1)

        # 显示轮廓
        outlineData = vtk.vtkOutlineFilter()
        outlineData.SetInputData(ori_data_output)
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outlineData.GetOutputPort())

        outline = vtk.vtkActor()
        outline.SetMapper(outline_mapper)
        outline.GetProperty().SetColor(0, 0, 0)
        renderer_left.AddActor(outline)

        # 启动交互
        ori_widget.On()
        render_window.Render()
        render_window_interactor.Initialize()
        
        logging.info('三维重建完成，启动可视化窗口...')
        render_window_interactor.Start()

    def run_pipeline(self):
        """运行完整管道"""
        args = self.setup_arguments()
        
        try:
            # 步骤1: 预测
            logging.info('=== 步骤1: 图像分割预测 ===')
            config_obj = self.PredictionConfig(args)
            pred_args, config = config_obj.get_config()
            
            predictor = self.Predictor(pred_args, config, self.device)
            mask_output_dir = predictor.predict_batch()
            
            # 步骤2: 类别提取
            logging.info('=== 步骤2: 类别提取 ===')
            extract_output_dir = self.extract_categories(args)
            
            # 步骤3: 三维重建（可选）
            if not args.skip_3d_reconstruction:
                logging.info('=== 步骤3: 三维重建 ===')
                self.three_d_reconstruction(args)
            else:
                logging.info('跳过三维重建步骤')
                
            logging.info('管道执行完成!')
            
        except Exception as e:
            logging.error(f'管道执行失败: {str(e)}')
            raise

def main():
    """主函数"""
    pipeline = MouldCTSegPipeline()
    pipeline.run_pipeline()

if __name__ == '__main__':
    main()