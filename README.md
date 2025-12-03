# Semantic segmentation and 3D reconstruction of CT images of bronze casting moulds and cores based on a deep learning method

This repository contains deep learning code for semantic segmentation of bronze casting moulds and cores, based on Swin U-Net.
For detailed information, please refer to the paper "Semantic segmentation and 3D reconstruction of CT images of bronze casting moulds and cores based on a deep learning method".

---
## Dataset

The datasets of bronze casting moulds and cores used in this study came from the Taijiasi site, a high-ranking settlement in the Huaihe River basin during the early to middle Shang Dynasty (14th-13th century BCE).
By conducting CT scanning on moulds and cores, we can obtain a series of continuous 2D CT images.
However, some of them may suffer from poor quality due to underexposure or blurriness.
These images could make a poor influence on network training and need to be filtered out before annotation.
Finally, 12 samples with distinct features were selected to construct the dataset.
Under the guidance of archaeological experts, the original CT images in the bronze casting mould and core datasets were annotated using the polygon annotation tool in LabelMe software.

---
## Environment

### Using Conda to Build Environment (Recommended)
We strongly recommend using Conda to build the environment as it provides better support for GUI applications like VTK-based 3D visualization.

1. Create a virtual environment required by the project:
```
conda create -n MouldCTSegNet python=3.9.24
```
2. Activate the virtual environment:
```
conda activate MouldCTSegNet
```
3. Install the required packages:
```
pip install -r requirements.txt
```
Then, activate the virtual environment and continue to train or test.

### Using Docker to Build Environment

**Note**: Docker environment is mainly suitable for training and prediction tasks. For 3D reconstruction with VTK visualization, we recommend using the Conda environment due to better GUI support.
1. Build the Docker image:
```
docker build -t mouldctsegnet:latest .
```
2. Run the Docker container:
```
docker run --gpus all -it --shm-size=16g -v ${PWD}:/workspace -p 6006:6006 --name mouldctsegnet-container mouldctsegnet
```

**Important Limitations​​**:
- The Docker environment can handle training (trainer.py) and prediction (predict.py) tasks
- For 3D reconstruction using 3D_Reconstruction.py, you may need to install VTK on the host machine and run the visualization outside the container
- GUI applications within Docker require complex X11 forwarding configuration

---
## Pipeline Overview
The complete workflow consists of four main steps:
1. **Training**: Use trainer.pyto train the segmentation model
2. **Prediction**: Use predict.pyto generate segmentation masks
3. **Region Extraction**: Use ExtractCertainCategory.pyto extract specific regions
4. **3D Reconstruction**: Use 3D_Reconstruction.pyfor 3D visualization and reconstruction

### Dataset Preparation
To train the MouldCTSegNet model, we need to prepare the dataset of bronze casting moulds and cores.

The dataset should be arranged as follows:
```
datasets
├── train
│   ├── Image
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └──...
│   └── Mask
│       ├── image1_label.png
│       ├── image2_label.png
│       └──...
├── val
│   ├── Image
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └──...
│   └── Mask
│       ├── image1_label.png
│       ├── image2_label.png
│       └──...
└── test
    ├── Image
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── Mask
        ├── image1_label.png
        ├── image2_label.png
        └── ...
```


You can download the file '[datasets.zip](https://pan.baidu.com/s/1lWqWhH9h44lJUblZMQICag?pwd=8nyh)' and unzip it to the folder that contains 'train.py' and 'predict.py'.

You can download the .pth file '[MouldCTSegNet_best.pth](https://pan.baidu.com/s/1NgrRAHdkiWLf7K_mczP2EA?pwd=vhe4)'.

The pretrain .pth file '[swin_T.pth](https://pan.baidu.com/s/1PYEDpnpMTgAmlIMMwEn6BA?pwd=6ige)'.

### Argument Configuration
If you want to change the location of the dataset, you can open the './configs/MouldCTSegNet_train.yaml' file and change the corresponding parameters.

### Step 1: Training
After the configuration, start training using the following command:
```
python trainer.py --batch_size=24 --max_epochs=350 --n_gpu=1 --base_lr=0.01 --num_classes=3
```

`--batch_size` argument is the number of samples in each batch. Default is 24.

`--max_epochs` argument is the maximum number of epochs to train. Default is 350.

`--n_gpu` argument is the number of GPUs to use. Default is 1.

`--base_lr` argument is the initial learning rate. Default is 0.01.

`--num_classes` argument is the number of classes in the dataset. Default is 3.

When the training is complete, the trained model weight files will be saved directly in the `./checkpoint` folder (or the directory specified by --output_dir parameter). The folder will contain:
- `MouldCTSegNet_best.pth`: the best model weights based on validation loss
- `MouldCTSegNet_Last_epoch.pth`: the latest checkpoint containing model and optimizer states

In the same directory, you can find a `log` folder which contains the training logs for TensorBoard. You can use TensorBoard to monitor the training process by running:
```
tensorboard --logdir=./checkpoint/log
```

---
### Step 2: Prediction
After training, evaluation can be performed using the following command:
```
python predict.py --root_path=./datasets/test/Image --output_dir=./datasets/pred_output
```
`--root_path` is the path of the input CT images, which should be in the format of **.png** files.

`--output_dir` is the path of the output binary masks, which will be saved in the `./datasets/pred_output` folder.

In the output folder, the predicted binary masks will be saved in the format of **.png** files.

---
### Step 3: Region Extraction
Different from the previous step, we need to prepare all CT images of the samples in `.png` format in a folder such as `./datasets/H351-1-0001`.

Using the model we trained in the previous step, we can predict the binary masks of moulding CT images.
```
python predict.py --root_path=./datasets/H351-1-0001 --output_dir=./datasets/H351-1-0001_pred
```

To extract bright and dark regions from the segmented images:
```
python ExtractCertainCategory.py --sample_folder=.//datasets//H351-1-0001 \
                                 --mask_folder=.//datasets//H351-1-0001_pred \
                                 --output_folder=.//datasets//H351-1-0001_output \
```

`--sample_folder` is the path of the input CT images, which should be in the format of **.png** files.

`--mask_folder` is the path of the predicted binary masks, which should be in the format of **.png** files.

`--output_folder` is the path of the output folder, which will be saved in the `./datasets/H351-1-0001_output` folder.

### Step 4: 3D Reconstruction
To reconstruct 3D models of bronze casting moulds and cores:
```
python 3D_Reconstruction.py --sample_folder=.//datasets//H351-1-0001 --mask_folder=.//datasets//H351-1-0001_pred --output_folder=.//datasets//H351-1-0001_output --start_index=0 --end_index=1000
```
`--sample_folder` is the path of the input CT images, which should be in the format of **.png** files.

`--mask_folder` is the path of the predicted binary masks, which should be in the format of **.png** files.

`--output_folder` is the path of the output folder, which will be saved in the `./datasets/H351-1-0001_output` folder.

`--start_index` is the starting index of the CT images to be reconstructed. Default is 0.

`--end_index` is the ending index of the CT images to be reconstructed. Default is 1000.

After running the command, a window will pop up, showing the 3D reconstruction model of the first 1000 CT images.
In this window, the left viewport is the reconstructed 3D model with virtual plane, 
and the right viewport is the fusion display of the original image and the color segmentation results at the virtual plane position.

In addition, you can use the mouse to rotate, zoom in and out, and click the **CHANGE** button to show different parts of 3D reconstruction model.
Pressing the mouse wheel and dragging the mouse will change the position of the position of the virtual slice.

**Important Note for Docker Users**: The 3D reconstruction step requires GUI support. If you're using Docker, you may need to run this step on the host machine with proper VTK installation, or configure X11 forwarding for Docker.