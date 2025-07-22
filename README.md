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

Firstly, create a virtual Python 3.8 environment in Anaconda with the necessary dependencies from the **environment.yaml** file provided in the code.

```
conda env create -f MouldCTSegNet.yml
```
Then, activate the virtual environment and continue to train or test.

---
## Train

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

You can download the .pth file '[MouldCTSegNet.pth](https://pan.baidu.com/s/1GaRSyi38pa9oG_0qb859fg?pwd=n27p)'.

### Argument Configuration
If you want to change the location of the dataset, you can open the 'MouldCTSegNet.yml' file and change the corresponding parameters.

### Start Training
After the configuration, start training using the following command:
```
python train.py --weights=./checkpoint/MouldCTSegNet.pth  \
                --batch_size=24 \
                --max_epochs=350 \
                --n_gpu=1 \
                --base_lr=0.01
                --num_classes=3
```

The checkpoints of the trained model will be saved in the `/checkpoint` folder.

`--weights` argument is the weight file of the pre-trained model, which is used to initialize the model. Default is `./checkpoint/MouldCTSegNet.pth`.

`--batch_size` argument is the number of samples in each batch. Default is 24.

`--max_epochs` argument is the maximum number of epochs to train. Default is 350.

`--n_gpu` argument is the number of GPUs to use. Default is 1.

`--base_lr` argument is the initial learning rate. Default is 0.01.

`--num_classes` argument is the number of classes in the dataset. Default is 3.

When the training is complete, the trained model weight file will be saved in a subfolder named with your training start time, like `1_15_50`, in the `/checkpoint` folder.

In your folder which keeps trained model weight file, you can find a `logs` folder, which contains the training logs of the model.
You can use TensorBoard to monitor the training process.

---
## Test
After training, evaluation can be performed using the following command:
```
python predict.py --weights=./checkpoint/MouldCTSegNet.pth \
                  --root_path=./datasets/test \
                  --output_dir=./datasets/pred_output \
```

`--weights` is the weight file of the trained model, which is saved in the `/checkpoint` folder.

`--root_path` is the path of the input CT images, which should be in the format of **.png** files.

`--output_dir` is the path of the output binary masks, which will be saved in the `./datasets/pred_output` folder.

In the output folder, the predicted binary masks will be saved in the format of **.png** files.

---
## 3D reconstruction
To reconstruct 3D models of bronze casting moulds and cores, we use VTK library.

### Dataset Preparation
Different from the previous step, we need to prepare all CT images of the samples in `.png` format in a folder such as `./datasets/H351-1-0001`.

Using the model we trained in the previous step, we can predict the binary masks of moulding CT images.

Based on the binary masks of moulding CT images, 
we can extract the bright part, dark part, original image and binary mask of each CT image by using the following command:
```
python ExtractCertainCategory.py --sample_folder=.//datasets//H351-1-0001 \
                                  --mask_folder=.//datasets//H351-1-0001_pred \
                                  --output_folder=.//datasets//H351-1-0001_output \
```

`--sample_folder` is the path of the input CT images, which should be in the format of **.png** files.

`--mask_folder` is the path of the predicted binary masks, which should be in the format of **.png** files.

`--output_folder` is the path of the output folder, which will be saved in the `./datasets/H351-1-0001_output` folder.

### Visualize 3D reconstruction model
By preparing `bright part`, `dark part`, `original image` and `binary mask`, we can use the following command to reconstruct 3D models of moulding CT images:
```
python 3D_Reconstruction.py --sample_folder=.//datasets//H351-1-0001 \
                             --mask_folder=.//datasets//H351-1-0001_pred \
                             --output_folder=.//datasets//H351-1-0001_output \
                             --start_index=0 \
                             --end_index=1000
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
