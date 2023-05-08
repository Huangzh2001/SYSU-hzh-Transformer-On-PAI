# Transformer On PAI

## 一、项目摘要

该项目的目的是搭建一个Swin-Unet模型，将低传感器数量下得到的低质量光声成像重建图像优化为高质量图像。

采用的原始数据集是Skin Cancer MNIST: HAM10000。可以在官网[Skin Cancer MNIST: HAM10000 | Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)中下载。

## 二、项目代码目录

本项目的代码分成五部分：

* 问题的发现与课题的提出

* 构建数据集

* 搭建并训练Swin-Unet神经网络

* 评估与分析训练结果兼论文作图

* 论文latex代码

## 三、代码详解

## 0、准备工作

如果需要复现论文结果。则需要将Skin Cancer MNIST: HAM10000数据集下载，并存放在该目录下的DataSet文件夹中，并将文件夹命名为Skin Cancer MNIST。如不出意外的话，该文件夹内会有如下两个文件夹：

* HAM10000_images_part_1

* HAM10000_images_part_2
  
  它们分别存放着5000张皮肤癌图像。下面的代码只对HAM10000_images_part_1中的图像进行操作，如果想要对HAM10000_images_part_2进行同样的操作，只需修改读取路径和保存路径即可。

## 1、数据集的预处理

* PAI_Pretreatment.m
  
  功能：对数据集进行预处理，并将结果存放在DataSet文件夹中的Skin Cancer MNIST Pre photos。

### 2、问题的发现与课题的提出

首先我们在Skin Cancer MNIST Pre photos中挑选出100张图像（我选的是前100张）。设置不同的传感器数量（50、100、200、400、800个），进行光声成像的仿真与重建，各自得到100张重建图像。再计算它们与原图像之间的MSE、PSNR、SSIM值，画出变化曲线图。

* k-wave-toolbox-version-1.4
  
  存放着k-Wave库，在运行光声成像仿真与重建的matlab代码时，要load该库。

* PAI_SIMULATION_AND_PAI_reconstruction_100photos.m
  
  功能：对100张图片进行光声成像仿真与重建，存放在DataSet中的Skin Cancer MNIST 100 Rec Photos。注意，代码只写了50个传感器的情况，要记得修改传感器数量与保存文件夹路径来得到其他传感器数量下的重建图像。

* MSE_100_image.ipynb
  
  简介：计算100张重建图片在不同传感器数量下与原图片的MSE

* PSNR_100_image.ipynb
  
  简介：计算100张重建图片在不同传感器数量下与原图片的PSNR

* SSIM_100_image.ipynb
  
  简介：计算100张重建图片在不同传感器数量下与原图片的SSIM

* MSE_PSNR_SSIM_100_image.ipynb
  
  简介：将上述代码进行汇总与作图，图片存放在Image文件夹。

最终我们从画出的图中可以得出一个结论：**光声重建图像的质量跟传感器数量成正比**。

### 3、构建数据集

* k-wave-toolbox-version-1.4
  
  存放着k-Wave库，在运行光声成像仿真与重建的matlab代码时，要load该库。

* PAI_SIMULATION_AND_PAI_reconstruction.m
  
  功能：进行光声成像仿真与重建（传感器数量设置为50个），重建图像存放在DataSet中的Skin Cancer MNIST Rec photos文件夹。

* img_crop.ipynb
  
  功能：将图像裁剪为448*448。存放在DataSet中的Crop Skin Cancer MNIST文件夹。

### 4、搭建并训练Swin-Unet神经网络

* hzh_SwinT_50_sensor_Data
  
  简介：该文件夹里存放着训练用的数据集：
  
  * Crop Skin Cancer MNIST Pre photos
    
    * train_images_y：训练集中的“y”
    
    * val_images_y：测试集中的“y”
  
  * Crop Skin Cancer MNIST Rec photos
    
    * train_images_x：训练集中的“x”
    
    * val_images_x：测试集中的“x”
  
  请自行将所得数据集分类到上述文件中。

* weights
  
  简介：该文件夹用于存放每次训练后的模型

* loss
  
  简介：该文件夹用于存放每次训练的train loss和val loss，便于后续画图。

* predict photos
  
  简介：用于存放预测后的图像。

* model.py
  
  简介：Swin-Unet神经网络的实现

* my_dataset.py
  
  简介：自定义的数据集

* utils.py
  
  简介：定义了单个epoch的训练与测试代码供train.py调用

* train.py
  
  简介：训练函数

* predict.py
  
  简介：使用训练好的模型进行预测的代码。将预测图像存放在DataSet中的Skin Cancer MNIST Predict photos文件夹。

### 5、评估与分析训练结果兼论文作图

* Compare_predict_MSE_SSIM_PSNR.ipynb
  
  简介：将“模型预测图像与原图像的MSE、SSIM、PSNR”与“重建图像与原图像的MSE、SSIM、PSNR”进行比较并作图。图片存放在Image文件夹。

* plot_training_loss_img.ipynb
  
  简介：画出训练模型时不同epoch下loss的变化曲线。图片存放在Image文件夹。