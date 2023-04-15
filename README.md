# 2023April

## 项目简介

本项目的代码分成三部分：

* 问题的发现与课题的提出

* 构建数据集

* 搭建并训练Swin-Unet神经网络

* 评估与分析训练结果兼论文作图

## 代码详解

### 1、问题的发现与课题的提出

* MSE_100_image.ipynb
  
  简介：计算100张重建图片在不同传感器数量下与原图片的MSE

* PSNR_100_image.ipynb
  
  简介：计算100张重建图片在不同传感器数量下与原图片的PSNR

* SSIM_100_image.ipynb
  
  简介：计算100张重建图片在不同传感器数量下与原图片的SSIM

* MSE_PSNR_SSIM_100_image.ipynb
  
  简介：将上述代码进行汇总与作图

### 2、构建数据集

* k-wave-toolbox-version-1.4
  
  存放着k-Wave库，在运行构建数据集的matlab代码时，要load该库。

* PAI_Pretreatment.m
  
  功能：对数据集进行预处理

* PAI_SIMULATION_AND_PAI_reconstruction.m
  
  功能：进行光声成像仿真与重建

* img_crop.ipynb
  
  功能：将图像裁剪为448*448

### 3、搭建并训练Swin-Unet神经网络

* model.py
  
  简介：Swin-Unet神经网络的实现

* my_dataset.py
  
  简介：自定义的数据集

* utils.py
  
  简介：定义了单个epoch的训练与测试代码供train.py调用

* train.py
  
  简介：训练函数

* predict.py
  
  简介：使用训练好的模型进行预测的代码

### 4、评估与分析训练结果兼论文作图

* Compare_predict_MSE_SSIM_PSNR.ipynb
  
  简介：将“模型预测图像与原图像的MSE、SSIM、PSNR”与“重建图像与原图像的MSE、SSIM、PSNR”进行比较并作图、

* plot_training_loss_img.ipynb
  
  简介：画出训练模型时不同epoch下loss的变化曲线