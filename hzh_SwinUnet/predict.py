import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import hzh_swin_unet_window7_448_448 as create_model
import numpy as np


def predict_img(img_path):
    # 如果没有GPU服务就CPU服务
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图片的尺寸，这里是448*448
    img_size = 448

    # 如果输入路径的图片不存在，则报错
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    # 加载一张图片
    img = torch.from_numpy(np.array(Image.open(img_path))).unsqueeze(0)
    # 显示图片
    # plt.imshow(img)
    # [N, C, H, W]
    # 从[C, H, W]变成[1, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    # 创建模型
    model = create_model().to(device)
    # 将训练好的权重加载入模型（注意，这里的model-9指的是训练了10个epoch的模型）
    model_weight_path = "./train1/weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # 不启用 BatchNormalization 和 Dropout
    model.eval()
    # torch.no_grad()中的数据不需要计算梯度,也不会进行反向传播
    with torch.no_grad():
        # 将原图片输入，得到预测后的图片
        output = torch.squeeze(model(img.to(device))).cpu()

    return output


if __name__ == '__main__':
    # 要预测的图片所在的路径
    img_path = "./hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Rec photos/val_images_x"
    # 预测完成后的图片所在路径
    store_path = "./predict photos"

    # 批量预测img_path中的所有图片，并将预测结果存放在store_path中
    list_of_files = os.listdir(img_path)
    for file_name in list_of_files:
        file_path = img_path + '/' + file_name
        img_predict = predict_img(file_path)
        pic = np.array(img_predict)
        # print(pic.shape)
        # plt.imshow(pic)
        plt.imsave(store_path + '/' + file_name, pic, cmap='gray')


