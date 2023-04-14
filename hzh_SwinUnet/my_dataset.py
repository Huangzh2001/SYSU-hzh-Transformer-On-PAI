from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MyDataSet(Dataset):
    def __init__(self, root_x, root_y):
        data_list = os.listdir(root_x)
        self.x_data_path_list = [os.path.join(root_x, str(i)) for i in data_list]
        self.y_data_path_list = [os.path.join(root_y, str(i)) for i in data_list]

    def __getitem__(self, index):
        x_index_path = self.x_data_path_list[index]
        y_index_path = self.y_data_path_list[index]
        x_image = torch.from_numpy(np.array(Image.open(x_index_path)))
        y_image = torch.from_numpy(np.array(Image.open(y_index_path)))

        # 由于这里读取的是灰度图，所以要加一个维度
        x_image = x_image.unsqueeze(0)
        y_image = y_image.unsqueeze(0)

        return x_image, y_image

    def __len__(self):
        return len(self.x_data_path_list)

# 下面是测试代码
# if __name__ == '__main__':
#     train_x_path = 'D:/Document/DataSet/hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Rec photos/train_images_x'
#     train_y_path = 'D:/Document/DataSet/hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Pre photos/train_images_y'
#     data = MyDataSet(train_x_path, train_y_path)
#     print(data[100])
#     print(len(data))



# import os
# import argparse
#
# import torch
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
#
# from my_dataset import MyDataSet
# from model import hzh_swin_unet_window7_448_448 as create_model
# from utils import train_one_epoch, evaluate
# def main(args):
#     # 如果没有GPU服务就CPU服务
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#
#     # 新建一个文件来存放weight
#     if os.path.exists("./weights") is False:
#         os.makedirs("./weights")
#
#     # TensorBoard是一个独立的包（不是pytorch中的），这个包的作用就是可视化您模型中的各种参数和结果。
#     tb_writer = SummaryWriter()
#
#     # 训练集、测试集路径
#     train_x_path = 'D:/Document/DataSet/hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Rec photos/train_images_x'
#     train_y_path = 'D:/Document/DataSet/hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Pre photos/train_images_y'
#     val_x_path = 'D:/Document/DataSet/hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Rec photos/val_images_x'
#     val_y_path = 'D:/Document/DataSet/hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Pre photos/val_images_y'
#
#     # 实例化训练数据集
#     # 分别传入训练集路径、训练集标签、对训练集进行预处理的transform
#     train_dataset = MyDataSet(root_x=train_x_path,
#                               root_y=train_y_path,)
#     # 实例化验证数据集
#     val_dataset = MyDataSet(root_x=val_x_path,
#                             root_y=val_y_path,)
#
#     # 获取当前系统能允许的最大载入线程数
#     batch_size = args.batch_size
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#     print('Using {} dataloader workers every process'.format(nw))
#
#     # 加载训练集
#     # batch_size:每一批的图片数量
#     # shuffle：数据集是否要打乱
#     # num_workers：载入数据的线程数
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                pin_memory=True,
#                                                num_workers=nw)
#     # 加载测试集
#     val_loader = torch.utils.data.DataLoader(val_dataset,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              pin_memory=True,
#                                              num_workers=nw)
#
#     for step, data in enumerate(train_loader):
#         x_image, y_image = data
#         print(x_image.shape)
#
#
# if __name__ == '__main__':
#     # argparse我把它理解为一个列表，比如你写parser.add_argument('--num_classes', type=int, default=5)
#     # 就相当于再这个列表里加了一个参数num_classes，你可以通过parser.num_classes得到5
#     parser = argparse.ArgumentParser()
#     # 训练次数设置为10次
#     parser.add_argument('--epochs', type=int, default=10)
#     # batch size设置为16
#     parser.add_argument('--batch-size', type=int, default=16)
#     # 学习率设置为0.0001
#     parser.add_argument('--lr', type=float, default=0.0001)
#     # 设置cpu/gpu服务
#     parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
#
#     # 将上面定义的这一个“队列”实例化为opt（这里有点“类”的味道了）
#     opt = parser.parse_args()
#
#     main(opt)