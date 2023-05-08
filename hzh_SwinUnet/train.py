import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import hzh_swin_unet_window7_448_448 as create_model
from utils import train_one_epoch, evaluate
import numpy as np


def main(args):
    # 如果没有GPU服务就CPU服务
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 新建一个文件来存放weight
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 新建一个文件夹来存放每个epoch的loss
    if os.path.exists("./loss") is False:
        os.makedirs("./loss")

    # TensorBoard是一个独立的包（不是pytorch中的），这个包的作用就是可视化您模型中的各种参数和结果。
    # tensorboard文件保存在当前目录下的runs/hzh_swin_unet文件夹中
    tb_writer = SummaryWriter(log_dir="runs/hzh_swin_unet")

    # 训练集、测试集路径
    train_x_path = './hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Rec photos/train_images_x'
    train_y_path = './hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Pre photos/train_images_y'
    val_x_path = './hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Rec photos/val_images_x'
    val_y_path = './hzh_SwinT_50_sensor_Data/Crop Skin Cancer MNIST Pre photos/val_images_y'

    # 实例化训练数据集
    # 分别传入训练集路径、训练集标签、对训练集进行预处理的transform
    train_dataset = MyDataSet(root_x=train_x_path,
                              root_y=train_y_path,)
    # 实例化验证数据集
    val_dataset = MyDataSet(root_x=val_x_path,
                            root_y=val_y_path,)

    # 获取当前系统能允许的最大载入线程数
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 加载训练集
    # batch_size:每一批的图片数量
    # shuffle：数据集是否要打乱
    # num_workers：载入数据的线程数
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
    # 加载测试集
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    # 实例化transformer模型
    model = create_model().to(device)

    # 中断训练后如果想继续，则用下面的代码读取之前训练的模型
    # model.load_state_dict(torch.load('./weights/model-9.pth'))

    # 实例化优化器（adamw）
    # 优化器的第一个参数是我们想训练的参数，lr是学习率
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    # 用于存放各个epoch的train_loss和val_loss
    train_loss_list = []
    val_loss_list = []

    for epoch in range(args.epochs):
        # 训练
        epoch = epoch
        train_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        # 测试
        val_loss = evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch)

        tags = ["train_loss", "val_loss", "learning_rate"]
        # 将这一次epoch的训练集损失存放在tensorboard中
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        # 将这一次epoch的测试集损失存放在tensorboard中
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        # 将这一次epoch的学习率存放在tensorboard中
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # 将本epoch的train_loss和val_loss存放到list里面
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # 保存模型参数
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 最后，将各个epoch的train_loss和val_loss存放起来
    train_loss_list = np.array(train_loss_list)
    val_loss_list = np.array(val_loss_list)
    # 保存为.npy格式
    np.save('./loss/train_loss_list.npy', train_loss_list)
    np.save('./loss/val_loss_list.npy', val_loss_list)


if __name__ == '__main__':
    # argparse我把它理解为一个列表，比如你写parser.add_argument('--num_classes', type=int, default=5)
    # 就相当于再这个列表里加了一个参数num_classes，你可以通过parser.num_classes得到5
    parser = argparse.ArgumentParser()
    # 训练次数设置为10次
    parser.add_argument('--epochs', type=int, default=200)
    # batch size设置为16
    parser.add_argument('--batch-size', type=int, default=16)
    # 学习率设置为0.0001
    parser.add_argument('--lr', type=float, default=0.0001)
    # 设置cpu/gpu服务
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # 将上面定义的这一个“队列”实例化为opt（这里有点“类”的味道了）
    opt = parser.parse_args()

    main(opt)
