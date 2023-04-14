import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

# 这里定义了训练一个epoch所做的操作
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    # model.train() 就是告诉 Dropout 层，你下面应该遮住一神经元
    # model.test() 就是告诉 Dropout 层，你下面别遮住了，我全都需要
    model.train()
    # 定义损失函数
    loss_function = torch.nn.MSELoss()
    # 定义累计损失
    accu_loss = torch.zeros(1).to(device)
    # 清空梯度
    optimizer.zero_grad()

    # 样本数等于0
    sample_num = 0
    # 这是用于可视化进度的
    data_loader = tqdm(data_loader, file=sys.stdout)
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    for step, data in enumerate(data_loader):
        x_image, y_image = data
        # 累加样本数量
        sample_num += x_image.shape[0]

        # pred是将图像输入神经网络后得到的输出
        y_pred = model(x_image.to(device))

        y_image = y_image.to(torch.float32)
        # 算出loss
        loss = loss_function(y_image.to(device), y_pred.to(device))
        # 反向传播
        loss.backward()
        # 累计loss
        accu_loss += loss.detach()

        # 可视化进度
        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        # 查看这个loss是不是有限的。如果是无穷大、小，则报错
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # optimizer.step()是优化器更新
        # 只有用了optimizer.step(),模型才会更新。
        optimizer.step()
        # 清空梯度
        optimizer.zero_grad()

    # 返回这次epoch的平均损失和平均预测正确的样本数
    return accu_loss.item() / (step + 1)

# @torch.no_grad()中的数据不需要计算梯度,也不会进行反向传播
@torch.no_grad()
# 在经过一个epoch的训练之后呢，下面的函数就是测试啦
def evaluate(model, data_loader, device, epoch):
    # 定义loss function
    loss_function = torch.nn.MSELoss()

    # 不启用 BatchNormalization 和 Dropout
    model.eval()

    # 定义累计损失
    accu_loss = torch.zeros(1).to(device)

    # 样本数初始化为0
    sample_num = 0
    # 这是用于可视化进度的
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        x_image, y_image = data
        # 累计样本数
        sample_num += x_image.shape[0]

        # 将图片输入模型
        y_pred = model(x_image.to(device))

        y_image = y_image.to(torch.float32)
        # 算loss
        loss = loss_function(y_pred.to(device), y_image.to(device))
        # 累计loss
        accu_loss += loss

        # 可视化进度
        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

    # 返回这次测试的平均损失和平均预测正确的样本数
    return accu_loss.item() / (step + 1)
