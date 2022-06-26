#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：hu
# albert time:2022/6/9


"""
FashionMNIST数据集分类，预处理
"""

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


batch_size = 64
learning_rate = 0.001

# 从公开数据集下载训练数据
training_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=False,
    transform=ToTensor(),
)

# 从公开数据集下载测试数据
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=False,
    transform=ToTensor(),
)

# 打包数据集
train_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 查看数据集
for X, Y in train_data_loader:
    print("一个batch数据的维度[N,C,H,W]:", X.shape)  # [N,C,H,W] [在索引中的编号,通道,高,宽]
    print("一个batch标签的维度: ", Y.shape)
    break

print(f"batch_size的大小{batch_size}")
print("训练集batch的个数: ", len(train_data_loader))
print("测试集batch的个数: ", len(test_data_loader))
print("训练集图片个数:", len(training_data))  # 训练集图片个数
print('测试集图片个数:', len(test_data))  # 测试集图片个数
print("训练集最后一个batch图片数量", len(training_data) - (len(train_data_loader) - 1) * 64)
print("测试集最后一个batch图片数量", len(test_data) - (len(test_data_loader) - 1) * 64)
