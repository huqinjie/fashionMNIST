#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：hu
# albert time:2022/6/9

"""
FashionMNIST数据集分类，调用模型并验证
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import DNN

# 网络类实例化
net = DNN()
# 导入网络的参数
net.load_state_dict(torch.load("DNN.pth"))
# 加载测试集
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=False,
    transform=ToTensor(),
)
# 图片类别
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

correct_num = 0
for idx, (x, y) in enumerate(test_data):
    # 测试集不用更新参数,不记录梯度
    net.eval()
    with torch.no_grad():
        out = net(x)  # pred是一个二维[1x10]的tensor
        pred = out[0].argmax()  # 取一维tensor中最大值的索引
        if pred == y:
            correct_num += 1
        predicted = classes[pred]  # 预测类别
        actual = classes[y]  # 真实类别
        print(f'第{idx}张图片:')
        print(f'预测值: "{predicted}", 实际值: "{actual}"')

print("\n模型在测试集上的准确率为{:.4f}%".format(correct_num / len(test_data) * 100))
