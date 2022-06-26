#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：hu
# albert time:2022/6/9

"""
FashionMNIST数据集分类，定义模型
"""

from torch import nn


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),  # 有10个类别
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)  # 将原始图片数据[28x28]展平成一维[784]
        x = self.linear_relu_stack(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 6 * 6, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 10),  # 有10个类别
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.classifier(x)
        return x


class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),

            nn.Conv2d(40, 80, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),

            nn.Conv2d(80, 160, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)))

        self.classifier = nn.Sequential(
            nn.Linear(160 * 8 * 8, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 160 * 8 * 8)
        x = self.classifier(x)
        return x


class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10))

    def forward(self, x):
        x = self.conv(x)
        # print(" x shape ", x.size())
        x = x.view(-1, 256 * 8 * 8)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = CNN_2()
    print(model)
