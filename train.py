#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：hu
# albert time:2022/6/9

"""
FashionMNIST数据集分类，训练模型并保存
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import CNN

# 参数设置
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

train_data, val_data = torch.utils.data.random_split(training_data, [50000, 10000])

# 打包数据集
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# 使用cuda加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将模型送入GPU中
model = CNN().to(device)
# 打印网络参数
print(model)
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 损失函数
criterion = nn.CrossEntropyLoss()


# 训练过程
def train(train_loader, net):
    size = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播和计算误差
        out = net(data)
        loss = criterion(out, target)  # 交叉熵会自动对y进行one-hot

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型

        if batch_idx % 100 == 0:  # 每100个batch打印一下误差
            print('Loss: {:.6f}\t[{:>5d}/{} ({:>4.1f}%)]'.format(
                loss.item(), batch_idx * len(data), size,
                100. * batch_idx * len(data) / size))


# 验证过程
def val(val_loader, net):
    val_loss, correct = 0, 0
    net.eval()  # 测试时不更新参数
    with torch.no_grad():  # 测试集不用更新参数,不记录梯度
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            out = net(data)
            val_loss += criterion(out, target).item()
            pred = out.argmax(1)
            correct += pred.eq(target.data).sum().item()

    val_loss /= len(val_loader)
    correct /= len(val_loader.dataset)
    print(f"Test:\nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")


if __name__ == '__main__':
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n------------------------------")
        print("Training:")
        train(train_data_loader, model)
        val(val_data_loader, model)
    print("Done!")

    torch.save(model.state_dict(), "CNN.pth")
    print("把模型参数保存在CNN.pth")
