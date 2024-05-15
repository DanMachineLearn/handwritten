# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_model_cnn.py
@Time		:	2024/05/08 09:54:23
@Author		:	dan
@Description:	手写识别卷积模型
'''

import torch
from torch import nn, tensor
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# 定义神经网络模型
class HandWrittenCnnModel(nn.Module):
    def __init__(self, input_shape, output_classes):
        super(HandWrittenCnnModel, self).__init__()
        # 定义隐藏层
        self.layers = []
        # 输入层到第一个隐藏层
        self.layers.append(nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=(3, 3), stride=1, padding=1))
        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), padding=0))
        self.layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1))
        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), padding=0))
        self.layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1))
        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), padding=0))
        self.layers.append(nn.Flatten()) # 数据展平
        self.layers.append(nn.Linear(256 * (input_shape[1] // 8) * (input_shape[2] // 8), 1024))  # 注意这个数字，确保与模型结构匹配
        self.layers.append(nn.Linear(1024, output_classes))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Softmax(dim=1))


        # 根据 https://blog.csdn.net/Blossomers/article/details/124080960 的描述，CrossEntropyLoss 本身已经内置了softmax算法
        # 如果在模型层中再次添加softmax，会导致2次计算概率，导致假概率逼近0，真概率逼近1
        self.model = nn.Sequential(*self.layers)  # 使用 Sequential 构建模型

    def forward(self, x):

        # for l in self.layers:
        #     # print(f"x.shape = {x.shape}")
        #     x = l(x)
        

        x = self.model(x)
        # input: Softmax 的输入张量。可以是任何形状，但常用于二维张量，在这种情况下，每一行（或者每一列）通常代表一个样本或一个类得分
        # dim：沿着哪个维度应用 Softmax。如果 dim 是 0，则 Softmax 沿着第一维度进行；如果是 1，则沿着第二维度进行。在多分类任务中，
        # 通常指定 dim=1，表示每一行是一个样本，Softmax 操作在每行之间进行。
        # dtype：指定输出张量的数据类型。通常不需要手动指定，默认情况下会与 input 相同。
        # out：指定输出张量。通常不需要手动设置，默认情况下 F.softmax 返回一个新的张量。
        return x;