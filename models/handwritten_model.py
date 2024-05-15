# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_train.py
@Time		:	2024/05/06 16:15:09
@Author		:	dan
@Description:	训练手写识别的模型
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


# 隐藏层大小
HIDDEN_SIZE = [2048]


# 定义神经网络模型
class HandWrittenModel(nn.Module):
    def __init__(self, input_features, output_classes):
        super(HandWrittenModel, self).__init__()
        # 定义隐藏层
        self.layers = []
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_features, HIDDEN_SIZE[0]))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Sigmoid())
        # self.layers.append(nn.LeakyReLU(negative_slope=0.01))
        # self.layers.append(nn.Dropout(0.5))  # Dropout 用于正则化
        # 隐藏层
        for i in range(1, len(HIDDEN_SIZE)):
            self.layers.append(nn.Linear(HIDDEN_SIZE[i - 1], HIDDEN_SIZE[i]))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Sigmoid())
            # self.layers.append(nn.LeakyReLU(negative_slope=0.01))
            # self.layers.append(nn.Dropout(0.5))  # Dropout 用于正则化
        # 输出层
        self.layers.append(nn.Linear(HIDDEN_SIZE[-1], output_classes))  # 7000 个输出类


        # 根据 https://blog.csdn.net/Blossomers/article/details/124080960 的描述，CrossEntropyLoss 本身已经内置了softmax算法
        # 如果在模型层中再次添加softmax，会导致2次计算概率，导致假概率逼近0，真概率逼近1
        # self.layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*self.layers)  # 使用 Sequential 构建模型

    def forward(self, x):

        x = self.model(x)
        # input: Softmax 的输入张量。可以是任何形状，但常用于二维张量，在这种情况下，每一行（或者每一列）通常代表一个样本或一个类得分
        # dim：沿着哪个维度应用 Softmax。如果 dim 是 0，则 Softmax 沿着第一维度进行；如果是 1，则沿着第二维度进行。在多分类任务中，
        # 通常指定 dim=1，表示每一行是一个样本，Softmax 操作在每行之间进行。
        # dtype：指定输出张量的数据类型。通常不需要手动指定，默认情况下会与 input 相同。
        # out：指定输出张量。通常不需要手动设置，默认情况下 F.softmax 返回一个新的张量。
        return x;