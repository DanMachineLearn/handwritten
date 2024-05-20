# -*- encoding: utf-8 -*-
'''
@File		:	HCCRGoogLeNetModel.py
@Time		:	2024/05/15 11:42:56
@Author		:	dan
@Description:	根据论文 https://arxiv.org/pdf/1505.04925 设计的 GoogLeNet 网络模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 定义 Gabor-GoogLeNet 模型
class HandwrittenResNet(nn.Module):
    def __init__(self, num_classes=1000, in_channels = 1):
        super(HandwrittenResNet, self).__init__()
        self.res_net = models.resnet.resnet101()
        # self.googlenet.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x = self.gabor_layer(x)
        x = self.res_net(x)
        return x


def main():
    # 示例用法
    model = HandwrittenResNet(num_classes=1000)
    print(model)
    pass

if __name__ == '__main__':
    main()
