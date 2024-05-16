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

# 定义 Gabor 滤波器层
class GaborLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GaborLayer, self).__init__()
        self.out_channels = out_channels
        self.gabor_filters = nn.ParameterList()
        for _ in range(out_channels):
            self.gabor_filters.append(nn.Parameter(self.create_gabor_kernel(kernel_size), requires_grad=True))

    def create_gabor_kernel(self, kernel_size):
        # 示例代码，创建一个简单的 Gabor 滤波器
        theta = torch.randn(1).item() * 3.14159  # 随机方向
        sigma = kernel_size / 6.0
        lambd = kernel_size / 2.0
        gamma = 0.5
        psi = 0
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        y, x = torch.meshgrid(y, x)
        theta = torch.tensor(theta)
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
        gb = torch.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2) * torch.cos(2 * 3.14159 * x_theta / lambd + psi)
        return gb.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        outputs = []
        for gabor_filter in self.gabor_filters:
            outputs.append(F.conv2d(x, gabor_filter))
        return torch.cat(outputs, dim=1)

# 定义 Gabor-GoogLeNet 模型
class GaborGoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GaborGoogLeNet, self).__init__()
        self.gabor_layer = GaborLayer(in_channels=1, out_channels=8, kernel_size=5)


        ### 参数解释
        # pretrained (bool)：
        # 默认值：False
        # 说明：如果设置为 True，则返回一个在 ImageNet 数据集上预训练的模型。使用预训练的模型可以加速训练，并可能提高模型性能，特别是在较小的数据集上进行微调时。

        # progress (bool)：
        # 默认值：True
        # 说明：如果设置为 True，则在下载预训练模型的过程中会显示下载进度。

        # aux_logits (bool)：
        # 默认值：True
        # 说明：如果设置为 True，则包括辅助分类器。GoogLeNet 具有两个辅助分类器，用于在训练过程中辅助梯度传播。这些辅助分类器在推理（inference）阶段通常会被关闭。

        # transform_input (bool)：
        # 默认值：False
        # 说明：如果设置为 True，则在输入图像之前进行标准化处理。具体来说，将输入图像从 [0, 1] 范围内转换为 [-1, 1] 范围内。

        # init_weights (bool)：
        # 默认值：None
        # 说明：如果设置为 True，则对模型的权重进行初始化。这个参数通常不需要显式指定，因为当 pretrained=True 时，权重会自动加载。

        # blocks (list of BasicInceptionBlock)：
        # 默认值：None
        # 说明：允许指定自定义的 Inception 模块。这是一个高级选项，通常用于修改 GoogLeNet 的架构。

        # num_classes (int)：
        # 默认值：1000
        # 说明：分类器的输出类别数量。如果你的数据集类别数量不同于 1000（ImageNet 的类别数量），你需要根据你的数据集设置 num_classes。这个参数在进行迁移学习或微调时特别重要。
        self.googlenet = models.googlenet(pretrained=False, 
                                          init_weights=True, 
                                          num_classes=num_classes, 
                                          init_weights=True)
        # self.googlenet.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x = self.gabor_layer(x)
        x = self.googlenet(x)
        return x


def main():
    # 示例用法
    model = GaborGoogLeNet(num_classes=1000)
    print(model)
    pass

if __name__ == '__main__':
    main()
