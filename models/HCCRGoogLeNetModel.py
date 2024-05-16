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
    def __init__(self, num_classes=1000, in_channels = 1):
        super(GaborGoogLeNet, self).__init__()
        # self.gabor_layer = GaborLayer(in_channels=in_channels, out_channels=8, kernel_size=5)


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
        # self.googlenet = models.googlenet(pretrained=False, 
        #                                   init_weights=True, 
        #                                   num_classes=num_classes)
        self.googlenet = GoogLeNet(in_channels=in_channels, num_classes=num_classes)
        # self.googlenet.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x = self.gabor_layer(x)
        x = self.googlenet(x)
        return x



def BasicConv2d(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


class InceptionV1Module(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce,
                 out_channels3, out_channels4):
        super(InceptionV1Module, self).__init__()
        # 线路1，单个1×1卷积层
        self.branch1_conv = BasicConv2d(in_channels, out_channels1, kernel_size=1)
        # 线路2，1×1卷积层后接3×3卷积层
        self.branch2_conv1 = BasicConv2d(in_channels, out_channels2reduce, kernel_size=1)
        self.branch2_conv2 = BasicConv2d(out_channels2reduce, out_channels2, kernel_size=3)
        # 线路3，1×1卷积层后接5×5卷积层
        self.branch3_conv1 = BasicConv2d(in_channels, out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = BasicConv2d(out_channels3reduce, out_channels3, kernel_size=5)
        # 线路4，3×3最大池化层后接1×1卷积层
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = BasicConv2d(in_channels, out_channels4, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv(self.branch4_pool(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class GoogLeNet(nn.Module):
    def __init__(self, num_classes, in_channels = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception3 = nn.Sequential(
            InceptionV1Module(192, 64, 96, 128, 16, 32, 32),
            InceptionV1Module(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception4 = nn.Sequential(
            InceptionV1Module(480, 192, 96, 208, 16, 48, 64),
            InceptionV1Module(512, 160, 112, 224, 24, 64, 64),
            InceptionV1Module(512, 128, 128, 256, 24, 64, 64),
            InceptionV1Module(512, 112, 144, 288, 32, 64, 64),
            InceptionV1Module(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception5 = nn.Sequential(
            InceptionV1Module(832, 256, 160, 320, 32, 128, 128),
            InceptionV1Module(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            # nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def main():
    # 示例用法
    model = GaborGoogLeNet(num_classes=1000)
    print(model)
    pass

if __name__ == '__main__':
    main()
