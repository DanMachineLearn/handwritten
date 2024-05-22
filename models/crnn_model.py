# -*- encoding: utf-8 -*-
'''
@File		:	crnn_model.py
@Time		:	2024/05/21 17:33:29
@Author		:	dan
@Description:	根据gitee开源项目修改
'''

import torch
import torch.nn as nn
import numpy as np

# 批量归一化层
def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9, affine=True)

# 定义卷积层
class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, use_bn=False, pad_mode='same'):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2 if pad_mode == 'same' else 0
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = _bn(out_channel) if use_bn else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

# 定义VGG网络结构
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = Conv(3, 64, use_bn=True)
        self.conv2 = Conv(64, 128, use_bn=True)
        self.conv3 = Conv(128, 256, use_bn=True)
        self.conv4 = Conv(256, 256, use_bn=True)
        self.conv5 = Conv(256, 512, use_bn=True)
        self.conv6 = Conv(512, 512, use_bn=True)
        self.conv7 = Conv(512, 512, kernel_size=2, pad_mode='valid', use_bn=True)
        self.maxpool2d1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool2d1(x)
        x = self.conv2(x)
        x = self.maxpool2d1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2d2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool2d2(x)
        x = self.conv7(x)
        return x

# 定义CRNN网络，包括双向LSTM层和VGG层
class CRNN(nn.Module):
    def __init__(self, config):
        super(CRNN, self).__init__()
        self.batch_size = config.batch_size
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_classes = config.class_num
        self.vgg = VGG()
        
        self.rnn1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=self.hidden_size * 2, hidden_size=self.hidden_size, num_layers=1, bidirectional=True)
        
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)
        self.use_dropout = config.use_dropout
        self.dropout = nn.Dropout(0.5)
        self.rnn_dropout = nn.Dropout(0.9)
    
    def forward(self, x):
        x = self.vgg(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = x.permute(2, 0, 1)
        
        y1_out, _ = self.rnn1(x)
        if self.use_dropout:
            y1_out = self.rnn_dropout(y1_out)
        
        y2_out, _ = self.rnn2(y1_out)
        if self.use_dropout:
            y2_out = self.dropout(y2_out)
        
        T, b, h = y2_out.size()
        y2_out = y2_out.view(T * b, h)
        output = self.fc(y2_out)
        output = output.view(T, b, -1)
        
        return output

# 配置类，用于保存模型参数
class Config:
    def __init__(self, batch_size, input_size, hidden_size, class_num, num_step, use_dropout):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.class_num = class_num
        self.num_step = num_step
        self.use_dropout = use_dropout

# 创建CRNN网络
def crnn(config, full_precision=False):
    net = CRNN(config)
    if not full_precision:
        net = net.half()
    return net


def main():
    # 示例配置
    config = Config(batch_size=64, input_size=512, hidden_size=256, class_num=10, num_step=32, use_dropout=True)
    model = crnn(config)
    pass

if __name__ == '__main__':
    main()
