# -*- encoding: utf-8 -*-
'''
@File		:	grad-12.py
@Time		:	2024/05/09 16:20:03
@Author		:	dan
@Description:	使用梯度方向特征分解图片
'''
import cv2  # 用于图像处理
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
from math import pi, cos, sin  # 用于计算方向角

# 读取图像
image_path = 'deng.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像

# Sobel参数
sobel_size = 3  # 滤波器大小
angles = np.linspace(0, 2 * pi, 13)[:-1]  # 生成12个方向的角度，不包含2*pi

angle_0 = 2 * pi / 12

# 创建一个空列表，用于存储特征图
feature_maps = []

gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向
gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向

n = 0
# 对每个角度计算Sobel滤波
for angle in angles:

    gn = np.sin((n + 1) * angle_0) / np.sin(angle_0) * gx - np.cos((n + 1) * angle_0) / np.sin(angle_0) * gy
    gn_1 = np.cos(n * angle_0) / np.sin(angle_0) * gy - np.sin(n * angle_0) / np.sin(angle_0) * gx
    # g = gn ** 2 + gn_1 ** 2 + 2 * gn * gn_1 * np.cos(angle_0)
    # g = np.sqrt(g)
    n += 1

    feature_maps.append(gn)  # 将特征图添加到列表中

# 创建一个3x4的图网格来显示结果
plt.figure(figsize=(12, 9))
for i, feature_map in enumerate(feature_maps):
    plt.subplot(3, 4, i + 1)
    plt.imshow(feature_map, cmap='gray')
    plt.title(f'Angle {int(angles[i] * 180 / pi)} degrees')
    plt.axis('off')  # 关闭坐标轴

plt.tight_layout()  # 调整子图间距
plt.show()  # 显示图
