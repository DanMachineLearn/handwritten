# -*- encoding: utf-8 -*-
'''
@File		:	feature_grad-12.py
@Time		:	2024/05/13 14:23:27
@Author		:	dan
@Description:	通过grad-12 提取特征
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt  # 用于绘图

def get_features(image : np.ndarray, show_plt : bool = False):
    ''' 
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    
    '''
    # Sobel参数
    sobel_size = 3  # 滤波器大小
    angles = np.linspace(0, 2 * np.pi, 13)[:-1]  # 生成12个方向的角度，不包含2*pi

    angle_0 = 2 * np.pi / 12

    # 创建一个空列表，用于存储特征图
    feature_images = []
    features = []

    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向

    height_count = image.shape[0] // 8
    width_count = image.shape[1] // 8

    n = 0
    # 对每个角度计算Sobel滤波
    for angle in angles:

        gn = np.sin((n + 1) * angle_0) / np.sin(angle_0) * gx - np.cos((n + 1) * angle_0) / np.sin(angle_0) * gy
        # gn_1 = np.cos(n * angle_0) / np.sin(angle_0) * gy - np.sin(n * angle_0) / np.sin(angle_0) * gx
        n += 1

        feature_images.append(gn)  # 将特征图添加到列表中
        xxx_list = np.vsplit(gn, height_count)
        for xx in xxx_list:
            x_list = np.hsplit(xx, width_count)
            for x in x_list:
                # block_8_8 = np.sum(x)
                # 改为亚采样，即获取 该区域的平均值
                block_8_8 = np.mean(x)
                features.append(block_8_8)
        

    if show_plt:
        # 创建一个3x4的图网格来显示结果
        plt.figure(figsize=(12, 9))
        for i, feature_map in enumerate(feature_images):
            plt.subplot(3, 4, i + 1)
            plt.imshow(feature_map, cmap='gray')
            plt.title(f' {int(angles[i] * 180 / np.pi)} 度')
            plt.axis('off')  # 关闭坐标轴

        plt.tight_layout()  # 调整子图间距
        plt.show()  # 显示图

    return np.array(features).flatten()


def main():
    from normalization_nonlinear_global import remap
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

    # 读取原始图像
    image_path = 'handwritten_chinese.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = remap(image=image)
    features = get_features(image, show_plt=True)
    print(features)
    pass

if __name__ == '__main__':
    main()