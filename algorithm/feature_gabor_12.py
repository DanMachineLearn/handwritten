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

def get_features(image : np.ndarray, show_plt : bool = False, image_only=False, direction_count=12):
    ''' 
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    
    '''
    # 定义Gabor滤波器的参数
    ksize = (31, 31)  # 滤波器的大小
    sigma = 4.0  # 标准差
    theta = 0  # 初始角度
    lambd = 10.0  # 波长
    gamma = 0.5  # 纵横比
    psi = 0  # 相位偏移

    # 创建Gabor滤波器
    gabor_bank = []
    for angle in np.linspace(0, np.pi, 12, endpoint=False):  # 分12个角度
        gabor_filter = cv2.getGaborKernel(ksize, sigma, angle, lambd, gamma, psi, ktype=cv2.CV_32F)
        gabor_bank.append(gabor_filter)

    # 应用Gabor滤波器
    gabor_images = []
    for gabor_filter in gabor_bank:
        filtered_img = cv2.filter2D(image, cv2.CV_32F, gabor_filter)  # 应用滤波器
        gabor_images.append(filtered_img)

    return gabor_images;

    # if show_plt:
    #     # 创建一个3x4的图网格来显示结果
    #     plt.figure(figsize=(12, 9))
    #     for i, feature_map in enumerate(feature_maps):
    #         plt.subplot(3, 4, i + 1)
    #         plt.imshow(feature_map, cmap='gray')
    #         plt.title(f' {int(angles[i] * 180 / np.pi)} 度')
    #         plt.axis('off')  # 关闭坐标轴

    #     plt.tight_layout()  # 调整子图间距
    #     plt.show()  # 显示图

    # return np.array(feature_maps).flatten()


def main():
    from normalization_nonlinear_global import remap
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

    # 读取原始图像
    image_path = 'deng.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = remap(image=image)
    get_features(image, show_plt=True)
    pass

if __name__ == '__main__':
    main()