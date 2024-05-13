# -*- encoding: utf-8 -*-
'''
@File		:	normalization_memtion_nonlinear.py
@Time		:	2024/05/13 10:22:37
@Author		:	dan
@Description:	质心归一化 + 非线性均衡归一化
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
from normalization_memtion import remap as memtion_remap
from normalization_nonlinear_global import remap as global_remap


def remap(image : np.ndarray, target_size : tuple[2] | list[2] = (64, 64), show_plt : bool = False) -> np.ndarray:
    ''' 
    
    Parameters
    ----------
    image : np.ndarray 使用cv.imread方法加载

    show_plt 是否显示plt图，用于测试
    
    Returns
    -------
    np.ndarray

    '''
    memtion_image = memtion_remap(image=image, target_size=target_size)
    global_image = global_remap(image=memtion_image, target_size=target_size)
    
    if show_plt:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("原始图像")

        plt.subplot(1, 2, 2)
        plt.imshow(global_image, cmap='gray')
        plt.title("使用质心 + 整体密度均衡的非线性归一化图形")

        plt.tight_layout()
        plt.show()

    pass

def main():
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


    # 读取原始图像
    image_path = 'handwritten_chinese.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    remap(image=image, show_plt=True)
    pass

if __name__ == '__main__':
    main()