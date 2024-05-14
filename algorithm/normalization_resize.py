# -*- encoding: utf-8 -*-
'''
@File		:	normalization_resize.py
@Time		:	2024/05/13 10:27:54
@Author		:	dan
@Description:	直接调整图像大小
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    # 获取非空区域的边界
    # 使用行和列求和来判断非空区域
    rows_sum = np.sum(image - 255, axis=1)
    cols_sum = np.sum(image - 255, axis=0)
    # 找到行和列的最小和最大非空索引
    top = np.argmax(rows_sum > 0)  # 第一个非空行
    bottom = len(rows_sum) - np.argmax(rows_sum[::-1] > 0)  # 最后一个非空行
    left = np.argmax(cols_sum > 0)  # 第一个非空列
    right = len(cols_sum) - np.argmax(cols_sum[::-1] > 0)  # 最后一个非空列
    shape_image = image[top:bottom, left:right]

    target_image = cv2.resize(shape_image, target_size, interpolation=cv2.INTER_LINEAR)
    
    if show_plt:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("原始图像")

        plt.subplot(1, 2, 2)
        plt.imshow(target_image, cmap='gray')
        plt.title("直接调整图像大小")

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