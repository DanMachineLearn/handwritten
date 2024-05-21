# -*- encoding: utf-8 -*-
'''
@File		:	normalization_memtion.py
@Time		:	2024/05/13 10:10:09
@Author		:	dan
@Description:	实现质心归一化
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def remap(image : np.ndarray, target_size : tuple[2] | list[2] = (64, 64), show_plt : bool = False, need_dilate = True) -> np.ndarray:
    ''' 
    采用基于笔画的整体归一化方法
    Parameters
    ----------
    image : np.ndarray 使用cv2.imread方法加载

    show_plt 是否显示plt图，用于测试
    
    Returns
    -------
    np.ndarray
    
    '''

    # 获取图像尺寸
    height, width = image.shape

    # 目标大小
    target_height, target_width = target_size

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



    
    # 计算图像的重心
    # 使用图像的矩计算质心
    _, binary_image = cv2.threshold(shape_image, 255 // 2, 1, cv2.THRESH_BINARY_INV);

    # 获取非零像素的坐标
    nonzero_indices = np.nonzero(binary_image)
    # 计算质心
    centroid : np.ndarray = np.mean(nonzero_indices, axis=1)
    center_y, center_x = centroid.astype(np.int32)

    # moments = cv2.moments(shape_image, True)
    # center_x = int(moments['m10'] / moments['m00'])  # x 轴重心
    # center_y = int(moments['m01'] / moments['m00'])  # y 轴重心
    left_top = shape_image[0 : center_y, 0 : center_x]
    left_bottom = shape_image[center_y : shape_image.shape[0], 0 : center_x]
    right_top = shape_image[0 : center_y, center_x : shape_image.shape[1]]
    right_bottom = shape_image[center_y : shape_image.shape[0], center_x : shape_image.shape[1]]

    half_width = target_width // 2
    half_height = target_height // 2

    left_top = cv2.resize(left_top, (half_height, half_width), interpolation=cv2.INTER_LINEAR)
    left_bottom = cv2.resize(left_bottom, (half_height, half_width), interpolation=cv2.INTER_LINEAR)
    right_top = cv2.resize(right_top, (half_height, half_width), interpolation=cv2.INTER_LINEAR)
    right_bottom = cv2.resize(right_bottom, (half_height, half_width), interpolation=cv2.INTER_LINEAR)

    resized_image = np.zeros((target_height, target_width))
    resized_image[0 : half_height, 0 : half_width] = left_top
    resized_image[half_height : target_height, 0 : half_width] = left_bottom
    resized_image[0 : half_height, half_width : target_width] = right_top
    resized_image[half_height : target_height, half_width : target_width] = right_bottom
    resized_image = resized_image.astype(np.uint8)

    # ## 使用直方图均衡化
    # resized_image = cv2.equalizeHist(resized_image)

    # # 3. 调整图像大小
    # resized_image = resized_image.reshape((1, target_height, target_width))


    # 显示原始图像和重映射后的图像
    if show_plt:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.vlines(center_x + left, 0, height, colors='r', label='', linestyles='--')
        plt.hlines(center_y + top, 0, width, colors='r', label='', linestyles='--')
        plt.vlines(left, top, bottom, colors='k', linestyles='-')
        plt.vlines(right, top, bottom, colors='k', linestyles='-')
        plt.hlines(top, left, right, colors='k', linestyles='-')
        plt.hlines(bottom, left, right, colors='k', linestyles='-')
        plt.title("原始图像")

        plt.subplot(1, 2, 2)
        plt.imshow(resized_image, cmap='gray')
        plt.title("使用重心归一化图形")

        plt.tight_layout()
        plt.show()

    return resized_image
    


def main():
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


    # 读取原始图像
    image_path = '1.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    remap(image=image, show_plt=True)
    pass

if __name__ == '__main__':
    main()

