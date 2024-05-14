# -*- encoding: utf-8 -*-
'''
@File		:	normalization_nonlinear.py
@Time		:	2024/05/14 12:21:51
@Author		:	dan
@Description:	简单非线性归一化
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def remap(image : np.ndarray, target_size : tuple[2] | list[2] = (64, 64), show_plt : bool = False) -> np.ndarray:
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
    # 膨胀和腐蚀都是对于白色像素而言的，所以对于黑色的膨胀，则需要进行白色的腐蚀。
    kernel = np.ones((5, 5), dtype=np.uint8) # 卷积核变为4*4
    image = cv2.erode(image, kernel, iterations=1)

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


    H1, W1 = shape_image.shape
    R1 = W1 / H1 if W1 <= H1 else H1 / W1
    # R2 = 1
    R2 = R1
    # R2 = R1 ** (1 / 2)
    # R2 = R1 ** (1 / 3)
    if W1 <= H1:
        H2 = int(target_height)
        W2 = int(H2 * R2)
    else:
        W2 = int(target_height)
        H2 = int(R2 * W2)
    
    Xc = center_x
    Yc = center_y
    Yc_ = target_height // 2
    Xc_ = target_width // 2

    # map_x = np.zeros((H2, W2), dtype=np.float32)
    # map_y = np.zeros((H2, W2), dtype=np.float32)
    # for y in range(H2):
    #     for x in range(W2):
    #         k = W2 / W1 * (x - Xc) + Xc_
    #         l = H2 / H1 * (y - Yc) + Yc_
    #         map_x[y, x] = int(k)
    #         map_y[y, x] = int(l)
    left_top = shape_image[0 : center_y, 0 : center_x]
    left_bottom = shape_image[center_y : shape_image.shape[0], 0 : center_x]
    right_top = shape_image[0 : center_y, center_x : shape_image.shape[1]]
    right_bottom = shape_image[center_y : shape_image.shape[0], center_x : shape_image.shape[1]]

    half_height = H2 // 2
    half_width = W2 // 2
    left_top = cv2.resize(left_top, (half_width, half_height), interpolation=cv2.INTER_LINEAR)
    left_bottom = cv2.resize(left_bottom, (half_width, half_height), interpolation=cv2.INTER_LINEAR)
    right_top = cv2.resize(right_top, (half_width, half_height), interpolation=cv2.INTER_LINEAR)
    right_bottom = cv2.resize(right_bottom, (half_width, half_height), interpolation=cv2.INTER_LINEAR)


    if W1 <= H1:
        # 左右补充
        ll = (target_width - W2) // 2
        rr = ll + W2
        tt = 0
        bb = target_height
    else:
        # 上下填充
        tt = (target_height - H2) // 2
        bb = tt + H2
        ll = 0
        rr = target_width

    resized_image = np.zeros((target_height, target_width))
    resized_image.fill(255)
    resized_image[tt : tt + half_height, ll : ll + half_width] = left_top
    resized_image[Yc_ : Yc_ + half_height, ll : ll + half_width] = left_bottom
    resized_image[tt : tt + half_height, Xc_ : Xc_ + half_width] = right_top
    resized_image[Yc_ : Yc_ + half_height, Xc_ : Xc_ + half_width] = right_bottom
    resized_image = resized_image.astype(np.uint8)


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
        plt.title("使用非线性重心归一化")

        plt.tight_layout()
        plt.show()

    return resized_image
    


def main():
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


    # 读取原始图像
    # image_path = '1.png'
    image_path = 'handwritten_chinese.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    remap(image=image, show_plt=True)
    pass

if __name__ == '__main__':
    main()

