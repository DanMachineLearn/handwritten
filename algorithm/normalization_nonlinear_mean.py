# -*- encoding: utf-8 -*-
'''
@File		:	normalization_nonlinear_mean.py
@Time		:	2024/05/14 15:07:29
@Author		:	dan
@Description:	整体密度均衡的非线性归一化算法
'''



import cv2
import numpy as np
import matplotlib.pyplot as plt

def remap(image : np.ndarray, target_size : tuple[2] | list[2] = (64, 64), show_plt : bool = False) -> np.ndarray:
    ''' 
    采用基于笔画的整体归一化方法
    Parameters
    ----------
    image : np.ndarray 使用cv.imread方法加载

    show_plt 是否显示plt图，用于测试
    
    Returns
    -------
    np.ndarray
    
    '''
    origin_image = image

    # 获取非空区域的边界
    # 使用行和列求和来判断非空区域
    rows_sum = np.sum(image - 255, axis=1)
    cols_sum = np.sum(image - 255, axis=0)
    # 找到行和列的最小和最大非空索引
    top = np.argmax(rows_sum > 0)  # 第一个非空行
    bottom = len(rows_sum) - np.argmax(rows_sum[::-1] > 0)  # 最后一个非空行
    left = np.argmax(cols_sum > 0)  # 第一个非空列
    right = len(cols_sum) - np.argmax(cols_sum[::-1] > 0)  # 最后一个非空列
    image = image[top:bottom, left:right]

    # 获取图像尺寸
    height, width = image.shape

    # 初始化 h 和 v 数组
    h = np.full((height, width), 1, dtype=np.float32)
    v = np.full((height, width), 1, dtype=np.float32)

    aH1 = 1     ## 背景为空白的像素点
    aH2 = 2     ## 背景为笔画的像素点
    aH3 = 3     ## 周边是汉字的像素点
    aaH = np.full((height, width), aH2, dtype=np.float32)
    aaV = np.full((height, width), aH2, dtype=np.float32)

    # 膨胀和腐蚀都是对于白色像素而言的，所以对于黑色的膨胀，则需要进行白色的腐蚀。
    kernel = np.ones((5, 5), dtype=np.uint8) # 卷积核变为4*4
    image = cv2.erode(image, kernel, iterations=1)

    # 找到每一行和每一列的黑点位置
    black_threshold = 127  # 定义黑点的阈值
    black_indices_rows = [np.where(image[i, :] < black_threshold)[0] for i in range(height)]
    black_indices_cols = [np.where(image[:, j] < black_threshold)[0] for j in range(width)]
    for row in black_indices_rows:
        c_point_h = np.diff(row, axis=0)
        c_point_h = c_point_h - 1
        print(len(c_point_h))


    # 计算 h 和 v
    for i in range(height):
        if black_indices_rows[i].size > 1:
            line_start = True
            for j in range(1, len(black_indices_rows[i])):
                start = black_indices_rows[i][j - 1]
                end = black_indices_rows[i][j]


                if line_start:
                    line_start = False
                    white_length = start
                    for k in range(0, start):
                        h[i, k] = white_length / 2 - abs(k - start) + 1
                    aaH[i, start] = aH1
                    aaH[start - 3, start] = aH3

                white_length = end - start
                if white_length == 1:
                    aaH[i, start : end] = aH2

                h[i, start : end] = end - start

                # h 为所在点所在的白边长度，如果 背景点为笔画，则长度为0

    for j in range(width):
        if black_indices_cols[j].size > 1:
            for i in range(1, len(black_indices_cols[j])):
                start = black_indices_cols[j][i - 1]
                end = black_indices_cols[j][i]

    
    # 计算 Fh 和 Fv
    Fh = 1 / h
    Fv = 1 / v

    # 计算 H 和 V
    H = np.sum(Fh, axis=1)
    V = np.sum(Fv, axis=0)

    # 计算累加值
    cumulative_H = np.cumsum(H)  # 水平方向累加
    cumulative_V = np.cumsum(V)  # 垂直方向累加

    # 计算目标图像尺寸
    # target_width = 64  # 根据目标图像的宽度定义
    # target_height = 64  # 根据目标图像的高度定义
    target_width, target_height = target_size

    # 调整系数
    A = target_width / cumulative_H[-1]
    B = target_height / cumulative_V[-1]

    # 创建映射矩阵
    map_x = np.zeros((target_height, target_width), dtype=np.float32)
    map_y = np.zeros((target_height, target_width), dtype=np.float32)

    # 计算目标图像上的坐标
    k = np.round(A * cumulative_H[:, None]).astype(int)  # 水平方向的新坐标
    l = np.round(B * cumulative_V).astype(int)  # 垂直方向的新坐标


    # 填充映射矩阵
    maybe_y = np.round(A * cumulative_H)  # 水平方向的新坐标
    maybe_x = np.round(B * cumulative_V)  # 垂直方向的新坐标
    for y in range(target_height):
        for x in range(target_width):
            yy_left = np.searchsorted(maybe_y, y, side='left')
            xx_left = np.searchsorted(maybe_x, x, side='left')
            map_x[y, x] = xx_left
            map_y[y, x] = yy_left
            
    target_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)


    # 用坐标直接映射的办法，会导致最终图像产生锯齿
    # target_image2 = np.zeros((target_height, target_width), dtype=np.uint8)
    # # 将原始图像映射到目标图像
    # for i in range(height):
    #     for j in range(width):
    #         if image[i, j] < black_threshold:
    #             k_val = k[i]
    #             l_val = l[j]
    #             if 0 <= k_val < target_width and 0 <= l_val < target_height:
    #                 target_image2[k_val, l_val] = 255  # 设置为黑色


    # 显示原始图像和重映射后的图像
    if show_plt:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(origin_image, cmap='gray')
        plt.title("原始图像")

        plt.subplot(1, 2, 2)
        plt.imshow(target_image, cmap='gray')
        plt.title("使用整体密度均衡的非线性归一化图形")

        plt.tight_layout()
        plt.show()

    return target_image

    


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

