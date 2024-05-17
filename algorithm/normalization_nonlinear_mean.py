# -*- encoding: utf-8 -*-
'''
@File		:	normalization_nonlinear_mean.py
@Time		:	2024/05/14 15:07:29
@Author		:	dan
@Description:	整体密度均衡的非线性归一化算法
'''



import cv2
from matplotlib.axes import Axes
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import math



def get_dx(image, aH1 = 1, aH2 = 2, aH3 = 3,around = 1):
    '''
    计算水平方向的密度函数
    aH1 = 1     ## 背景为空白的像素点
    aH2 = 2     ## 背景为笔画的像素点
    aH3 = 3     ## 周边是汉字的像素点
    around = 1  ## 多少范围内算作周边
    '''
    # 获取所有黑点位置
    _, binary_image = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY_INV)
    # print(f"height = {binary_image.shape[0]}, width = {binary_image.shape[1]}")

    # 获取图像尺寸
    height, width = binary_image.shape

    # 往最左边补充 0 计算
    row_line_info = np.hstack((np.zeros((binary_image.shape[0], 1)), binary_image, np.zeros((binary_image.shape[0], 1))))
    row_line_info = np.diff(row_line_info, axis=1)
    row_line_count = np.absolute(row_line_info).sum(axis=1) / 2

    col_line_info = np.vstack((np.zeros((1, binary_image.shape[1])), binary_image, np.zeros((1, binary_image.shape[1]))))
    col_line_info = np.diff(col_line_info, axis=0)
    col_line_count = np.absolute(row_line_info).sum(axis=0) / 2

    # 水平方向
    row_line_start = np.where(row_line_info > 0)
    row_line_end = np.where(row_line_info < 0)
    assert len(row_line_start[0]) == len(row_line_end[0])


    ## TODO 计算  aBFH 
    aBFH = np.zeros(binary_image.shape)
    ## 为背景填充 aH1
    aBFH.fill(aH1)

    ## 为画笔周边填充 aH3
    for i in range(len(row_line_start)):
        start_h, start_w = row_line_start[0][i], row_line_start[1][i]
        end_h, end_w = row_line_end[0][i], row_line_end[1][i]
        minh = max(start_h - around, 0)
        maxh = min(start_h + around, height)
        minw = max(start_w - around, 0)
        maxw = min(end_w + around, width)
        aBFH[minh : maxh][minw : maxw] = aH3

    ## 为画笔本身填充 aH2
    aBFH[np.where(binary_image == 1)] = aH2

    ## 计算水平密度函数
    dx = np.zeros(binary_image.shape)
    last_c = 0
    for i in range(len(row_line_start[0])):
        assert row_line_start[0][i] == row_line_end[0][i]


        start = row_line_start[1][i]
        end = row_line_end[1][i]

        h = row_line_start[0][i]
        line_count = row_line_count[h]

        ## 如果该行只有1条线
        if line_count == 1:
            # 计算c点的位置
            if start == 0:
                c = start
            else:
                c = (start + end) // 2
            
            for w in range(width):
                if w < c:
                    o = c // 2
                else:
                    o = c + (width - c) // 2
                oc = abs(o - c)
                fenmu = (oc - abs(w - o) + 1)
                assert fenmu != 0
                dx[h, w] = 1 / fenmu + aBFH[h, w]
            
            continue

        ## 如果该行有大于或等于2条线，需要获取d点位置，也就是下一条线的位置
        if line_count > 1:

            # 计算c点的位置
            if start == 0:
                c = start
            else:
                c = (start + end) // 2

            # 如果最后一个点，或者 下个点的行位置和当前不一致，说明这条线是最后的
            if i == len(row_line_start[0]) - 1:
                c = width
            else:
                next_line_start = row_line_start[0][i + 1]
                if next_line_start != h:
                    c = width

            # assert d > c
            cd = c - last_c
            cd2 = cd / 2
            o = last_c + cd2
            for w in range(last_c, c):
                d = 1 / (cd2 - abs(w - o) + 1) + aBFH[h, w]
                dx[h, w] = d
            last_c = c
            if last_c == width:
                last_c = 0
        else:
            assert False 
    return dx


def remap(image : np.ndarray, target_size : tuple[2] | list[2] = (64, 64), show_plt : bool = False, 
          aH1 = 1, aH2 = 2, aH3 = 3,around = 1) -> np.ndarray:
    ''' 
    采用基于笔画的整体归一化方法
    Parameters
    ----------
    image : np.ndarray 使用cv.imread方法加载

    show_plt 是否显示plt图，用于测试

    aH1 = 1     ## 背景为空白的像素点
    aH2 = 2     ## 背景为笔画的像素点
    aH3 = 3     ## 周边是汉字的像素点
    around = 1  ## 多少范围内算作周边
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


    # 计算水平和垂直的密度函数
    dx = get_dx(image=image, aH1=aH1, aH2=aH2, aH3=aH3, around=around)
    dy = get_dx(image=image.T, aH1=aH1, aH2=aH2, aH3=aH3, around=around).T

    # 整体密度函数
    dxy = np.maximum(dx, dy)
    # 水平密度函数
    H = dxy.sum(axis=1)
    # 垂直密度函数
    V = dxy.sum(axis=0)

    image_map_y = np.zeros(dxy.shape).astype(np.float32)
    image_map_x = np.zeros(dxy.shape).astype(np.float32)
    M, N = dxy.shape[0], dxy.shape[1]
    for j in range(N):
        for i in range(M):
            image_map_y[i, j] = round(np.sum(H[0: i]) * M / np.sum(H))
            image_map_x[i, j] = round(np.sum(V[0: j]) * N / np.sum(V))
    image = cv2.remap(image, image_map_x, image_map_y, cv2.INTER_LINEAR)

    # 显示原始图像和重映射后的图像
    if show_plt:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(origin_image, cmap='gray')
        plt.title("原始图像")

        plt.subplot(1, 3, 2)
        plt.imshow(dxy, cmap='gray')
        plt.title("整体密度分布")

        plt.subplot(1, 3, 3)
        plt.imshow(image, cmap='gray')
        plt.title("均衡化之后的图像")

        plt.tight_layout()
        plt.show()

    return image

    


def main():
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

    global aH1     ## 背景为空白的像素点
    aH1 = 0.1
    global aH2     ## 背景为笔画的像素点
    aH2 = 0.3
    global aH3     ## 周边是汉字的像素点
    aH3 = 0.5
    global around  ## 多少范围内算作周边
    around = 10

    


    # 读取原始图像
    image_paths = ['deng_1.jpg', 'deng_2.jpg', 'deng_3.jpg', 'deng_4.jpg', 'deng_5.jpg', 'deng_6.jpg']
    images = []
    for p in image_paths:
        images.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    
    fig, ax = plt.subplots(len(image_paths) + 1, 2)
    aH1_ax = plt.axes([0.1, 0.1, 0.8, 0.04])
    aH1_slider = Slider(
        ax = aH1_ax,
        label="aH1",
        valmin=0,
        valmax=2,
        valinit=aH1,
    )

    
    aH2_ax = plt.axes([0.1, 0.15, 0.8, 0.04])
    aH2_slider = Slider(
        ax = aH2_ax,
        label="aH2",
        valmin=0,
        valmax=2,
        valinit=aH2,
    )
    
    aH3_ax = plt.axes([0.1, 0.2, 0.8, 0.04])
    aH3_slider = Slider(
        ax = aH3_ax,
        label="aH3",
        valmin=0,
        valmax=2,
        valinit=aH3,
    )
    
    around_ax = plt.axes([0.1, 0.25, 0.8, 0.04])
    around_slider = Slider(
        ax = around_ax,
        label="around",
        valmin=1,
        valmax=20,
        valinit=around,
    )
    # image_path = 'handwritten_chinese.jpg'
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # from normalization_memtion import remap as memtion_remap
    # image = memtion_remap(image=image)
    # remap_image = remap(image=image, show_plt=False, aH1=aH1, aH2=aH2, aH3=aH3, around=around)

    ax_map : Axes = ax[-1, 0]
    ax_map.remove()
    ax_map : Axes = ax[-1, 1]
    ax_map.remove()

    for i in range(len(image_paths)):
        ax_map : Axes = ax[i, 0]
        ax_map.imshow(images[i], cmap='gray')
        ax_map.set_title("原图像")

    ax_remap : Axes = ax[0, 1]
    # ax_remap.imshow(remap_image, cmap='gray')
    # ax_remap.set_title("均衡化之后的图像")

    def update_img():
        for i in range(len(image_paths)):
            ax_map : Axes = ax[i, 1]
            ax_map.clear()
            image_remap = remap(image=images[i], show_plt=False, aH1=aH1, aH2=aH2, aH3=aH3, around=around)
            ax_map.imshow(image_remap, cmap='gray')
            ax_map.set_title("均衡化之后的图像")

    def update_around(val):
        global around
        around = int(val)
        update_img()

    def update_aH1(val):
        global aH1
        aH1 = val
        update_img()

    def update_aH2(val):
        global aH2
        aH2 = val
        update_img()

    def update_aH3(val):
        global aH3
        aH3 = val
        update_img()

    around_slider.on_changed(update_around)
    aH1_slider.on_changed(update_aH1)
    aH2_slider.on_changed(update_aH2)
    aH3_slider.on_changed(update_aH3)
    update_img()
    plt.show()
    pass

if __name__ == '__main__':
    main()

