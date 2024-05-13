# -*- encoding: utf-8 -*-
'''
@File		:	img_to_64_64_transform.py
@Time		:	2024/05/08 10:23:47
@Author		:	dan
@Description:	将图像正则化为 64*64 图像的转换器
'''

from os import PathLike
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import torch


class ImgTo64Transform:

    @property
    def input_shape(self) -> int:
        '''
        转换之后特征的矩阵
        '''
        return np.array([1, 64, 64])
    
    '''
    pot数据转64*64的图像
    '''
    def __init__(self, need_dilate : bool = True) -> None:
        ''' 
        Parameters
        ----------
        need_dilate : bool = True, 是否需要膨胀处理，如果原始图像是大于64 * 64，则为了避免在图像缩小的时候信息丢失，在缩小之前要进行膨胀处理。
        
        '''
        self.__need_dilate = need_dilate
        pass

    def __call__(self, image : np.ndarray | str):




        ''' 
        将图像转为64 64 大小的图
        Parameters
        ----------
        Returns
        -------
        
        
        '''

        if isinstance(image, str):
            image = cv.imread(image, cv.IMREAD_GRAYSCALE)

        # 膨胀图像
        if self.__need_dilate:
            # 膨胀和腐蚀都是对于白色像素而言的，所以对于黑色的膨胀，则需要进行白色的腐蚀。
            kernel = np.ones((5, 5), dtype=np.uint8) # 卷积核变为4*4
            image = cv.erode(image, kernel, iterations=1)


            # # 所以也可以将白色和黑色进行转换，通过 dilate 膨胀之后，再切换回来。
            # image = 255 - image
            # kernel = np.ones((5, 5), dtype=np.uint8) # 卷积核变为4*4
            # image = cv2.dilate(image, kernel, iterations=1)
            # image = 255 - image

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


        
        # 计算图像的重心
        # 使用图像的矩计算质心
        moments = cv.moments(image)
        center_x = int(moments['m10'] / moments['m00'])  # x 轴重心
        center_y = int(moments['m01'] / moments['m00'])  # y 轴重心
        left_top = image[0 : center_y, 0 : center_x]
        left_bottom = image[center_y : image.shape[0], 0 : center_x]
        right_top = image[0 : center_y, center_x : image.shape[1]]
        right_bottom = image[center_y : image.shape[0], center_x : image.shape[1]]

        left_top = cv.resize(left_top, (32, 32), interpolation=cv.INTER_LINEAR)
        left_bottom = cv.resize(left_bottom, (32, 32), interpolation=cv.INTER_LINEAR)
        right_top = cv.resize(right_top, (32, 32), interpolation=cv.INTER_LINEAR)
        right_bottom = cv.resize(right_bottom, (32, 32), interpolation=cv.INTER_LINEAR)

        resized_image = np.zeros((64, 64))
        resized_image[0 : 32, 0 : 32] = left_top
        resized_image[32 : 64, 0 : 32] = left_bottom
        resized_image[0 : 32, 32 : 64] = right_top
        resized_image[32 : 64, 32 : 64] = right_bottom
        resized_image = resized_image.astype(np.uint8)

        ## 使用直方图均衡化
        resized_image = cv.equalizeHist(resized_image)

        # 3. 调整图像大小
        # 将图像调整为 65x65
        # TODO 此处需要补充 根据质点调整大小的算法
        # resized_image = cv.resize(image, (64, 64), interpolation=cv.INTER_LINEAR)
        resized_image = resized_image.reshape((1, 64, 64))
        # resized_image = torch.tensor(resized_image, dtype=torch.uint8)
        return resized_image




def main():
    test = "handwritten_chinese.jpg"
    transform = ImgTo64Transform(need_dilate=True)
    img = transform(test)
    img = img.reshape((64, 64))
    cv.imshow("img", img)
    cv.waitKey(-1)
    pass

if __name__ == '__main__':
    main()