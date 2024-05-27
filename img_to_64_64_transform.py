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
from algorithm.normalization_nonlinear_mean import remap
from algorithm.normalization_memtion import remap as memtion_remap


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
    def __init__(self, need_dilate : bool = True, need_reshape = False, show_plt = False, channel_count=1, fast_handle = False) -> None:
        ''' 
        Parameters
        ----------
        need_dilate : bool = True, 是否需要膨胀处理，如果原始图像是大于64 * 64，则为了避免在图像缩小的时候信息丢失，在缩小之前要进行膨胀处理。
        
        need_reshape : bool = False, 是否需要修改图像的 shape

        channel_count=1 色彩通道数要么1，要么3

        fast_handle=False 使用较快速的归一化算法（主要是因为自己的机器太慢）

        '''
        self.__need_dilate = need_dilate
        self.__need_reshape = need_reshape
        self.__show_plt = show_plt
        self.__channel_count = channel_count
        self.__fast_handle = fast_handle
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
        if image.shape == (1, 64, 64):
            image = image.reshape((64, 64))


        if self.__need_dilate:
            # 膨胀和腐蚀都是对于白色像素而言的，所以对于黑色的膨胀，则需要进行白色的腐蚀。
            kernel = np.ones((5, 5), dtype=np.uint8) # 卷积核变为4*4
            image = cv.erode(image, kernel, iterations=1)

        if self.__fast_handle:
            resized_image = memtion_remap(image=image)
        else:
            resized_image = remap(image=image, target_size=(64, 64), show_plt=False)

        if self.__show_plt:
            cv.imshow("image", resized_image)
            cv.waitKey(-1)

        # 3. 调整图像大小
        if self.__need_reshape:
            resized_image = resized_image.reshape((1, 64, 64))

        if self.__channel_count == 3:
            resized_image = cv.cvtColor(resized_image, cv.COLOR_GRAY2RGB)
            resized_image = resized_image.reshape((3, 64, 64))

        if self.__channel_count == 4:
            resized_image = cv.cvtColor(resized_image, cv.COLOR_GRAY2RGBA)
            resized_image = resized_image.reshape((4, 64, 64))

        if self.__channel_count == 1:
            resized_image = resized_image.reshape((1, 64, 64))

        return resized_image




def main():
    test = "deng.jpg"
    transform = ImgTo64Transform()
    img = transform(test)
    img = img.reshape((64, 64))
    cv.imshow("img", img)
    cv.waitKey(-1)
    pass

if __name__ == '__main__':
    main()