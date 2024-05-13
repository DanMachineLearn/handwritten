

# -*- encoding: utf-8 -*-
'''
@File		:	pot_to_4direction_transform.py
@Time		:	2024/05/06 14:59:10
@Author		:	dan
@Description:	numpy 图像转 4方向线素的转换器
'''

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import torch
from algorithm.feature_grad_12 import get_features


class ImgToGrad12Transform:

    INPUT_FEATURES = 8 * 8 * 12


    @property
    def feature_count(self) -> int:
        '''
        特征数量
        '''
        return self.__feature_count

    '''
    pot数据转grad-12线素数据
    '''
    def __init__(self, show_plt : bool = False) -> None:
        ''' 
        show_plt 是否显示plt图像，用于测试
        '''
        self.__feature_count = 8 * 8 * 12
        self.__show_plt = show_plt
        pass


    def __call__(self, img : np.ndarray):
        features = get_features(img)
        return features



def main():
    image_path = "handwritten_chinese.jpg"
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    transform = ImgToGrad12Transform(show_plt=True)
    features = transform(image)
    print(features)


if __name__ == '__main__':
    main()
