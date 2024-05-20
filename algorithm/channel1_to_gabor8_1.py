# -*- encoding: utf-8 -*-
'''
@File		:	channel1_to_channel3.py
@Time		:	2024/05/16 10:05:55
@Author		:	dan
@Description:	图像的 1通道改为3通道
'''


if __name__ == '__main__':
    import sys
    sys.path.append(".")
from typing import Any, Optional
from torch.types import _dtype
import numpy as np
import torch
import cv2
from .feature_gabor_12 import get_features


class Channel1ToGabor8_1:
    '''
    1通道改为3通道
    '''

    def __init__(self, image_only = True) -> None:
        ''' 
        
        Parameters
        ----------
        
        
        '''
        self.__image_only = image_only
        
        pass

    @property
    def input_shape(self) -> int:
        '''
        输入特征的矩阵形状
        '''
        if self.__image_only:
            return (9, 64, 64)
        else:
            return 8 * 8 * 8
    
    def __call__(self, img : np.ndarray) -> Any:
        '''
        将输入转为tensor格式 用于训练
        '''
        # if img.shape[0] == 3:
        #     return img
        img = img.reshape((64, 64))
        features = get_features(img, image_only=self.__image_only, direction_count=8)
        if self.__image_only:
            img = np.stack((
                img, 
                features[0],
                features[1],
                features[2],
                features[3],
                features[4],
                features[5],
                features[6],
                features[7]))
            return img
        else:
            return features