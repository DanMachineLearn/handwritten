# -*- encoding: utf-8 -*-
'''
@File		:	channel1_to_channel3.py
@Time		:	2024/05/16 10:05:55
@Author		:	dan
@Description:	图像的 1通道改为3通道
'''



from typing import Any, Optional
from torch.types import _dtype
import numpy as np
import torch
import cv2


class Channel1ToChannel3:
    '''
    1通道改为3通道
    '''

    @property
    def input_shape(self) -> int:
        '''
        输入特征的矩阵形状
        '''
        return (3, 64, 64)
    

    def __init__(self) -> None:
        pass
    
    def __call__(self, img : np.ndarray) -> Any:
        '''
        将输入转为tensor格式 用于训练
        '''
        if img.shape[0] == 3:
            return img
        img = img.reshape((64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.reshape((3, 64, 64))
        return img