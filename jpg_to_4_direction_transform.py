# -*- encoding: utf-8 -*-
'''
@File		:	jpg_to_4_direction_transform.py
@Time		:	2024/05/06 16:34:23
@Author		:	dan
@Description:	jpg 或 png 文件转 img图像
'''

import numpy as np
import cv2 as cv
import torch


class JpgToImgTransform:


    def __init__(self, frame_count = 8) -> None:
        ''' 
        
        Parameters
        ----------
        
        
        '''
        self.__pot_transform = ImgTo4DirectionTransform(frame_count=frame_count)
        pass


    def __call__(self, image_path) -> list:
        ''' 加载图像，获取特征值
        
        Parameters
        ----------
        
        
        Returns
        -------
        
        
        '''
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        features = self.__pot_transform.get_direction_sums(image)
        features = np.array(features).flatten()
        features = torch.tensor(features, dtype=torch.float32)
        return features