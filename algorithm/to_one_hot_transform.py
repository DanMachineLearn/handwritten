# -*- encoding: utf-8 -*-
'''
@File		:	to_one_hot_transform.py
@Time		:	2024/05/08 12:24:42
@Author		:	dan
@Description:	将输入数据转换成one_hot格式
'''


from typing import Any, Optional
from torch.types import _dtype
from sklearn.preprocessing import OneHotEncoder
import torch
import scipy.sparse as sp
import numpy as np


class ToOneHot:


    @property
    def input_shape(self) -> int:
        '''
        输入特征的矩阵形状
        '''
        return 1
    

    def __init__(self, outer_classes : list = None) -> None:
        ''' 
        
        Parameters
        ----------
        需要转换的  tensor 类型
        
        '''
        if outer_classes:
            self.__encoder = OneHotEncoder()
            self.__encoder.fit(np.array(outer_classes).reshape((-1, 1)))
        pass

    def create_encoder(self, outer_classes : list):
        ''' 
        
        Parameters
        ----------
        
        
        Returns
        -------
        
        
        '''
        self.__encoder = OneHotEncoder()
        self.__encoder.fit(np.array(outer_classes).reshape((-1, 1)))
        pass
    
    def __call__(self, input) -> Any:
        '''
        将输入转为tensor格式 用于训练
        '''
        if isinstance(input, int):
            input = np.array((input))
            input = input.reshape((-1, 1))
        else:
            if len(input.shape) < 2:
                input = input.reshape((-1, 1))
        input : sp.csr_matrix = self.__encoder.transform(input)
        input = input.toarray()
        input = input.flatten()
        return input