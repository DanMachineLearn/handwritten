# -*- encoding: utf-8 -*-
'''
@File		:	to_tensor_transform.py
@Time		:	2024/05/08 10:01:29
@Author		:	dan
@Description:	直接转tensor
'''



from typing import Any, Optional
from torch.types import _dtype

import torch


class ToTensor:


    @property
    def input_shape(self) -> int:
        '''
        输入特征的矩阵形状
        '''
        return 1
    

    def __init__(self, tensor_type : Optional[_dtype] = torch.long) -> None:
        ''' 
        
        Parameters
        ----------
        需要转换的  tensor 类型
        
        '''
        self.__tensor_type = tensor_type
        pass
    
    def __call__(self, input) -> Any:
        '''
        将输入转为tensor格式 用于训练
        '''
        return torch.tensor(input, dtype=self.__tensor_type)