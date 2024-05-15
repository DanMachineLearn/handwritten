# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_dataset_csv_grad_12.py
@Time		:	2024/05/14 08:57:03
@Author		:	dan
@Description:	已经转为grad-12格式的csv数据源
'''
if __name__ == '__main__':
    import sys
    sys.path.append(".")
from io import BufferedReader
from alive_progress import alive_bar
import torch
from matplotlib import pyplot as plt
import numpy as np
import struct
import cv2 as cv
from torch.utils.data import Dataset
import glob
import os
import pandas as pd
from torch.utils.data import IterableDataset

from to_tensor_transform import ToTensor


class HandWrittenDatasetCsvGrad12(IterableDataset):
    '''
    已经转为grad-12格式的csv数据源
    '''


    @property
    def x_transforms(self) -> any:
        '''
        图像提取特征的工具
        '''
        return self.__x_transforms

    @property
    def feature_count(self) -> int:
        '''
        x 的特征数量，取决于转换器
        '''
        return 8 * 8 * 12

    @property
    def labels(self) -> list[str]:
        '''
        字符种类
        '''
        return self.__labels

    @property
    def file_count(self) -> int:
        '''
        文件数量
        '''
        return 1



    def __init__(self, 
                 data_csv_path : str, 
                 label_csv_path : str, 
                 batch_read_size = None, 
                 max_length = None,
                 start_index = 0,
                 x_transforms : list = None,
                 y_transforms : list = None) -> None:
        ''' 
        
        Parameters
        ----------
        csv_path : str, csv文件位置
        
        batch_read_size = None 每次读取的条数

        test_size = 500 测试集的数量

        max_length = 200, 最大读取长度

        start_index = 200, 开始读取位置
        '''
        
        self.__data_csv_path = data_csv_path
        self.__batch_read_size = batch_read_size
        self.__read_index = start_index
        self.__read_length = start_index
        self.__start_index = start_index
        self.__max_length = max_length


        # 读取最后一行获取行数
        data_frame = pd.read_csv(data_csv_path)
        bottom : pd.Series = data_frame.tail(1)
        bottom = bottom['id']
        all_chat_count = int(bottom.iloc[0]) + 1
        if max_length is not None:
            self.__char_count = min(all_chat_count - start_index, max_length)
        else:
            self.__char_count = all_chat_count - start_index

        # # 读取第一批数据
        # data_frame = pd.read_csv(data_csv_path, nrows=batch_read_size, skiprows=0)
        # self.__X = data_frame['XX']
        # self.__y = data_frame['yy']
        # self.__read_length += len(self.__X)

        self.__x_transforms = x_transforms
        self.__y_transforms = y_transforms

        ## 读取所有标签
        labels_data_frame = pd.read_csv(label_csv_path)
        self.__labels = labels_data_frame['labels']

        self.read_next()
        pass


    def read_next(self):
        ''' 
        读取下一批
        '''
        if self.__read_length == 0:
            data_frame = pd.read_csv(self.__data_csv_path, 
                                 nrows=self.__batch_read_size, 
                                 skiprows=self.__read_length)
        else:
            data_frame = pd.read_csv(self.__data_csv_path, 
                                 names=["id", "XX", "yy", "labels"],
                                 header=0,
                                 nrows=self.__batch_read_size, 
                                 skiprows=self.__read_length)
            
        if not data_frame.columns.__contains__('XX'):
            raise StopIteration()
        self.__X = data_frame['XX']
        self.__y = data_frame['yy']
        more = len(self.__X)
        if more == 0:
            raise StopIteration()
        self.__read_length += more
        pass

    def __next__(self):
        '''
        获取下一个数据
        '''
        # if self.__read_index >= self.__read_length:
        #     self.read_next()
        if self.__max_length is not None:
            if self.__read_index >= self.__max_length + self.__start_index:
                raise StopIteration()
        else:
            if self.__read_index >= self.__char_count + self.__start_index:
                raise StopIteration()
        index = self.__read_index - self.__start_index
        if self.__batch_read_size is not None:
            index = index % self.__batch_read_size
        X : str = self.__X[index]
        X = X.strip('[')
        X = X.strip(']')
        X = X.split(',')
        X = np.array(X, dtype=np.float32)

        y = self.__y[index]
        self.__read_index += 1


        if self.__x_transforms:
            for x_transform in self.__x_transforms:
                X = x_transform(X)

        if self.__y_transforms:
            for y_transform in self.__y_transforms:
                y = y_transform(y)
        return X, y

    def __iter__(self) :
        self.__read_index = self.__start_index
        # self.__read_length = self.__start_index
        return self

    def __len__(self):
        return self.__char_count


def main():
    data_csv_path = 'work/grad_12.csv'
    labels_csv_path = 'work/grad_12.labels.csv'
    x_transforms = [ToTensor(tensor_type=torch.float32)]
    y_transforms = [ToTensor(tensor_type=torch.long)]
    dataset = HandWrittenDatasetCsvGrad12(
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        start_index=30,
        data_csv_path=data_csv_path, 
        label_csv_path=labels_csv_path)
    for X, y in dataset:
        print(y)
    pass

if __name__ == '__main__':
    main()