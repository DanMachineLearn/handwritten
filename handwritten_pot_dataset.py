# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_features.py
@Time		:	2024/05/06 13:38:08
@Author		:	dan
@Description:	直接读取pot文件，转成方向线素
'''
from io import BufferedReader
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
from img_to_4_direction_transform import ImgTo4DirectionTransform
from img_to_64_64_transform import ImgTo64Transform
from pot import Pot
from to_one_hot_transform import ToOneHot
from to_tensor_transform import ToTensor




class HandWrittenDataSet(IterableDataset):
    '''
    中科院pot文件读取的iterator类
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
        return self.__feature_count

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
        return self.__file_count

    def __init__(self, 
                 pot_folders : list[str], 
                 frame_count=8, 
                 outter_labels:list[str] = None, 
                 cache_csv_file : str = None,
                 x_transforms : list = None,
                 y_transforms : list = None) -> None:
        ''' 
        Parameters
        ----------
        pot_folders 存放.pot 的文件夹列表

        frame_count 表示提取方向线素之后，每行和每列分别平均划分多少个方格
        
        outter_labels 测试集的labels列表需要和训练集的相同，所以要从训练集那边传递过来
        
        cache_csv_file 第二次开始训练的时候，直接获取缓存中的数据训练
        '''

        self.pot_folders = pot_folders
        self.__current_pot_index = 0
        if cache_csv_file is None:
            self.__cache_csv_file = pot_folders[0] + "/data_csv.csv"
        else:
            self.__cache_csv_file = cache_csv_file
        
        ## 统计所有的图像数据、字符数量、所有字符标签
        self.__pots = []
        self.__char_count = 0
        self.__labels = []
        self.__file_count = 0
        for pot_folder in pot_folders:
            p = Pot(pot_folder=pot_folder, chineses_only=True)
            self.__char_count += p.char_count
            self.__pots.append(p)
            self.__labels.extend(p.labels)
            self.__file_count += p.file_count;
        if outter_labels:
            self.__labels = outter_labels
        else:
            self.__labels = sorted(set(self.__labels))
        if x_transforms:
            self.__x_transforms = x_transforms
        else:
            self.__x_transforms = [ImgTo4DirectionTransform(frame_count)]
        if y_transforms:
            self.__y_transforms = y_transforms
        else:
            self.__y_transforms = [ToTensor]

        # x 的特征数量，取决于最后一个转换器
        self.__input_shape = self.__x_transforms[-1].input_shape

        ## 如果存在one hot 的转换器，需要传入总类别
        for y_trans in self.__y_transforms:
            if isinstance(y_trans, ToOneHot):
                y_trans.create_encoder(list(range(0, len(self.labels))))

    def __iter__(self):
        self.__current_pot_index = 0
        for p in self.__pots:
            p.close()
        return self
    
    def __len__(self):
        '''
        返回字符sample总数
        '''
        return self.__char_count
    

    def __getitem__(self, index):
        ''' 
        这个方法需要加载所有字符到内存中，暂时不支持        
        '''
        
        pass


    def __next__(self):
        '''
        该方法在每次pytorch拿数据的时候会调用
        '''
        if self.__current_pot_index >= len(self.__pots):
            # 已经没有文件了
            raise StopIteration()
        p : Pot = self.__pots[self.__current_pot_index]
        X, y = p.next()
        if X is None and y is None:
            self.__current_pot_index += 1
            return self.__next__()

        if self.__x_transforms:
            for x_transform in self.__x_transforms:
                X = x_transform(X)

        if not self.__labels.__contains__(y):
            return self.__next__()
        
        y = self.__labels.index(y)

        if self.__y_transforms:
            for y_transform in self.__y_transforms:
                y = y_transform(y)

        return X, y
            

def main():
    pot_folder = []
    pot_folder.append("work/data/HWDB_pot/potSimple")
    # pot_folder.append("work/data/HWDB_pot/PotTest")
    # pot_folder.append("work/data/HWDB_pot/PotTrain")

    import time
    start_time = time.time()
    dataset = HandWrittenDataSet(
        pot_folders=pot_folder, 
        x_transforms=[ImgTo64Transform(need_dilate=False)],
        y_transforms=[ToTensor(tensor_type=torch.long)])
    
    for X, y in dataset:
        cv.imshow("X", X)
        if cv.waitKey(-1) == ord('q'):
            break;

    len(dataset)
    print("字符总数: ", len(dataset))
    print("打开pot文件数量: ", dataset.file_count)
    print("打开所有pot文件总耗时: ", '{:.2f} s'.format(time.time() - start_time))
    pass

if __name__ == '__main__':
    main()