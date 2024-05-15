# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_features.py
@Time		:	2024/05/06 13:38:08
@Author		:	dan
@Description:	直接读取pot文件，转成方向线素
'''
if __name__ == '__main__':
    import sys
    sys.path.append('.')
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
from img_to_4_direction_transform import ImgTo4DirectionTransform
from img_to_64_64_transform import ImgTo64Transform
from utils.pot import Pot
from algorithm.to_one_hot_transform import ToOneHot
from algorithm.to_tensor_transform import ToTensor
import utils.my_wget as my_wget




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
                 need_label=False,
                 outter_labels:list[str] = None, 
                 cache_csv_file : str = None,
                 load_all_on_init = False, 
                 x_transforms : list = None,
                 y_transforms : list = None) -> None:
        ''' 
        Parameters
        ----------
        pot_folders 存放.pot 的文件夹列表

        frame_count 表示提取方向线素之后，每行和每列分别平均划分多少个方格
        
        outter_labels 测试集的labels列表需要和训练集的相同，所以要从训练集那边传递过来
        
        cache_csv_file 第二次开始训练的时候，直接获取缓存中的数据训练

        need_label=True, 是否在输出的y里面添加label

        load_all_on_init = False 是否一次将所有pot数据读取到内存，方便训练
        '''

        self.__need_label = need_label
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


        ## 一次读取所有的pot文件到内存，加快训练速度
        self.__load_all_on_init = load_all_on_init
        if load_all_on_init:
            self.__X = []
            self.__y = []
            for X, y in self:
                self.__X.append(X)
                self.__y.append(y)
        else:
            self.__X = None
            self.__y = None

    def __iter__(self):
        self.__current_pot_index = 0
        for p in self.__pots:
            p.close()
        return self
    
    def __len__(self):
        '''
        返回字符sample总数
        '''
        if self.__load_all_on_init:
            return len(self.__X)
        return self.__char_count
    

    def __getitem__(self, index):
        ''' 
        这个方法需要加载所有字符到内存中  
        '''
        return self.__X[index], self.__y[index]


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
        
        if not self.__need_label:
            y = self.__labels.index(y)
            if self.__y_transforms:
                for y_transform in self.__y_transforms:
                    y = y_transform(y)

        else:
            index = self.__labels.index(y)
            y = (index, y)



        return X, y

def main():
    pot_folder = []
    pot_folder.append("work/PotSimple")
    pot_folder.append("work/PotSimpleTest")
    pot_folder.append("work/PotTest")
    pot_folder.append("work/PotTrain")
    # pot_folder.append("work/data/HWDB_pot/PotTest")
    # pot_folder.append("work/data/HWDB_pot/PotTrain")

    import time
    start_time = time.time()
    dataset = HandWrittenDataSet(
        pot_folders=pot_folder, 
        load_all_on_init=True,
        x_transforms=[ImgTo64Transform()],
        y_transforms=[ToTensor(tensor_type=torch.long)])
    
    for X, y in dataset:
        # cv.imshow("X", X)
        # if cv.waitKey(-1) == ord('q'):
        #     break;
        pass

    len(dataset)
    print("字符总数: ", len(dataset))
    print("打开pot文件数量: ", dataset.file_count)
    print("打开所有pot文件总耗时: ", '{:.2f} s'.format(time.time() - start_time))
    pass

if __name__ == '__main__':
    main()