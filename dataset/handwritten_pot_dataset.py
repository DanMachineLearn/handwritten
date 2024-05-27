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
    def X(self):
        return self.__X
    
    @property
    def y(self):
        return self.__y
    
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

        load_from_bin = True 是否加载work目录下的二进制持久化文件
        '''

        self.__need_label = need_label
        self.pot_folders = pot_folders
        self.__current_pot_index = 0

        
        ## 统计所有的图像数据、字符数量、所有字符标签
        self.__pots = []
        self.__char_count = 0
        self.__labels = []
        self.__file_count = 0

        self.__x_transforms = x_transforms
        self.__y_transforms = y_transforms

        ## 如果存在one hot 的转换器，需要传入总类别
        for y_trans in self.__y_transforms:
            if isinstance(y_trans, ToOneHot):
                y_trans.create_encoder(list(range(0, len(self.labels))))

        print("正在获取字符总数")
        with alive_bar(len(pot_folders)) as bar:
            for pot_folder in pot_folders:
                p = Pot(pot_folder=pot_folder, chineses_only=True)
                self.__char_count += p.char_count
                self.__pots.append(p)
                self.__labels.extend(p.labels)
                self.__file_count += p.file_count;
                bar()
        if outter_labels:
            self.__labels = outter_labels
        else:
            self.__labels = sorted(set(self.__labels))
        
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
        return self.__char_count


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

        try:
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
        except:
            return self.__next__()
        return X, y


def export(train = True, out_labels : list[str] = None, chars_only : list[str] = None):
    ''' 
    导出pot成bin文件

    chars_only = None 只导出特定的字符
    '''
    pot_folder = []
    if train:
        pot_folder.append(f"work/Pot1.0/Pot1.0Train.zip_out")
        pot_folder.append(f"work/Pot1.1/Pot1.1Train.zip_out")
        # pot_folder.append(f"work/Pot1.2/Pot1.2Train.zip_out")
    else:
        pot_folder.append(f"work/Pot1.0/Pot1.0Test.zip_out")
        pot_folder.append(f"work/Pot1.1/Pot1.1Test.zip_out")
        # pot_folder.append(f"work/Pot1.2/Pot1.2Test.zip_out")

    import time
    start_time = time.time()
    # x_transforms = [ImgTo64Transform(channel_count=1, fast_handle=True)]
    x_transforms = []
    y_transforms = []
    dataset = HandWrittenDataSet(
        pot_folders=pot_folder, 
        outter_labels=out_labels,
        x_transforms=x_transforms,
        y_transforms=y_transforms)
    
    i = 0
    XX = []
    yy = []
    file_index = 1
    if train:
        torch.save(dataset.labels, f'work/Bin/labels.bin')
    max_width = 0
    max_height = 0
    is_train = "train" if train else "test"
    with alive_bar(len(dataset)) as bar:
        for X, y in dataset:
            if chars_only is not None:
                char = dataset.labels[y]
                if not chars_only.__contains__(char):
                    bar()
                    continue
            XX.append(X)
            yy.append(y)
            i += 1
            if i % 10000 == 0:
                for i in range(len(XX)):
                    X = XX[i]
                    y = yy[i]
                    label = dataset.labels[y]
                    cv.imwrite(f"work/jpg/{is_train}_{file_index}_{i}_{y}_{label}.jpg", X)
                XX = []
                yy = []
                file_index += 1
            bar()
    for i in range(len(XX)):
        X = XX[i]
        y = yy[i]
        label = dataset.labels[y]
        cv.imwrite(f"work/jpg/{is_train}_{file_index}_{i}_{y}_{label}.jpg", X)
    XX = []
    yy = []


    len(dataset)
    print("字符总数: ", len(dataset))
    print("打开pot文件数量: ", dataset.file_count)
    print("打开所有pot文件总耗时: ", '{:.2f} s'.format(time.time() - start_time))
    return dataset

def main():
    dataset = export(train=True)
    export(train=False, out_labels=dataset.labels)
    pass

if __name__ == '__main__':
    main()