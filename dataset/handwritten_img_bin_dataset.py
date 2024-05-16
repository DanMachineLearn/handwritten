# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_img_bin_dataset.py
@Time		:	2024/05/16 08:54:37
@Author		:	dan
@Description:	将pot转换成64 * 64 的图像之后，保存成bin格式，方便读取
'''
if __name__ == '__main__':
    import sys
    sys.path.append('.')
import functools
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
from algorithm.channel1_to_channel3 import Channel1ToChannel3
from img_to_4_direction_transform import ImgTo4DirectionTransform
from img_to_64_64_transform import ImgTo64Transform
from utils.pot import Pot
from algorithm.to_one_hot_transform import ToOneHot
from algorithm.to_tensor_transform import ToTensor
import utils.my_wget as my_wget




class HandWrittenBinDataSet(IterableDataset):
    '''
    64 * 64 图像的bin序列
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
    def input_shape(self) -> int:
        '''
        x 的特征数量，取决于转换器
        '''
        return (1, 64, 64)

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
                 bin_folder : str, 
                 train : True,
                 x_transforms : list = None,
                 y_transforms : list = None) -> None:
        ''' 
        Parameters
        ----------
        bin_folder 存放.bin 的文件夹

        outter_labels 测试集的labels列表需要和训练集的相同，所以要从训练集那边传递过来

        train : True, 是否加载训练数据，False为加载测试数据
        '''

        self.__bin_folder = bin_folder
        self.__train = train

        
        ## 统计所有的图像数据、字符数量、所有字符标签
        self.__char_count = 0
        self.__labels = []
        self.__file_count = 0

        self.__x_transforms = x_transforms
        self.__y_transforms = y_transforms

        ## 如果存在one hot 的转换器，需要传入总类别
        if self.__y_transforms is not None:
            for y_trans in self.__y_transforms:
                if isinstance(y_trans, ToOneHot):
                    y_trans.create_encoder(list(range(0, len(self.labels))))

        self.__y_bin_files = []
        self.__X_bin_files = []
        char_count = 0
        print("正在获取字符总数")
        ff = os.listdir(bin_folder)
        files = []
        for f in ff:
            if f.endswith('.bin'):
                files.append(f)
        with alive_bar(len(files)) as bar:
            for file_name in files:

                if file_name.endswith('labels.bin'):
                    self.__labels = torch.load(os.path.join(bin_folder, file_name))

                if file_name.endswith('y.bin'):
                    if train and file_name.endswith('test.y.bin'):
                        bar()
                        continue;

                    if not train and file_name.endswith('train.y.bin'):
                        bar()
                        continue;   
                    
                    file_path = os.path.join(bin_folder, file_name)
                    self.__file_count += 1
                    self.__y_bin_files.append(file_path)
                    X = torch.load(file_path)
                    char_count += len(X)

                elif file_name.endswith('x.bin'):
                    if train and file_name.endswith('test.x.bin'):
                        bar()
                        continue;

                    if not train and file_name.endswith('train.x.bin'):
                        bar()
                        continue;  
                    self.__X_bin_files.append(os.path.join(bin_folder, file_name))

                bar()
        

        def my_compare(x0, x1):
            file0 : str = os.path.basename(x0)
            file1 : str = os.path.basename(x1)
            index0 = file0[0 : file0.index('_')]
            index1 = file1[0 : file1.index('_')]
            index0 = int(index0)
            index1 = int(index1)
            if index0 > index1:
                return -1
            elif index1 < index0:
                return 1
            return 0

        self.__X_bin_files = sorted(self.__X_bin_files, key=functools.cmp_to_key(my_compare))
        self.__y_bin_files = sorted(self.__y_bin_files, key=functools.cmp_to_key(my_compare))

        print("已加载")
        print(",".join(self.__y_bin_files))

        self.__char_count = char_count
        self.__X = []
        self.__y = []
        self.__index = 0
        self.__current_bin_index = 0

    def __iter__(self):
        self.__X = []
        self.__y = []
        self.__index = 0
        self.__current_bin_index = 0
        return self
    
    def __len__(self):
        '''
        返回字符sample总数
        '''
        return self.__char_count
    

    # def __getitem__(self, index):
    #     ''' 
    #     这个方法需要加载所有字符到内存中  
    #     '''
    #     return self.__X[index], self.__y[index]
    def next_bin(self):
        ''' 
        加载下一个bin文件，如果没有，则抛出异常
        '''
        if self.__current_bin_index >= len(self.__y_bin_files):
            raise StopIteration()
        
        y_bin = self.__y_bin_files[self.__current_bin_index]
        X_bin = self.__X_bin_files[self.__current_bin_index]

        self.__current_bin_index += 1
        self.__y = torch.load(y_bin)
        self.__X = torch.load(X_bin)
        self.__index = 0

    def __next__(self):
        '''
        该方法在每次pytorch拿数据的时候会调用
        '''
        if self.__index >= len(self.__X):
            self.next_bin()

        if self.__index >= len(self.__X):
            print(f"数据错误，输出当前的变量", 
                  f"\nself.__current_bin_index = {self.__current_bin_index}", 
                  f"\nbin_y = {self.__y_bin_files[self.__current_bin_index - 1]}",
                  f"\nbin_x = {self.__X_bin_files[self.__current_bin_index - 1]}",
                  f"\nself.__index = {self.__index}", 
                  f"\nlen(self.__X) = {len(self.__X)}", 
                  f"\nlen(self.__y) = {len(self.__y)}")
        X = self.__X[self.__index]

        if self.__index >= len(self.__y):
            print(f"数据错误，输出当前的变量", 
                  f"\nself.__current_bin_index = {self.__current_bin_index}", 
                  f"\nbin_y = {self.__y_bin_files[self.__current_bin_index - 1]}",
                  f"\nbin_x = {self.__X_bin_files[self.__current_bin_index - 1]}",
                  f"\nself.__index = {self.__index}", 
                  f"\nlen(self.__X) = {len(self.__X)}", 
                  f"\nlen(self.__y) = {len(self.__y)}")
        y = self.__y[self.__index]
        try:
            if self.__x_transforms:
                for x_transform in self.__x_transforms:
                    X = x_transform(X)
            
            if self.__y_transforms:
                for y_transform in self.__y_transforms:
                    y = y_transform(y)
            self.__index += 1
        except Exception as ex:
            print(ex)
            self.__index += 1
            return self.__next__()
        return X, y

def main():
    pot_folder = []
    # pot_folder.append("work/PotSimple")
    # pot_folder.append("work/PotSimpleTest")
    # pot_folder.append("work/PotTest")
    pot_folder.append("work/Bin")
    # pot_folder.append("work/data/HWDB_pot/PotTest")
    # pot_folder.append("work/data/HWDB_pot/PotTrain")


    import time
    start_time = time.time()
    x_transforms = [Channel1ToChannel3(), ToTensor(tensor_type=torch.float32)]
    y_transforms = [ToTensor(tensor_type=torch.long)]
    dataset = HandWrittenBinDataSet(
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        bin_folder="work/Bin", 
        train=True)

    with alive_bar(len(dataset)) as bar:
        for X, y in dataset:
            bar()

    dataset = HandWrittenBinDataSet(
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        bin_folder="work/Bin", 
        train=False)
    with alive_bar(len(dataset)) as bar:
        for X, y in dataset:
            bar()

    len(dataset)
    print("字符总数: ", len(dataset))
    print("打开pot文件数量: ", dataset.file_count)
    print("打开所有pot文件总耗时: ", '{:.2f} s'.format(time.time() - start_time))
    pass

if __name__ == '__main__':
    main()