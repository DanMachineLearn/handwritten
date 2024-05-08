# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_features.py
@Time		:	2024/05/06 13:38:08
@Author		:	dan
@Description:	直接读取grad-12的 csv 文件
'''
from io import BufferedReader
from sklearn.model_selection import train_test_split
import torch
from matplotlib import pyplot as plt
import numpy as np
import struct
from torch.utils.data import Dataset
import glob
import os
import pandas as pd
from torch.utils.data import IterableDataset


# 创建一个自定义数据集
class SimpleDataset(Dataset):

    @property
    def file_count(self) -> int:
        '''
        文件数量
        '''        
        return self.__file_count
    
    @property
    def feature_count(self) -> int:
        '''
        特征数量
        '''
        return 12
    
    def __init__(self, X, y, file_count, labels):
        # 创建一个数据集，包含size个随机数和相应的标签
        self.__X = X
        self.__y = y
        self.__y = self.__y - 1
        self.__X = self.__X.to_numpy()
        self.__file_count = file_count
        self.__labels = labels


    @property
    def labels(self) -> list[str]:
        '''
        字符种类
        '''
        return self.__labels

    @property
    def input_feature(self) -> int:
        '''
        输入特征的数量
        '''
        return self.__input_feature


    @property
    def out_classes(self) -> int:
        '''
        输出类别数量
        '''
        return self.__out_classes

    def __len__(self):
        return self.__X.shape[0]

    def __getitem__(self, idx):
        # 返回数据和标签

        y = self.__y[idx]
        # y = y[0]

        ## 不需要 one hot 作为结果准确率反而变高
        # y = y.reshape((1, 1))
        # y = self.__one_hot_encoder.transform(y)[0]

        X_train = self.__X[idx]

        return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    


class HandWrittenGrad12CsvDataSet:
    '''
    中科院pot文件读取的iterator类
    '''

    @property
    def train_dataset(self) -> Dataset:
        '''
        训练用的数据集
        '''
        return self.__train_dataset

    @property
    def test_dataset(self) -> Dataset:
        '''
        测试用的数据集
        '''
        return self.__test_dataset

    @property
    def feature_count(self) -> int:
        '''
        x 的特征数量，取决于转换器
        '''
        return 12

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
                 csv_folders : list[str], 
                 test_size = 200) -> None:
        ''' 
        Parameters
        ----------
        csv_folders 存放.csv 的文件夹列表

        test_size = 200, 测试组大小

        outter_labels 测试集的labels列表需要和训练集的相同，所以要从训练集那边传递过来
        
        '''

        self.__csv_folders = csv_folders
        self.__current_csv_index = 0
        
        ## 统计所有的图像数据、字符数量、所有字符标签
        self.__pots = []
        self.__char_count = 0
        self.__labels = []
        self.__file_count = 0
        __X = None
        __y = None
        for pot_folder in csv_folders:
            ## 遍历csv文件
            for file_name in os.listdir(pot_folder):
                if file_name.endswith('.csv'):
                    self.__file_count += 1
                    file_path = os.path.join(pot_folder, file_name)
                    X, y = self.caculate_csv_file(file_path)
                    if not __X :
                        __X = X
                        __y = y
                    else:
                        __X.add(X)
                        __y.add(y)
        label_map = {}
        __y = __y.to_numpy()
        for y_item in __y:
            label_map[y_item[0]] = y_item[1]
        __y = __y[:, 0]
        X_train , X_test, y_train, y_test = train_test_split(__X, __y, test_size=test_size, random_state=42, shuffle=True)
        label_index = sorted(set(y_train))
        self.__labels = []
        for i in label_index:
            self.__labels.append(label_map[i])
        
        self.__train_dataset = SimpleDataset(X_train, y_train, file_count=self.__file_count, labels=self.__labels)
        self.__test_dataset = SimpleDataset(X_test, y_test, 0, self.__labels)
    
    # def __len__(self):
    #     '''
    #     返回字符sample总数
    #     '''
    #     return len(self.__y)
    

    # def __getitem__(self, index):
    #     ''' 
    #     这个方法需要加载所有字符到内存中，暂时不支持        
    #     '''
        
    #     return torch.tensor(self.__X[index], dtype=torch.float32), torch.tensor(self.__y[index, 0], dtype=torch.long)
    

    def caculate_csv_file(self, csv_file):
        ''' 
        获取csv文件里的x 和 y
        Parameters
        ----------
        
        
        Returns
        -------
        list, list
        
        '''
        # 从 CSV 文件加载数据
        data = pd.read_csv(csv_file)
        X = data.iloc[:, 1:-3]  # 特征
        y = data.iloc[:, -2:]  # 目标标签
        return X, y
            

def main():
    csv_folders = []
    csv_folders.append("D:\\Gitee\\python-learn\\work\\data\\HWDB_pot\\Grad-12-csv-simple")
    # csv_folders.append("D:\\Gitee\\python-learn\\work\\data\\HWDB_pot\\PotTest")
    # pot_folder.append("D:\\Gitee\\python-learn\\work\\data\\HWDB_pot\\PotTrain")

    import time
    start_time = time.time()
    dataset = HandWrittenGrad12CsvDataSet(csv_folders=csv_folders)
    print("字符总数: ", len(dataset.train_dataset))
    dataset.train_dataset[0]
    print("打开csv文件数量: ", dataset.file_count)
    print("打开所有csv文件总耗时: ", '{:.2f} s'.format(time.time() - start_time))
    pass

if __name__ == '__main__':
    main()