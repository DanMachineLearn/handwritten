# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_img_bin_dataset.py
@Time		:	2024/05/16 08:54:37
@Author		:	dan
@Description:	将pot转换成原始图像之后，保存成jpg格式，针对图片文件进行训练
'''
if __name__ == '__main__':
    import sys
    sys.path.append('.')
from alive_progress import alive_bar
import torch
import os
from torch.utils.data import IterableDataset
from algorithm.channel1_to_channel3 import Channel1ToChannel3
from algorithm.channel1_to_gabor8_1 import Channel1ToGabor8_1
from algorithm.to_one_hot_transform import ToOneHot
from algorithm.to_tensor_transform import ToTensor
from img_to_64_64_transform import ImgTo64Transform

class HandWrittenJpgDataSet(IterableDataset):
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
                 jpg_folder : str, 
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

        self.__jpg_folder = jpg_folder
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

        print("正在获取字符总数")
        ff = os.listdir(jpg_folder)
        XX = []
        yy = []
        for f in ff:
            if f.endswith('.jpg'):
                filename = os.path.basename(f)
                is_train, file_index, i, y, label = filename.split('_')
                y = int(y)
                if train and is_train == 'train':
                    XX.append(os.path.join(jpg_folder, f))
                    yy.append(y)
                elif (not train) and is_train == 'test':
                    XX.append(os.path.join(jpg_folder, f))
                    yy.append(y)
                    
            if f.endswith('labels.bin'):
                self.__labels = torch.load(os.path.join(jpg_folder, f))

        self.__char_count = len(XX)
        self.__X = XX
        self.__y = yy

    def __iter__(self):
        self.__index = 0
        return self
    
    def __len__(self):
        '''
        返回字符sample总数
        '''
        return self.__char_count
    
    
    def __getitem__(self, index):
        X = self.__X[index]
        y = self.__y[index]
        try:
            if self.__x_transforms:
                for x_transform in self.__x_transforms:
                    X = x_transform(X)
            
            if self.__y_transforms:
                for y_transform in self.__y_transforms:
                    y = y_transform(y)
        except Exception as ex:
            print(ex)
        return X, y

    def __next__(self):
        '''
        该方法在每次pytorch拿数据的时候会调用
        '''
        if self.__index >= self.__char_count:
            raise StopIteration()
        
        X, y = self.__getitem__(self.__index)
        self.__index += 1
        return X, y
        

def main():
    import time
    start_time = time.time()
    x_transforms = [ImgTo64Transform(channel_count=1, fast_handle=True), Channel1ToGabor8_1(image_only=True), ToTensor(tensor_type=torch.float32)]
    y_transforms = [ToTensor(tensor_type=torch.long)]
    dataset = HandWrittenJpgDataSet(
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        jpg_folder="work/jpg", 
        train=True)

    i = 0
    with alive_bar(len(dataset)) as bar:
        for X, y in dataset:
            bar()

    dataset = HandWrittenJpgDataSet(
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        jpg_folder="work/jpg", 
        train=False)
    with alive_bar(len(dataset)) as bar:
        for X, y in dataset:
            bar()

    len(dataset)
    print("字符总数: ", len(dataset))
    print("打开所有文件总耗时: ", '{:.2f} s'.format(time.time() - start_time))
    pass

if __name__ == '__main__':
    main()