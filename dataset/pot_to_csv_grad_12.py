# -*- encoding: utf-8 -*-
'''
@File		:	pot_to_csv_grad_12.py
@Time		:	2024/05/13 15:55:44
@Author		:	dan
@Description:	数据预处理，将pot文件转为grad-12 特征 csv文件
'''

import sys
sys.path.append('.')
import torch
from handwritten_pot_dataset import HandWrittenDataSet
import os
from torch.utils.data import DataLoader
from alive_progress import alive_bar

from img_to_64_64_transform import ImgTo64Transform
from img_to_grad_12_transform import ImgToGrad12Transform
from to_tensor_transform import ToTensor

import pandas as pd

def main():

    DATA_SET_FOLDER = os.environ["DATA_SET_FOLDER"] if os.environ.__contains__("DATA_SET_FOLDER") else "work"
    # 训练数据集的文件夹
    train_folder = os.environ["TRAIN_FOLDER"] if os.environ.__contains__("TRAIN_FOLDER") else "PotSimple"
    # 测试数据集的文件夹
    test_folder = os.environ["TEST_FOLDER"] if os.environ.__contains__("TEST_FOLDER") else "PotSimpleTest"
    # 子线程数量
    NUM_WORKERS = int(os.environ["NUM_WORKERS"] if os.environ.__contains__("NUM_WORKERS") else 1)
    # 每次训练的批次
    batch_size = int(os.environ["BATCH_SIZE"] if os.environ.__contains__("BATCH_SIZE") else 32)
    # 输出的文件名字
    OUT_FILE = os.environ["TEST_FOLDER"] if os.environ.__contains__("TEST_FOLDER") else "grad_12.csv"


    # 样品的数据来源
    train_pot_folder = []
    train_pot_folder.append(f"{DATA_SET_FOLDER}/{train_folder}")
    train_pot_folder.append(f"{DATA_SET_FOLDER}/{test_folder}")

    x_transforms = [ImgTo64Transform(), ImgToGrad12Transform(), ToTensor(tensor_type=torch.float32)]
    y_transforms = [ToTensor(tensor_type=torch.long)]

    train_dataset = HandWrittenDataSet(
        pot_folders=train_pot_folder, 
        need_label=True,
        x_transforms=x_transforms,
        y_transforms=y_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    data_map = {}
    XX = []
    yy = []
    labels = []
    data_map['XX'] = XX
    data_map['yy'] = yy
    data_map['labels'] = labels
    def reset_data():
        data_map = {}
        XX = []
        yy = []
        labels = []
        data_map['XX'] = XX
        data_map['yy'] = yy
        data_map['labels'] = labels
        pass


    out_f = f"{DATA_SET_FOLDER}/{OUT_FILE}"
    if os.path.isfile(out_f):
        os.remove(out_f)
    with alive_bar(len(train_loader)) as bar:
        for X, y in iter(train_loader):
            for i in range(len(X)):
                XX.append(X[i].tolist())
                yy.append(y[0][i].item())
                labels.append(y[1][i])
            data = pd.DataFrame(data_map)
            data.to_csv(out_f, header=False, mode='a')
            reset_data()
            
            # 显示进度条
            bar()

    # data = pd.DataFrame(data_map)
    # data.to_csv(f"{DATA_SET_FOLDER}/{OUT_FILE}", header=False, mode='a')

if __name__ == '__main__':
    main()