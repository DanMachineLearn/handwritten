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
from dataset.handwritten_pot_dataset import HandWrittenDataSet
import os
from torch.utils.data import DataLoader
from alive_progress import alive_bar

from img_to_64_64_transform import ImgTo64Transform
from img_to_grad_12_transform import ImgToGrad12Transform
from algorithm.to_tensor_transform import ToTensor

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
    OUT_FILE = os.environ["OUT_FILE"] if os.environ.__contains__("OUT_FILE") else "grad_12.csv"
    # 标签种类表
    OUT_LABEL_FILE = os.environ["OUT_LABEL_FILE"] if os.environ.__contains__("OUT_LABEL_FILE") else "grad_12.labels.csv"


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


    # 输出所有标签
    all_label_map = {}
    all_label_map['labels'] = train_dataset.labels
    all_label_data = pd.DataFrame(all_label_map)
    all_label_data.to_csv(f"{DATA_SET_FOLDER}/{OUT_LABEL_FILE}", mode='w+', index_label='index')


    # 输出所有字符
    out_f = f"{DATA_SET_FOLDER}/{OUT_FILE}"
    if os.path.isfile(out_f):
        os.remove(out_f)
    first = True
    index_item = 0
    with alive_bar(len(train_loader)) as bar:
        for X, y in iter(train_loader):
            XX = []
            yy = []
            labels = []
            index = []
            for i in range(len(X)):
                XX.append(X[i].tolist())
                yy.append(y[0][i].item())
                labels.append(y[1][i])
                index.append(index_item)
                index_item += 1
            data_map = {}
            data_map['id'] = index
            data_map['XX'] = XX
            data_map['yy'] = yy
            data_map['labels'] = labels
            data = pd.DataFrame(data_map)
            if first:
                data.to_csv(out_f, mode='w+', index=False)
                first = False
            else:
                data.to_csv(out_f, header=False, mode='a', index=False)
            # 显示进度条
            bar()

    # data = pd.DataFrame(data_map)
    # data.to_csv(f"{DATA_SET_FOLDER}/{OUT_FILE}", header=False, mode='a')

if __name__ == '__main__':
    main()