# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_training_rnn.py
@Time		:	2024/05/21 15:18:18
@Author		:	dan
@Description:	根据笔画信息，训练RNN网络
'''

if __name__ == '__main__':
    import sys
    sys.path.append(".")
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from alive_progress import alive_bar
from algorithm.channel1_to_channel3 import Channel1ToChannel3
from algorithm.channel1_to_gabor8_1 import Channel1ToGabor8_1
from algorithm.channel1_to_grad8_1 import Channel1ToGrad8_1
from dataset.handwritten_img_bin_dataset import HandWrittenBinDataSet
from dataset.handwritten_pot_online_dataset import HandWrittenPotOnlineDataSet
from models.HCCRGoogLeNetModel import GaborGoogLeNet
from dataset.handwritten_pot_dataset import HandWrittenDataSet
from img_to_64_64_transform import ImgTo64Transform
from img_to_grad_12_transform import ImgToGrad12Transform
from models.rnn_model import HandwrittenRNN
from utils.pot_downloader import PotDownloader
from algorithm.to_one_hot_transform import ToOneHot
from algorithm.to_tensor_transform import ToTensor
from torch.utils.data import IterableDataset
from torchvision.models.googlenet import GoogLeNetOutputs
from torch.nn.utils.rnn import pad_sequence
import os

def main():

    # 是否从网站上下载
    DOWNLOAD = bool(os.environ["DOWNLOAD"] if os.environ.__contains__("DOWNLOAD") else True)

    # 下载的地址
    DOWNLOAD_URL = os.environ["DOWNLOAD"] if os.environ.__contains__("DOWNLOAD") else "https://gonsin-common.oss-cn-shenzhen.aliyuncs.com/handwritten"

    if DOWNLOAD:
        PotDownloader(download_url=DOWNLOAD_URL).start()


    ## 数据集目录
    # /gemini/data-1
    DATA_SET_FOLDER = os.environ["DATA_SET_FOLDER"] if os.environ.__contains__("DATA_SET_FOLDER") else "work"
    ## 模型目录
    # /gemini/pretrain
    MODEL_FOLDER = os.environ["MODEL_FOLDER"] if os.environ.__contains__("MODEL_FOLDER") else ""
    # 初始的学习率
    init_ln = float(os.environ["INIT_LN"] if os.environ.__contains__("INIT_LN") else 0.001)
    # 最低的学习率
    min_ln = float(os.environ["MIN_LN"] if os.environ.__contains__("MIN_LN") else 0.00001)
    # 每次训练的批次
    batch_size = int(os.environ["BATCH_SIZE"] if os.environ.__contains__("BATCH_SIZE") else 512)
    # 循环训练的次数
    num_epochs = int(os.environ["NUM_EPOCHS"] if os.environ.__contains__("NUM_EPOCHS") else 15)
    # 前几次训练不修改学习率
    patience = int(os.environ["PATIENCE"] if os.environ.__contains__("PATIENCE") else 1)
    # 训练数据集的文件夹
    train_folder = os.environ["TRAIN_FOLDER"] if os.environ.__contains__("TRAIN_FOLDER") else "PotSimple"
    # 测试数据集的文件夹
    test_folder = os.environ["TEST_FOLDER"] if os.environ.__contains__("TEST_FOLDER") else "PotSimpleTest"
    # 一次性加载所有数据到内存中
    LOAD_ALL_ON_INIT = bool(os.environ["LOAD_ALL_ON_INIT"] if os.environ.__contains__("LOAD_ALL_ON_INIT") else True)

    ## 优先使用cuda
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    print(f"using {device} device")

    optimizer = 1 # 使用adam，否则使用SDG

    # 样品的数据来源
    train_pot_folder = []
    # train_pot_folder.append(f"{DATA_SET_FOLDER}/PotSimple")
    train_pot_folder.append(f"{DATA_SET_FOLDER}/{train_folder}")
    # train_pot_folder.append(f"{DATA_SET_FOLDER}/PotTest")
    test_pot_folder = []
    test_pot_folder.append(f"{DATA_SET_FOLDER}/{test_folder}")
    # test_pot_folder.append(f"{DATA_SET_FOLDER}/PotSimple")
    # test_pot_folder.append(f"{DATA_SET_FOLDER}/PotSimpleTest")
    # pot_folder.append(f"{DATA_SET_FOLDER}/PotTest")
    # pot_folder.append(f"{DATA_SET_FOLDER}/PotTrain")

    import time
    start_time = time.time()
    ## 加载数据集

    train_pot_folder = []
    test_pot_folder = []
    train_pot_folder.append(f"work/Pot1.0/Pot1.0Train.zip_out")
    test_pot_folder.append(f"work/Pot1.0/Pot1.0Test.zip_out")
    
    
    x_transforms = [ToTensor(tensor_type=torch.float32)]
    y_transforms = [ToTensor(tensor_type=torch.long)]

    train_dataset = HandWrittenPotOnlineDataSet(
        pot_folders=train_pot_folder,
        x_transforms=x_transforms, y_transforms=y_transforms)
    
    test_dataset = HandWrittenPotOnlineDataSet(
        pot_folders=test_pot_folder,
        outter_labels=train_dataset.labels,
        x_transforms=x_transforms, y_transforms=y_transforms)
    


    def collate_fn(batch_data):  
        """
        自定义 batch 内各个数据条目的组织方式
        :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
        :return: 填充后的句子列表、实际长度的列表、以及label列表
        """
        # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
        # (tensor([1, 2, 3, 5]), 4, 0)
        # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
        # pack_padded_sequence函数要求要按照序列的长度倒序排列
        batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
        data_length = [len(xi[0]) for xi in batch_data]
        sent_seq = [xi[0] for xi in batch_data]
        # sent_seq = torch.tensor(sent_seq, dtype=torch.float32)
        label = [xi[1] for xi in batch_data]
        padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0)
        return padded_sent_seq, data_length, torch.tensor(label, dtype=torch.long)




    shuffle = not isinstance(train_dataset, IterableDataset) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # print("打开文件数量: ", train_dataset.file_count + test_dataset.file_count)
    print("打开文件数量: ", train_dataset.file_count + test_dataset.file_count)
    print("打开所有文件总耗时: ", '{:.2f} s'.format(time.time() - start_time))

    ## 创建模型
    model = HandwrittenRNN(output_size=len(train_dataset.labels))
    model = model.to(device)


    ## 创建学习器
    if optimizer == 1:
        
        # lr：学习率。学习率决定了每一步优化更新的大小。较大的学习率可能导致不稳定，较小的学习率
        # 可能导致训练缓慢或停滞。
        optimizer = optim.Adam(model.parameters(), lr=init_ln)  # 设置初始学习率
    else:
        # lr：学习率。学习率决定了每一步优化更新的大小。较大的学习率可能导致不稳定，较小的学习率
        # 可能导致训练缓慢或停滞。
        # momentum：动量。动量是在 SGD 中加入一种惯性，允许优化器在更新时“记住”之前的方向。这有
        # 助于加速收敛并减轻局部最小值的困扰。通常设定在 0 到 1 之间，如 0.9。
        # dampening：阻尼。与动量结合使用，阻尼控制动量在每次更新时的削弱程度。一般与动量相等，
        # 但在某些情况下（如不希望应用动量）可以设置为 0。
        # weight_decay：权重衰减（L2 正则化）。权重衰减在训练中用于防止过拟合，通过在更新过程中
        # 对参数施加 L2 正则化。它是一种惩罚项，通常设定为小数，如 0.0001。
        # nesterov：是否启用 Nesterov 动量。Nesterov 动量是一种改进的动量方法，考虑到梯度更新
        # 的趋势，有助于更快收敛和减少振荡。设定为 True 表示使用 Nesterov 动量。
        optimizer = optim.SGD(model.parameters(), lr=init_ln)
    


    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    # 使用 ReduceLROnPlateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5, min_lr=min_ln)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5)

    # 训练模型
    i = 0
    for epoch in range(num_epochs):

        print(f"训练循环第{epoch + 1}/{num_epochs}个:")
        ## 用于输出训练的总进度
        model.train()  # 设置为训练模式
        train_loss = 0.0
        train_correct = 0.0
        # train_size = int(len(train_dataset) / batch_size)
        with alive_bar(len(train_loader)) as bar:
            for X, length, y in iter(train_loader):
                X, y = X.to(device), y.to(device)
                test_output = model(X)  # 前向传播
                loss = criterion(test_output, y) 
                train_loss += loss.item()

                predicted = torch.max(test_output.data,1)[1]
                c = (predicted == y).type(torch.float).sum().item()
                train_correct +=  c
                loss.backward(retain_graph=False)  # 反向传播，不累计梯度
                optimizer.step()
                optimizer.zero_grad()  # 清空梯度
                i+=1
                # 显示进度条
                bar()
        # 计算验证损失
        model.eval()  # 设置为评估模式
        test_loss, correct = 0, 0
        with torch.no_grad():
            for test_X, length, test_y in iter(test_loader):
                test_X, test_y = test_X.to(device), test_y.to(device)
                test_output : torch.Tensor = model(test_X)
                val_loss = criterion(test_output, test_y)
                test_loss += val_loss.item()

                max_args = test_output.argmax(dim = 1)
                correct += (test_output.argmax(1) == test_y).type(torch.float).sum().item()
        test_loss /= len(test_loader)
        train_correct /= len(train_loader.dataset)
        correct /= len(test_loader.dataset)
        train_loss /= len(train_loader)
        print(f"训练集: \n 准确率: {100 * train_correct:>01f}%, 平均 Loss: {train_loss:>8f}")
        print(f"测试集: \n 准确率: {100 * correct:>01f}%, 平均 Loss: {test_loss:>8f}\n")

        # 根据验证损失调整学习率
        # 一般来说，如果学习率调整的频率与 epoch 相关，每个 epoch 后都应调用一次 scheduler.step()；
        # 如果学习率调整基于验证损失，通常在验证步骤后调用 scheduler.step(val_loss)。
        scheduler.step(val_loss)

    torch.save(model.state_dict(), f"{MODEL_FOLDER}/rnn_handwritten.pth")
    torch.save(train_dataset.labels, f"{MODEL_FOLDER}/rnn_labels.bin")

if __name__ == '__main__':
    main()