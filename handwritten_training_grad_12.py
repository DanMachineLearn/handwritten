# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_training_grad_12.py
@Time		:	2024/05/13 15:25:42
@Author		:	dan
@Description:	使用神经网络，训练grad-12数据
'''

from handwritten_model import HandWrittenModel
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from alive_progress import alive_bar
from handwritten_grad_12_csv_dataset import HandWrittenGrad12CsvDataSet
from handwritten_model_cnn import HandWrittenCnnModel
from handwritten_pot_dataset import HandWrittenDataSet
from img_to_64_64_transform import ImgTo64Transform
from img_to_grad_12_transform import ImgToGrad12Transform
from pot_downloader import PotDownloader
from to_one_hot_transform import ToOneHot
from to_tensor_transform import ToTensor
from torch.utils.data import IterableDataset

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
    min_ln = float(os.environ["MIN_LN"] if os.environ.__contains__("MIN_LN") else 0.0001)
    # 每次训练的批次
    batch_size = int(os.environ["BATCH_SIZE"] if os.environ.__contains__("BATCH_SIZE") else 512)
    # 循环训练的次数
    num_epochs = int(os.environ["NUM_EPOCHS"] if os.environ.__contains__("NUM_EPOCHS") else 1)
    # 前几次训练不修改学习率
    patience = int(os.environ["PATIENCE"] if os.environ.__contains__("PATIENCE") else 1)
    # 训练数据集的文件夹
    train_folder = os.environ["TRAIN_FOLDER"] if os.environ.__contains__("TRAIN_FOLDER") else "PotSimple"
    # 测试数据集的文件夹
    test_folder = os.environ["TEST_FOLDER"] if os.environ.__contains__("TEST_FOLDER") else "PotSimpleTest"

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

    x_transforms = [ImgTo64Transform(need_dilate=False), ImgToGrad12Transform(), ToTensor(tensor_type=torch.float32)]
    y_transforms = [ToTensor(tensor_type=torch.long)]

    train_dataset = HandWrittenDataSet(
        pot_folders=train_pot_folder, 
        x_transforms=x_transforms,
        y_transforms=y_transforms)
    
    test_dataset = HandWrittenDataSet(
        pot_folders=test_pot_folder, 
        outter_labels=train_dataset.labels,
        x_transforms=x_transforms,
        y_transforms=y_transforms)

    shuffle = not isinstance(train_dataset, IterableDataset) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print("打开文件数量: ", train_dataset.file_count + test_dataset.file_count)
    print("打开文件数量: ", train_dataset.file_count + test_dataset.file_count)
    print("打开所有文件总耗时: ", '{:.2f} s'.format(time.time() - start_time))

    ## 创建模型
    model = HandWrittenModel(input_features=ImgToGrad12Transform.INPUT_FEATURES, output_classes=len(train_dataset.labels))
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
        # train_size = int(len(train_dataset) / batch_size)
        with alive_bar(len(train_loader)) as bar:
            for X, y in iter(train_loader):
                X, y = X.to(device), y.to(device)
                test_output = model(X)  # 前向传播
                loss = criterion(test_output, y) 
                train_loss += loss.item()
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
            for test_X, test_y in iter(test_loader):
                test_X, test_y = test_X.to(device), test_y.to(device)
                test_output : torch.Tensor = model(test_X)
                val_loss = criterion(test_output, test_y)
                test_loss += val_loss.item()

                max_args = test_output.argmax(dim = 1)
                correct += (test_output.argmax(1) == test_y).type(torch.float).sum().item()
        test_loss /= len(train_loader)
        correct /= len(train_loader.dataset)
        train_loss /= len(train_loader)
        print(f"训练集: \n 平均 Loss: {train_loss:>8f}")
        print(f"测试集: \n 准确率: {100 * correct:>01f}%, 平均 Loss: {test_loss:>8f}\n")

        # 根据验证损失调整学习率
        # 一般来说，如果学习率调整的频率与 epoch 相关，每个 epoch 后都应调用一次 scheduler.step()；
        # 如果学习率调整基于验证损失，通常在验证步骤后调用 scheduler.step(val_loss)。
        scheduler.step(val_loss)


    torch.save(model.state_dict(), f"{MODEL_FOLDER}/handwritten_nn.pth")

    all_classes = train_dataset.labels
    
    test_x = "handwritten_chinese.jpg"
    for x_tran in x_transforms:
        test_x = x_tran(test_x)
    test_x = test_x.reshape((1, test_x.shape[0], test_x.shape[1], test_x.shape[2]))

    ## 预测结果
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        pred = model(test_x)
        max = pred[0].argmax(0).item()
        predicted = all_classes[max]
        print(f'预测值: "{predicted}"')
        max_list : np.ndarray = np.argsort(-pred[0])[0:9].numpy()
        max_list = max_list.astype(np.int32)
        max_list = max_list.tolist()
        print("结果输出")
        for l in max_list:
            print(f"{l}\t{all_classes[l]}")
        # print("结果输出2", min_list)
        print("总耗时", '{:.2f} ms'.format(time.time() - start_time))


if __name__ == '__main__':
    main()