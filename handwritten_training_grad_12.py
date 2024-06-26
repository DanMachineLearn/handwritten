# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_training_grad_12.py
@Time		:	2024/05/13 15:25:42
@Author		:	dan
@Description:	使用神经网络，训练grad-12数据
'''

from sklearn.metrics import accuracy_score
from dataset.handwritten_dataset_csv_grad_12 import HandWrittenDatasetCsvGrad12
from models.handwritten_model import HandWrittenModel
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from alive_progress import alive_bar
from img_to_64_64_transform import ImgTo64Transform
from img_to_grad_12_transform import ImgToGrad12Transform
from utils.pot_downloader import PotDownloader
from algorithm.to_tensor_transform import ToTensor
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
    MODEL_FOLDER = os.environ["MODEL_FOLDER"] if os.environ.__contains__("MODEL_FOLDER") else "pretrain"

    # 子线程数量
    NUM_WORKERS = int(os.environ["NUM_WORKERS"] if os.environ.__contains__("NUM_WORKERS") else 1)
    # 初始的学习率
    INIT_LN = float(os.environ["INIT_LN"] if os.environ.__contains__("INIT_LN") else 0.001)
    # 最低的学习率
    MIN_LN = float(os.environ["MIN_LN"] if os.environ.__contains__("MIN_LN") else 0.0001)
    # 每次训练的批次
    BATCH_SIZE = int(os.environ["BATCH_SIZE"] if os.environ.__contains__("BATCH_SIZE") else 32)
    # 循环训练的次数
    NUM_EPOCHS = int(os.environ["NUM_EPOCHS"] if os.environ.__contains__("NUM_EPOCHS") else 20)
    # 前几次训练不修改学习率
    PATIENCE = int(os.environ["PATIENCE"] if os.environ.__contains__("PATIENCE") else 1)
    # 训练数据集的文件
    DATA_CSV_FILE = os.environ["DATA_CSV_FILE"] if os.environ.__contains__("DATA_CSV_FILE") else "grad_12.csv"
    # 测试数据集的文件
    LABEL_CSV_FOILE = os.environ["LABEL_CSV_FOILE"] if os.environ.__contains__("LABEL_CSV_FOILE") else "grad_12.labels.csv"
    # 测试集大小
    TEST_SIZE = int(os.environ["TEST_SIZE"] if os.environ.__contains__("TEST_SIZE") else 512)

    MAX_LENGTH = int(os.environ['MAX_LENGTH'] if os.environ.__contains__('MAX_LENGTH') else -1)
    if MAX_LENGTH == -1:
        MAX_LENGTH = None

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

    import time
    start_time = time.time()
    ## 加载数据集

    # x_transforms = [ImgTo64Transform(), ImgToGrad12Transform(), ToTensor(tensor_type=torch.float32)]
    x_transforms = [ToTensor(tensor_type=torch.float32)]
    y_transforms = [ToTensor(tensor_type=torch.long)]

    train_dataset = HandWrittenDatasetCsvGrad12(
        data_csv_path=f"{DATA_SET_FOLDER}/{DATA_CSV_FILE}", 
        label_csv_path=f"{DATA_SET_FOLDER}/{LABEL_CSV_FOILE}",
        start_index=TEST_SIZE,
        max_length=MAX_LENGTH,
        x_transforms=x_transforms,
        y_transforms=y_transforms)
    
    test_dataset = HandWrittenDatasetCsvGrad12(
        data_csv_path=f"{DATA_SET_FOLDER}/{DATA_CSV_FILE}", 
        label_csv_path=f"{DATA_SET_FOLDER}/{LABEL_CSV_FOILE}",
        max_length=TEST_SIZE,
        # start_index=TEST_SIZE,
        x_transforms=x_transforms,
        y_transforms=y_transforms)
    
    print("len(train_dataset) = ", len(train_dataset))
    print("len(test_dataset) = ", len(test_dataset))

    shuffle = not isinstance(train_dataset, IterableDataset) 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # print("打开文件数量: ", train_dataset.file_count + test_dataset.file_count)
    print("打开文件数量: ", train_dataset.file_count)
    print("打开所有文件总耗时: ", '{:.2f} s'.format(time.time() - start_time))

    ## 创建模型
    model = HandWrittenModel(input_features=ImgToGrad12Transform.INPUT_FEATURES, output_classes=len(train_dataset.labels))
    model = model.to(device)


    ## 创建学习器
    if optimizer == 1:
        
        # lr：学习率。学习率决定了每一步优化更新的大小。较大的学习率可能导致不稳定，较小的学习率
        # 可能导致训练缓慢或停滞。
        optimizer = optim.Adam(model.parameters(), lr=INIT_LN)  # 设置初始学习率
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
        optimizer = optim.SGD(model.parameters(), lr=INIT_LN)
    


    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    # 使用 ReduceLROnPlateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, factor=0.5, min_lr=MIN_LN)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5)

    # 训练模型
    i = 0
    for epoch in range(NUM_EPOCHS):

        print(f"训练循环第{epoch + 1}/{NUM_EPOCHS}个:")
        ## 用于输出训练的总进度
        model.train()  # 设置为训练模式
        train_loss = 0.0
        train_correct = 0.0
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

                predicted = torch.max(test_output.data,1)[1]
                c = (predicted == y).type(torch.float).sum().item()
                train_correct +=  c

                # 显示进度条
                bar()
        # 计算验证损失
        model.eval()  # 设置为评估模式
        test_loss, correct = 0, 0
        with torch.no_grad():
            for test_X, test_y in iter(test_loader):
                test_X, test_y = test_X.to(device), test_y.to(device)
                test_output : torch.Tensor = model(test_X)
                # test_output : torch.Tensor = model.predict_proba(test_X)
                # pred_y = test_output[:, 1]
                val_loss = criterion(test_output, test_y)
                test_loss += val_loss.item()

                max_args = np.argmax(test_output, 1)
                max_args = np.around(max_args)
                correct += accuracy_score(test_y, max_args)
                # max_args = test_output.argmax(dim = 1)
                # correct += accuracy_score(test_y, test_output)
                # correct += (test_output.argmax(1).type(torch.long) == test_y).type(torch.float).sum().item()

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


    torch.save(model.state_dict(), f"{MODEL_FOLDER}/handwritten_nn.pth")

    all_classes = train_dataset.labels
    
    test_x = "deng.jpg"
    transforms = [ImgTo64Transform(), ImgToGrad12Transform(), ToTensor(tensor_type=torch.float32)]
    for x_tran in transforms:
        test_x = x_tran(test_x)
    test_x = test_x.reshape((1, test_x.shape[0]))

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