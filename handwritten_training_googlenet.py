# -*- encoding: utf-8 -*-
'''
@File		:	handwritten_training.py
@Time		:	2024/05/06 16:18:50
@Author		:	dan
@Description:	训练手写识别模型（使用卷积神经网络）
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
from dataset.handwritten_img_jpg_dataset import HandWrittenJpgDataSet
from models.HCCRGoogLeNetModel import GaborGoogLeNet
from dataset.handwritten_pot_dataset import HandWrittenDataSet
from img_to_64_64_transform import ImgTo64Transform
from img_to_grad_12_transform import ImgToGrad12Transform
from utils import matplot_tools
from utils.pot_downloader import PotDownloader
from algorithm.to_one_hot_transform import ToOneHot
from algorithm.to_tensor_transform import ToTensor
from torch.utils.data import IterableDataset
from torchvision.models.googlenet import GoogLeNetOutputs

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
    num_epochs = int(os.environ["NUM_EPOCHS"] if os.environ.__contains__("NUM_EPOCHS") else 5)
    # 前几次训练不修改学习率
    patience = int(os.environ["PATIENCE"] if os.environ.__contains__("PATIENCE") else 1)

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

    # x_transforms = [Channel1ToChannel3(), ToTensor(tensor_type=torch.float32)]
    # x_transforms = [Channel1ToGrad8_1(), ToTensor(tensor_type=torch.float32)]
    x_transforms = [ImgTo64Transform(channel_count=1, fast_handle=True), Channel1ToGabor8_1(image_only=True), ToTensor(tensor_type=torch.float32)]
    y_transforms = [ToTensor(tensor_type=torch.long)]

    train_dataset = HandWrittenJpgDataSet(train=True, jpg_folder=f"{DATA_SET_FOLDER}/jpg",
                                          x_transforms=x_transforms, y_transforms=y_transforms)
    
    test_dataset = HandWrittenJpgDataSet(train=False, jpg_folder=f"{DATA_SET_FOLDER}/jpg",
                                          x_transforms=x_transforms, y_transforms=y_transforms)

    shuffle = not isinstance(train_dataset, IterableDataset) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print("打开文件数量: ", train_dataset.file_count + test_dataset.file_count)
    print("打开文件数量: ", train_dataset.file_count + test_dataset.file_count)
    print("打开所有文件总耗时: ", '{:.2f} s'.format(time.time() - start_time))

    ## 创建模型
    model = GaborGoogLeNet(in_channels=9, num_classes=len(train_dataset.labels))
    if os.path.isfile(f"{MODEL_FOLDER}/googlenet_handwritten.pth"):
        model.load_state_dict(torch.load(f"{MODEL_FOLDER}/googlenet_handwritten.pth", map_location='cpu' if device == 'cpu' else None))

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5)

    # 训练模型
    i = 0
    test_loss_list = []
    test_correct_list = []
    train_loss_list = []
    train_correct_list = []
    for epoch in range(num_epochs):

        print(f"训练循环第{epoch + 1}/{num_epochs}个:")
        ## 用于输出训练的总进度
        model.train()  # 设置为训练模式
        train_loss = 0.0
        train_correct = 0.0
        import cv2
        # train_size = int(len(train_dataset) / batch_size)
        with alive_bar(len(train_loader)) as bar:
            for X, y in iter(train_loader):
                X : torch.Tensor = X
                # cv2.imshow("X", X.numpy()[0][0])
                # q = cv2.waitKey(-1)
                # if q == ord('q'):
                #     sys.exit(0)
                X, y = X.to(device), y.to(device)
                test_output : GoogLeNetOutputs = model(X)  # 前向传播
                # test_output = test_output.logits
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
            for test_X, test_y in iter(test_loader):
                test_X, test_y = test_X.to(device), test_y.to(device)
                test_output : torch.Tensor = model(test_X)


                 # 计算概率值
                probabilities = torch.softmax(test_output, dim=1)
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

        test_loss_list.append(test_loss)
        train_correct_list.append(train_correct)
        test_correct_list.append(correct)
        train_loss_list.append(train_loss)

        # 根据验证损失调整学习率
        # 一般来说，如果学习率调整的频率与 epoch 相关，每个 epoch 后都应调用一次 scheduler.step()；
        # 如果学习率调整基于验证损失，通常在验证步骤后调用 scheduler.step(val_loss)。
        scheduler.step(val_loss)


    torch.save(model.state_dict(), f"{MODEL_FOLDER}/googlenet_handwritten.pth")
    torch.save(train_dataset.labels, f"{MODEL_FOLDER}/googlenet_labels.bin")
    matplot_tools.draw_plot(test_loss_list, test_correct_list, train_loss_list, train_correct_list, "googlenet_train_log.jpg")

    all_classes = train_dataset.labels
    
    test_x = "deng.jpg"
    x_trainsforms = [ImgTo64Transform(need_dilate=True, channel_count=1), Channel1ToGrad8_1(), ToTensor(tensor_type=torch.float32)]
    for x_tran in x_trainsforms:
        test_x = x_tran(test_x)
    # test_x = test_x.reshape((1, test_x.shape[0], test_x.shape[1], test_x.shape[2]))
    test_x = torch.unsqueeze(torch.Tensor(test_x), 0)
    test_x = test_x.to(device=device)
    ## 预测结果
    model.eval()
    start_time = time.time()
    all_probs = []
    with torch.no_grad():
        pred = model(test_x)
        max = pred[0].argmax(0).item()
        predicted = all_classes[max]
        print(f'预测值: "{predicted}"')
        
            # 计算概率值
        probabilities = torch.softmax(test_output, dim=1)
        all_probs.append(probabilities)

    all_probs = torch.cat(all_probs, dim=0)
    print(f"前五值: {all_probs[:5]}")

if __name__ == '__main__':
    main()