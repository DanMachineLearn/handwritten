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
from models.HCCRGoogLeNetModel import GaborGoogLeNet
from dataset.handwritten_pot_dataset import HandWrittenDataSet
from img_to_64_64_transform import ImgTo64Transform
from img_to_grad_12_transform import ImgToGrad12Transform
from utils.pot_downloader import PotDownloader
from algorithm.to_one_hot_transform import ToOneHot
from algorithm.to_tensor_transform import ToTensor
from torch.utils.data import IterableDataset
from torchvision.models.googlenet import GoogLeNetOutputs

import os

def main():

    ## 优先使用cuda
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    print(f"using {device} device")


    import time
    start_time = time.time()
    ## 加载数据集
    MODEL_FOLDER = os.environ["MODEL_FOLDER"] if os.environ.__contains__("MODEL_FOLDER") else "pretrain"
    DATA_SET_FOLDER = os.environ["DATA_SET_FOLDER"] if os.environ.__contains__("DATA_SET_FOLDER") else "work"

    all_classes = torch.load(os.path.join(MODEL_FOLDER, "googlenet_labels.bin"))
    model = GaborGoogLeNet(in_channels=9, num_classes=len(all_classes))
    model = model.to(device)
    model.load_state_dict(torch.load(f"{MODEL_FOLDER}/googlenet_handwritten.pth", map_location='cpu' if device == 'cpu' else None))

    test_x = "chang.png"
    x_trainsforms = [ImgTo64Transform(need_dilate=True, channel_count=1, show_plt=True), Channel1ToGrad8_1(), ToTensor(tensor_type=torch.float32)]
    for x_tran in x_trainsforms:
        test_x = x_tran(test_x)
    # test_x = test_x.reshape((1, test_x.shape[0], test_x.shape[1], test_x.shape[2]))
    test_x = torch.unsqueeze(torch.Tensor(test_x), 0)
    test_x.to(device=device)
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