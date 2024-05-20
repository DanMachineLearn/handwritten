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
import time
import os


class ProductGooglenet:
    '''
    用于生产环境的模型
    '''
    @property
    def model_name(self) -> str:
        '''
        模型的名字
        '''
        return self.__model_name
    
    def __init__(self) -> None:
        ''' 
        
        Parameters
        ----------
        
        
        '''
        ## 优先使用cuda
        self.__device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.__model_name = 'GoogLeNet'
        self.__all_classes = torch.load(os.path.join('pretrain', "googlenet_labels.bin"))
        self.__model = GaborGoogLeNet(in_channels=9, num_classes=len(self.__all_classes))
        self.__model = self.__model.to(self.__device)
        self.__model.load_state_dict(torch.load(f"pretrain/googlenet_handwritten.pth", map_location='cpu' if self.__device == 'cpu' else None))
        self.__model.eval()
        self.__x_trainsforms = [
            ImgTo64Transform(need_dilate=True, channel_count=1), 
            Channel1ToGrad8_1(), 
            ToTensor(tensor_type=torch.float32)]
        
        pass

    def check(self, image : str | np.ndarray) -> tuple[list[str], list]:
        ''' 监测图片，判断图片属于哪个字体
        
        Parameters
        ----------
        
        
        Returns
        -------
        list[str]
        
        '''
        for x_tran in self.__x_trainsforms:
            image = x_tran(image)
        image = torch.unsqueeze(torch.Tensor(image), 0)
        image = image.to(device=self.__device)
        start_time = time.time()
        max_labels = []
        with torch.no_grad():
            pred = self.__model(image)
            max = pred[0].argmax(0).item()
            predicted = self.__all_classes[max]
            print(f'预测值: "{predicted}"')
            max_list : np.ndarray = np.argsort(-pred[0])[0:9].numpy()
            max_pred = pred[0][max_list]
            max_list = max_list.astype(np.int32)
            max_list = max_list.tolist()
            print("结果输出")
            i = 0
            for l in max_list:
                max_labels.append(self.__all_classes[l])
                print(f"{l}\t{self.__all_classes[l]}(置信度 {np.log(-max_pred[i])})")
                i += 1
            print("总耗时", '{:.2f} s'.format(time.time() - start_time))
        return max_labels, max_pred

def main():

    test_x = "chang.jpg"
    net = ProductGooglenet()
    net.check(test_x)

if __name__ == '__main__':
    main()