# -*- encoding: utf-8 -*-
'''
@File		:	cal_entropy.py
@Time		:	2024/05/18 10:51:23
@Author		:	dan
@Description:	根据 https://blog.csdn.net/YH_24H/article/details/112664232 的算法，获取一张图片的二维熵
'''
if __name__ == '__main__':
    import sys
    sys.path.append('.')
from alive_progress import alive_bar
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import entropy

from dataset.handwritten_img_bin_dataset import HandWrittenBinDataSet

def calc_2D_Entropy(img : np.ndarray):
    '''

    img : np.ndarray , 图像

    邻域 3*3的小格子
     __ __ __
    |__|__|__|
    |__||||__|
    |__|__|__|
    角点
     __ __
    ||||__|
    |__|__|
    边
     __ __
    |  |__|
    ||||__|
    |__|__|
    '''

    # N = 1 # 设置邻域属性，目标点周围1个像素点设置为邻域，九宫格，如果为2就是25宫格...
    # S=img.shape
    # IJ = []
    # #计算j
    # for row in range(S[0]):
    #     for col in range(S[1]):
    #         Left_x=np.max([0,col-N])
    #         Right_x=np.min([S[1],col+N+1])
    #         up_y=np.max([0,row-N])
    #         down_y=np.min([S[0],row+N+1])
    #         region=img[up_y:down_y,Left_x:Right_x] # 九宫格区域
    #         j = (np.sum(region) - img[row][col])/((2*N+1)**2-1)
    #         IJ.append([img[row][col],j])
    # # 计算F(i,j)
    # F=[]
    # arr = [list(i) for i in set(tuple(j) for j in IJ)] #去重，会改变顺序，不过此处不影响
    # for i in range(len(arr)):
    #     F.append(IJ.count(arr[i]))
    # # 计算pij
    # P=np.array(F)/(img.shape[0]*img.shape[1]) #也是img的W*H

    # # 计算熵
    # E = np.sum([p *np.log2(1/p) for p in P])



    # 计算每个灰度值的直方图
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    
    # 计算每个灰度值的概率
    probability_distribution = histogram / np.sum(histogram)
    
    # 计算图像的熵
    img_entropy = entropy(probability_distribution, base=2)
    if img_entropy < 0:
        print(img_entropy)
    # print('Image entropy:', entropy)
    return img_entropy


def main():
    '''
    计算所有图像的熵，获取最大值和最小值，分为16个区间，并且列出每个区间对应的字体是哪些
    '''

    import time
    start_time = time.time()
    
    # x_transforms = [Channel1ToChannel3(), ToTensor(tensor_type=torch.float32)]
    # y_transforms = [ToTensor(tensor_type=torch.long)]
    x_transforms = None
    y_transforms = None
    dataset = HandWrittenBinDataSet(
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        bin_folder="work/Bin", 
        train=True)
    max_entropy = 0
    min_entropy = float('inf')
    mean_entropy = None
    count = 0
    with alive_bar(len(dataset)) as bar:
        for X, y in dataset:
            count += 1
            # cv2.imshow("X", X.reshape((64, 64)))
            # cv2.waitKey(-1)
            E = calc_2D_Entropy(X)
            max_entropy = max(E, max_entropy)
            min_entropy = min(E, min_entropy)
            if count == 1:
                mean_entropy = E
            else:
                mean_entropy = mean_entropy - (mean_entropy - E) / count
            bar()

    dataset = HandWrittenBinDataSet(
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        bin_folder="work/Bin", 
        train=False)
    with alive_bar(len(dataset)) as bar:
        for X, y in dataset:
            count += 1
            E = calc_2D_Entropy(X)
            max_entropy = max(E, max_entropy)
            min_entropy = min(E, min_entropy)
            if count == 1:
                mean_entropy = E
            else:
                mean_entropy = mean_entropy - (mean_entropy - E) / count
            bar()
    print(f"计算了{count}个图像，最大值 为 {max_entropy}, 最小值 为 {min_entropy}, 平均值 为 {mean_entropy}")
    print("总耗时: ", '{:.2f} s'.format(time.time() - start_time))
    # 计算了1622935个图像，最大值 为 0.9999993120692872, 最小值 为 0.21257895128187496, 平均值 为 0.7090148988206749
    pass

    pass

if __name__ == '__main__':
    main()