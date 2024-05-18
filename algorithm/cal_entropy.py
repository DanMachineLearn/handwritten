# -*- encoding: utf-8 -*-
'''
@File		:	cal_entropy.py
@Time		:	2024/05/18 10:51:23
@Author		:	dan
@Description:	根据 https://blog.csdn.net/YH_24H/article/details/112664232 的算法，获取一张图片的二维熵
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

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

    N = 1 # 设置邻域属性，目标点周围1个像素点设置为邻域，九宫格，如果为2就是25宫格...
    S=img.shape
    IJ = []
    #计算j
    for row in range(S[0]):
        for col in range(S[1]):
            Left_x=np.max([0,col-N])
            Right_x=np.min([S[1],col+N+1])
            up_y=np.max([0,row-N])
            down_y=np.min([S[0],row+N+1])
            region=img[up_y:down_y,Left_x:Right_x] # 九宫格区域
            j = (np.sum(region) - img[row][col])/((2*N+1)**2-1)
            IJ.append([img[row][col],j])
    # 计算F(i,j)
    F=[]
    arr = [list(i) for i in set(tuple(j) for j in IJ)] #去重，会改变顺序，不过此处不影响
    for i in range(len(arr)):
        F.append(IJ.count(arr[i]))
    # 计算pij
    P=np.array(F)/(img.shape[0]*img.shape[1])#也是img的W*H

    # 计算熵
	E = np.sum([p *np.log2(1/p) for p in P])
    return E


def main():
    pass

if __name__ == '__main__':
    main()