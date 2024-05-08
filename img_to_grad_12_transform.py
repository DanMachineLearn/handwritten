

# -*- encoding: utf-8 -*-
'''
@File		:	pot_to_4direction_transform.py
@Time		:	2024/05/06 14:59:10
@Author		:	dan
@Description:	numpy 图像转 4方向线素的转换器
'''

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import torch


class ImgToGrad12Transform:

    @property
    def feature_count(self) -> int:
        '''
        特征数量
        '''
        return self.__feature_count

    '''
    pot数据转grad-12线素数据
    '''
    def __init__(self, show_plt : bool = False) -> None:
        ''' 
        show_plt 是否显示plt图像，用于测试
        '''
        self.__feature_count = 12
        self.__show_plt = show_plt
        pass

    def get_direction_sums(self, image : np.ndarray, need_dilate : bool = True):
        ''' 获取图像grad-12方向线素特征
        
        Parameters
        ----------
        need_dilate : bool = True, 是否需要膨胀处理，如果原始图像是大于64 * 64，则为了避免在图像缩小的时候信息丢失，在缩小之前要进行膨胀处理。
        
        Returns
        -------
        
        
        '''

        # 膨胀图像
        if need_dilate:
            # 膨胀和腐蚀都是对于白色像素而言的，所以对于黑色的膨胀，则需要进行白色的腐蚀。
            kernel = np.ones((5, 5), dtype=np.uint8) # 卷积核变为4*4
            image = cv.erode(image, kernel, iterations=1)


            # # 所以也可以将白色和黑色进行转换，通过 dilate 膨胀之后，再切换回来。
            # image = 255 - image
            # kernel = np.ones((5, 5), dtype=np.uint8) # 卷积核变为4*4
            # image = cv2.dilate(image, kernel, iterations=1)
            # image = 255 - image

        # # 计算图像的重心
        # # 使用图像的矩计算质心
        # moments = cv2.moments(image)
        # center_x = int(moments['m10'] / moments['m00'])  # x 轴重心
        # center_y = int(moments['m01'] / moments['m00'])  # y 轴重心


        # 获取非空区域的边界
        # 使用行和列求和来判断非空区域
        rows_sum = np.sum(image - 255, axis=1)
        cols_sum = np.sum(image - 255, axis=0)
        # 找到行和列的最小和最大非空索引
        top = np.argmax(rows_sum > 0)  # 第一个非空行
        bottom = len(rows_sum) - np.argmax(rows_sum[::-1] > 0)  # 最后一个非空行
        left = np.argmax(cols_sum > 0)  # 第一个非空列
        right = len(cols_sum) - np.argmax(cols_sum[::-1] > 0)  # 最后一个非空列
        image = image[top:bottom, left:right]

        # 3. 调整图像大小
        # 将图像调整为 64x64
        resized_image = cv.resize(image, (64, 64), interpolation=cv.INTER_LINEAR)


        # 通过阈值将图像二值化
        # src： 输入图像。可以是单通道或多通道图像，但在二值化操作中，通常是灰度图像。
        # thresh： 阈值。如果图像的像素值超过或小于此阈值，将根据指定的规则进行处理。
        # maxval： 最大值。当像素值满足指定条件时，将其替换为这个最大值。
        # type： 阈值类型，决定了如何处理像素值。主要的阈值类型包括：
        # cv2.THRESH_BINARY：二值化。像素值大于阈值时设置为 maxval，否则设置为 0。
        # cv2.THRESH_BINARY_INV：二值化反转。像素值小于或等于阈值时设置为 maxval，否则设置为 0。
        # cv2.THRESH_TRUNC：截断。像素值大于阈值时被截断到阈值，其他值保持不变。
        # cv2.THRESH_TOZERO：阈值到零。像素值大于阈值时保持不变，其他值设置为 0。
        # cv2.THRESH_TOZERO_INV：阈值到零反转。像素值小于或等于阈值时保持不变，其他值设置为 0。
        _, binary_image = cv.threshold(resized_image, 128, 255, cv.THRESH_BINARY_INV)
        sobel_x = cv.Sobel(binary_image, cv.CV_64F, 1, 0, ksize=3)  # 水平方向
        sobel_y = cv.Sobel(binary_image, cv.CV_64F, 0, 1, ksize=3)  # 垂直方向

        # 计算梯度的幅度和角度
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        angle = np.arctan2(sobel_y, sobel_x)

        # 将梯度角度转换到[0, 360)范围
        angle = np.degrees(angle)
        angle[angle < 0] += 360

        # 将角度分成12个区间，每个30度
        num_bins = 12
        bin_edges = np.linspace(0, 360, num_bins + 1)

        # 计算梯度角度的直方图
        grad_12_features = np.histogram(angle, bins=bin_edges, weights=magnitude)[0]

        if self.__show_plt:
            # 打印Grad-12特征
            print("Grad-12 Features:")
            print(grad_12_features)

        return grad_12_features;


    def __call__(self, img : np.ndarray):
        features = self.get_direction_sums(img)
        features = np.array(features).flatten()
        features = torch.tensor(features, dtype=torch.float32)
        return features



def main():
    image_path = "handwritten_chinese.jpg"
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    transform = ImgToGrad12Transform(show_plt=True)
    features = transform(image)
    print(features)


if __name__ == '__main__':
    main()
