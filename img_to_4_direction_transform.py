

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


class ImgTo4DirectionTransform:

    @property
    def feature_count(self) -> int:
        '''
        特征数量
        '''
        return self.__feature_count
    
    '''
    pot数据转4方向线素数据
    '''
    def __init__(self, frame_count : int = 8) -> None:
        ''' 
        Parameters
        ----------
        frame_count 表示提取方向线素之后，每行和每列分别平均划分多少个方格
        
        '''
        self.__frame_count = frame_count
        self.__feature_count = frame_count * frame_count * 4
        pass


    def handle_grad(self, d0, d1, d2, d3, d4, d5, d6, d7):
        ''' 处理二值图像，得到8方向的分数图
        1 2 3
        0   4
        7 6 5
        Parameters
        ----------
        
        
        Returns
        -------
        
        
        '''

        p0 = (d0 + d4) / 2
        p1 = (d1 + d5) / 2
        p2 = (d2 + d6) / 2
        p3 = (d3 + d7) / 2
        return (p0, p1, p2, p3)


    def split_grad_image(self, image) -> list:
        ''' 将图像分割成方向图，输入图像必须先被二值化 （0, 1）
        
        Parameters
        ----------
        
        
        Returns
        -------
        
        
        '''
        height, width = image.shape;
        grad_images0 = np.zeros((height, width), dtype=np.float32)
        grad_images1 = np.zeros((height, width), dtype=np.float32)
        grad_images2 = np.zeros((height, width), dtype=np.float32)
        grad_images3 = np.zeros((height, width), dtype=np.float32)
        for h in range(height):
            for w in range(width):
                if image[h, w] == 0:
                    continue

                d0 = d1 = d2 = d3 = d4 = d5 = d6 = d7 = 0
                if h == 0:
                    d1 = d2 = d3 = 0
                    ## 左上角的情况
                    if w == 0:
                        d1 = d0 = d7 = 0
                        d4 = image[h, w + 1] 
                        d5 = image[h + 1, w + 1] 
                        d6 = image[h + 1, w]

                    ## 右上角的情况
                    elif w == width - 1:
                        d3 = d4 = d5 = 0
                        d0 = image[h, w - 1] 
                        d7 = image[h + 1, w - 1] 
                        d6 = image[h + 1, w]

                    ## 上边沿情况
                    else:
                        d4 = image[h, w + 1]
                        d5 = image[h + 1, w + 1] 
                        d6 = image[h + 1, w]
                        d7 = image[h + 1, w - 1] 
                        d0 = image[h, w - 1] 


                elif h == height - 1:
                    d5 = d6 = d7 = 0
                    ## 左下角情况
                    if w == 0:
                        d1 = d0 = d7 = 0
                        d2 = image[h - 1, w] 
                        d3 = image[h - 1, w + 1] 
                        d4 = image[h, w + 1]

                    ## 右下角情况
                    elif w == width - 1:
                        d3 = d4 = d5 = 0
                        d0 = image[h, w - 1] 
                        d1 = image[h - 1, w - 1] 
                        d2 = image[h - 1, w]

                    ## 下边沿的情况
                    else:
                        d0 = image[h, w - 1] 
                        d1 = image[h - 1, w - 1] 
                        d2 = image[h - 1, w] 
                        d3 = image[h - 1, w + 1] 
                        d4 = image[h, w + 1]

                ## 左边延
                elif w == 0:
                    d1 = d0 = d7 = 0
                    d2 = image[h - 1, w] 
                    d3 = image[h - 1, w + 1] 
                    d4 = image[h, w + 1]
                    d5 = image[h + 1, w + 1] 
                    d6 = image[h + 1, w]

                ## 右边延
                elif w == width - 1:
                    d3 = d4 = d5 = 0
                    d0 = image[h, w - 1] 
                    d1 = image[h - 1, w - 1] 
                    d2 = image[h - 1, w] 
                    d7 = image[h + 1, w - 1] 
                    d6 = image[h + 1, w]

                # 其他区域
                else:
                    d0 = image[h, w - 1] 
                    d1 = image[h - 1, w - 1] 
                    d2 = image[h - 1, w] 
                    d3 = image[h - 1, w + 1] 
                    d4 = image[h, w + 1]
                    d5 = image[h + 1, w + 1] 
                    d6 = image[h + 1, w]
                    d7 = image[h + 1, w - 1] 

                grad = self.handle_grad(d0, d1, d2, d3, d4, d5, d6, d7)
                grad_images0[h, w] = grad[0]
                grad_images1[h, w] = grad[1]
                grad_images2[h, w] = grad[2]
                grad_images3[h, w] = grad[3]

        return (grad_images0, grad_images1, grad_images2, grad_images3)

    def get_direction_sum_from_grad_image(self, grad_image) -> list:
        ''' 获取网格方向线素特征向量
        
        Parameters
        ----------
        grad_image : image 方向图
        
        Returns
        -------
        
        
        '''
        min_w = int(grad_image.shape[1] / self.__frame_count);
        min_h = int(grad_image.shape[0] / self.__frame_count);


        direction_sum = np.zeros((min_h, min_w), dtype=np.float32)
        for i in range(self.__frame_count):
            for j in range(self.__frame_count):
                direction_sum[i, j] = np.sum(a=grad_image[i * min_h : (i + 1) * min_h, j * min_w : (j + 1) * min_w])
        
        ## 将 min_h * min_w 的矩阵，转为一维向量
        return direction_sum.flatten()


    def get_direction_sums(self, image : np.ndarray, show_plt : bool = False, need_dilate : bool = True):
        ''' 获取图像的4个方向线素特征，一共返回8 * 8 * 4 大小的特征向量
        
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
        # 将图像调整为 65x65
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


        # 获取图像边缘
        # image： 输入的图像。通常是灰度图像。如果传入彩色图像，可能需要先将其转换为灰度。
        # threshold1： 较低的阈值，用于确定边缘检测的弱梯度。梯度低于此阈值的区域不会被视为边缘。
        # threshold2： 较高的阈值，用于确定边缘检测的强梯度。梯度高于此阈值的区域肯定会被视为边缘。
        # apertureSize： 可选参数，Sobel 算子的孔径大小，用于计算梯度。默认值为 3。较大的孔径会产生更平滑的边缘。
        # L2gradient： 可选参数，布尔值，表示计算梯度幅度的方法。如果为 True，则使用 L2 范数（即欧几里得范数）计算梯度；如果为 False，则使用 L1 范数。默认值为 False。
        canny_edges = cv.Canny(image = binary_image, threshold1=10, threshold2=250)
        _, canny_edges = cv.threshold(canny_edges, 0.5, 1, cv.THRESH_BINARY)



        # 计算4方向的方向线素
        # 1 2 3
        # 0   4
        # 7 6 5
        grad_images = self.split_grad_image(canny_edges)
        sum_directions = []
        for grad_img in grad_images:
            sum_direction = self.get_direction_sum_from_grad_image(grad_img)
            sum_directions.extend(sum_direction)
        if show_plt:
            print(sum_directions)

        if show_plt:

            # 4. 显示处理后的图像
            plt.figure(figsize=(12, 6))

            plt.subplot(2, 4, 1)
            plt.imshow(image, cmap='gray')
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(2, 4, 2)
            plt.imshow(resized_image, cmap='gray')
            plt.title("resized_image")
            plt.axis("off")

            plt.subplot(2, 4, 3)
            plt.imshow(binary_image, cmap='gray')
            plt.title("binary_image")
            plt.axis("off")

            plt.subplot(2, 4, 4)
            plt.imshow(canny_edges, cmap='gray')
            plt.title("canny_edges ")
            plt.axis("off")



            ## 显示方向线素图
            plt.subplot(2, 4, 5)
            plt.imshow(grad_images[0], cmap='gray')
            plt.title("grad_images0")
            plt.axis("off")

            plt.subplot(2, 4, 6)
            plt.imshow(grad_images[1], cmap='gray')
            plt.title("grad_images1")
            plt.axis("off")

            plt.subplot(2, 4, 7)
            plt.imshow(grad_images[2], cmap='gray')
            plt.title("grad_images2")
            plt.axis("off")

            plt.subplot(2, 4, 8)
            plt.imshow(grad_images[3], cmap='gray')
            plt.title("grad_images3 ")
            plt.axis("off")


            plt.show()

        return sum_directions;


    def __call__(self, img : np.ndarray):
        features = self.get_direction_sums(img)
        features = np.array(features).flatten()
        features = torch.tensor(features, dtype=torch.float32)
        return features


