# -*- encoding: utf-8 -*-
'''
@File		:	pot_online.py
@Time		:	2024/05/21 14:27:50
@Author		:	dan
@Description:	该类读取pot文件，保留笔迹的列表
'''


if __name__ == '__main__':
    import sys
    sys.path.append(".")
from io import BufferedReader
import numpy as np
import struct
import cv2 as cv
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.utils.data import IterableDataset
from img_to_4_direction_transform import ImgTo4DirectionTransform


class PotOnline(object):
    """
    中科院字符文件的封装类
    """

    @property
    def labels(self) -> list[str]:
        '''
        唯一的字符标签列表
        '''
        return self.__labels
    

    @property
    def char_count(self) -> int:
        '''
        样品总数
        '''
        return self.__char_count
    

    @property
    def file_count(self) -> int:
        '''
        pot 文件数量
        '''
        return self.__file_count


    @property
    def current_pot(self) -> BufferedReader:
        '''
        当前已打开的文件
        '''
        return self.__current_pot

    def __init__(self, pot_folder, chineses_only : bool=True, using_nomaliztion=False):
        '''
        pot_folder 存放.pot 文件的文件夹

        chineses_only : bool=True 是否只获取中文字符

        using_nomaliztion=False 归一化手写字体
        '''
        self.__pot_folder = pot_folder
        self.__pot_files = []
        temp_label = []
        char_count = 0
        self.__file_count = 0
        self.__chinese_only = chineses_only

        ## 统计所有pot文件中的 字符总数 和 标签种类
        for file_name in os.listdir(pot_folder):
            if file_name.endswith('.pot'):
                self.__file_count += 1
                file_path = os.path.join(pot_folder, file_name)
                self.__pot_files.append(file_path)
                labels, count = self.caculate_pot_file(file_path)
                temp_label.extend(labels)
                char_count += count

        # for img, tagcode in self.get_data_iter(need_image=False):
        #     labels.add(tagcode);
        #     char_count += 1
        self.__labels = sorted(set(temp_label)) 
        self.__char_count = char_count
        self.__pot_file_index = 0
        self.__using_nomaliztion = using_nomaliztion

        # 当前已打开的文件
        self.__current_pot : BufferedReader = None



    def is_chinese(self, char):
        '''
        是否为中文字符
        '''
        if '\u4e00' <= char <= '\u9fff':
            return True
        else:
            return False


    def caculate_pot_file(self, pot_file : str):
        ''' 
        统计一个pot文件，里面有多少样品、字符种类
        '''
        header = bytes()
        labels, char_count = set(), 0
        with open(pot_file, mode='rb') as file_buffer:
            header = file_buffer.read()
        
        # 分析 buffer
        pre = 0
        while(True):
            tagcode = header[pre + 2] + (header[pre + 3]<<8) + (header[pre + 4]<<16) + (header[pre + 5]<<24)
            stroke_num = header[pre + 6] + (header[pre + 7]<<8)
            pre += 8
            for i in range(stroke_num):
                while True:
                    x1, x2, y1, y2 = header[pre], header[pre + 1], header[pre + 2], header[pre + 3]
                    pre += 4
                    if x1 == 255 and x2 == 255 and y1 == 0 and y2 == 0:
                        break;
            
            pre += 4
            tagcode = struct.pack('>H', tagcode).decode('gbk')
            tagcode = tagcode.replace('\x00', '')
            if self.__chinese_only:
                if self.is_chinese(tagcode):
                    labels.add(tagcode)
                    char_count += 1
            else:
                labels.add(tagcode)
                char_count += 1
            if pre >= len(header):
                break;
    
        return labels, char_count



        


    def draw_stroke(self, pts : np.ndarray, xmin, ymin, x_shift, y_shift):
        '''
        根据pot里面的提取成 x,y,s，，s为笔画对应的主键
        '''
        pt_length = len(pts)
        stroke_start_tag = False
        x_delta, y_delta = -xmin+x_shift, -ymin+y_shift


        s = 0


        i = 1
        points = []
        # 移除不必要的点
        while i < pt_length:
            xi, yi = pts[i][0], pts[i][1]
            xi_1, yi_1 = pts[i - 1][0], pts[i - 1][1]
            if xi == xi_1:
                xi = xi_1 + 1

            if xi == -1 and yi == 0:
                stroke_start_tag = True
                i += 1
                continue
            if stroke_start_tag:
                i += 1
                stroke_start_tag = False
                continue
            
            # (xi_1 + x_delta, yi_1 + y_delta), 
            # (xi + x_delta, yi + y_delta), 

            ## 根据论文，简化图像 https://blog.csdn.net/qq_35337126/article/details/83787766
            Tdist = np.sqrt((xi_1 - xi) ** 2 +  (yi_1 - yi) ** 2)
            if Tdist < 5:
                pts = np.delete(pts, i, 0)
                pt_length -= 1
                continue;
            
            if i > 1:
                xi_2, yi_2 = pts[i - 2][0], pts[i - 2][1]
                if xi_2 == -1 and yi_2 == 0:
                    i += 1
                    continue
                deltaXi_1 = xi_1 - xi_2
                deltaXi = xi - xi_1
                deltaYi_1 = yi_1 - yi_2
                deltaYi = yi - yi_1
                # RuntimeWarning: overflow encountered in scalar multiply
                Tcos = deltaXi_1 * deltaXi + deltaYi * deltaYi_1
                try:
                    if np.sqrt(deltaXi_1 ** 2 + deltaYi_1 ** 2) == 0:
                        print("np.sqrt(deltaXip1 ** 2 + deltaYip1 ** 2)")
                    Tcos = Tcos / (np.sqrt(deltaXi_1 ** 2 + deltaYi_1 ** 2))
                    Tcos = Tcos / (np.sqrt(deltaXi ** 2 + deltaYi ** 2))
                except Exception as ex:
                    print(ex)
                if Tcos > 0.99:
                    # pts.pop(i)
                    pts = np.delete(pts, i, 0)
                    pt_length -= 1
                    continue

            i += 1

        sumPxL = 0
        sumPyL = 0
        sumLen = 0
        stroke_start_tag = False
        lastI = 1
        for i in range(1, len(pts)):

            if pts[i][0] == -1 and pts[i][1] == 0:
                stroke_start_tag = True
                if sumLen == 0:
                    # print(sumLen)
                    continue;
                ux = sumPxL / sumLen
                uy = sumPyL / sumLen

                sumDxL = 0
                for j in range(lastI, i):
                    xj, yj = pts[j][0] + x_delta, pts[j][1] + y_delta
                    xj_1, yj_1 = pts[j - 1][0] + x_delta, pts[j - 1][1] + y_delta
                    if xj == xj_1:
                        xj = xj_1 + 1
                    lenL = np.sqrt((xj - xj_1) ** 2 + (yj - yj_1) ** 2)
                    dxL = 1/3 * lenL * (((xj - ux) ** 2) + ((xj_1 - ux) ** 2) + (xj_1 - ux) * (xj - ux))
                    sumDxL += dxL
                ox = np.sqrt(sumDxL / sumLen)
                if ox == 0:
                    print(ox)

                for j in range(lastI - 1, i):
                    xj, yj = pts[j][0] + x_delta, pts[j][1] + y_delta
                    x_new = (xj - ux) / ox
                    y_new = (yj - uy) / ox
                    points.append((x_new, y_new, s))
                
                sumPxL = 0
                sumPyL = 0
                sumLen = 0
                i += 1
                lastI = i
                s += 1
                continue
            if stroke_start_tag:
                stroke_start_tag = False
                i += 1
                continue
            
            xi, yi = pts[i][0] + x_delta, pts[i][1] + y_delta
            xi_1, yi_1 = pts[i - 1][0] + x_delta, pts[i - 1][1] + y_delta
            if xi == xi_1:
                xi = xi_1 + 1

            lenL = np.sqrt((xi - xi_1) ** 2 + (yi - yi_1) ** 2)
            sumPxL += 1/2 * (xi + xi_1) * lenL
            sumPyL += 1/2 * (yi + yi_1) * lenL
            sumLen += lenL



        L = []
        for i in range(1, len(points)):
            x, y, s = points[i]
            last_x, last_y, last_s = points[i - 1]
            L.append((x, y, x - last_x, y - last_y, 1 if s == last_s else 0, 0 if s == last_s else 1))
        return np.array(L)



    def next_chat_from_buffer(self, file_buffer : BufferedReader):
        '''
        根据文件流逐个读取下一个字符
        '''
        img, tagcode = None, None
        # 文件头，交代了该sample所占的字节数以及label以及笔画数
        header = np.fromfile(file_buffer, dtype='uint8', count=8)
        if not header.size: 
            return None, None
        sample_size = header[0] +(header[1]<<8)
        tagcode = header[2] + (header[3]<<8) + (header[4]<<16) + (header[5]<<24)
        tagcode = struct.pack('>H', tagcode).decode('gbk')
        tagcode = tagcode.replace('\x00', '')
        stroke_num = header[6] + (header[7]<<8)
        img = None
        
        # 以下是参考官方POTView的C++源码View部分的Python解析代码
        traj = []
        xmin, ymin, xmax, ymax = 100000, 100000, 0, 0
        for i in range(stroke_num):
            while True:
                header = np.fromfile(file_buffer, dtype='int16', count=2)
                x, y = header[0], header[1]
                traj.append([x, y])
                
                if x == -1 and y == 0:
                    break
                else:
                    # 个人理解此处的作用是找到描述该字符的采样点的xmin,ymin,xmax,ymax
                    # 但此处若采用源码的逻辑if x < xmin: xmin = x, else if x > xmax: xmax = x会出现了bug
                    # 如果points中x或y是递减的，由于不会执行else判断，会导致xmax或ymax始终为0
                    if x < xmin: xmin = x
                    if x > xmax: xmax = x
                    if y < ymin: ymin = y
                    if y > ymax: ymax = y
        # 最后还一个标志文件结尾的(-1, -1)
        header = np.fromfile(file_buffer, dtype='int16', count=2)
        
        # 根据得到的采样点重构出样本
        x_shift, y_shift = 5, 5 # 画线是有thickness的，所以上下左右多padding几格

        if self.__chinese_only:
            if not self.is_chinese(tagcode):
                return self.next_chat_from_buffer(file_buffer)

        pts = np.array(traj)
        img = self.draw_stroke(pts, xmin, ymin, x_shift, y_shift)
        
        return img, tagcode

    def get_data_iter(self):
        '''
        获取pot图像
        need_image = False 只获取标签，
        need_image = True  标签和图像一起获取
        '''
        pot_dir = self.__pot_folder;
        for file_name in os.listdir(pot_dir):
            if file_name.endswith('.pot'):
                file_path = os.path.join(pot_dir, file_name)
                with open(file_path, 'rb') as buffer:
                    while(True):
                        img, tagcode = self.next_chat_from_buffer(buffer)
                        if img is None and tagcode is None:
                            break;
                        yield img, tagcode


    def close(self):
        ''' 
        复位文件索引，使next()生效
        '''
        self.__pot_file_index = 0 
        if self.__current_pot is not None:
            self.__current_pot.close()
            self.__current_pot = None




    def next(self):
        ''' 
        获取下一个图像   
        need_image = False 只获取标签，
        need_image = True  标签和图像一起获取     
        '''
        if self.__current_pot is None:

            if self.__pot_file_index >= len(self.__pot_files):
                return None, None

            file_path = self.__pot_files[self.__pot_file_index]
            self.__current_pot = open(file_path, 'rb')

        img, tagcode = self.next_chat_from_buffer(file_buffer=self.__current_pot)
        
        # 文件头，交代了该sample所占的字节数以及label以及笔画数
        if img is None and tagcode is None:
            self.__pot_file_index += 1
            self.__current_pot.close()
            self.__current_pot = None
            return self.next()
        
        return img, tagcode


def main():
    pot_folder = "work/Pot1.0/Pot1.0Test.zip_out"
    p = PotOnline(pot_folder=pot_folder, using_nomaliztion=True)
    for img, tagcode in p.get_data_iter():
        # cv.imshow("img", img)
        # cv.waitKey(-1)
        pass

if __name__ == '__main__':

    main()