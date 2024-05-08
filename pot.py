# -*- encoding: utf-8 -*-
'''
@File		:	pot.py
@Time		:	2024/05/07 09:17:55
@Author		:	dan
@Description:	解析 pot 文件的类
'''
from io import BufferedReader
import numpy as np
import struct
import cv2 as cv
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.utils.data import IterableDataset
from img_to_4_direction_transform import ImgTo4DirectionTransform


class Pot(object):
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

    def __init__(self, pot_folder, outter_labels : list[str] = None):
        '''
        pot_folder 存放.pot 文件的文件夹

        outter_labels 训练集的标签
        '''
        self.__pot_folder = pot_folder
        self.__pot_files = []
        temp_label = []
        char_count = 0
        self.__file_count = 0

        ## 统计所有pot文件中的 字符总数 和 标签种类
        for file_name in os.listdir(pot_folder):
            if file_name.endswith('.pot'):
                self.__file_count += 1
                file_path = os.path.join(pot_folder, file_name)
                self.__pot_files.append(file_path)
                labels, count = self.caculate_pot_file(file_path, outter_labels)
                temp_label.extend(labels)
                char_count += count

        # for img, tagcode in self.get_data_iter(need_image=False):
        #     labels.add(tagcode);
        #     char_count += 1
        self.__labels = sorted(set(temp_label)) 
        self.__char_count = char_count
        self.__pot_file_index = 0

        # 当前已打开的文件
        self.__current_pot : BufferedReader = None


    def caculate_pot_file(self, pot_file : str, outter_labels : list[str] = None):
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
            if outter_labels is not None:
                if outter_labels.__contains__(tagcode):
                    labels.add(tagcode)
                else:
                    if pre >= len(header):
                        break;
                    continue
            else:
                labels.add(tagcode)
            char_count += 1
            if pre >= len(header):
                break;
    
        return labels, char_count



        


    def draw_stroke(self, img, pts, xmin, ymin, x_shift, y_shift):
        '''
        根据pot里面的数据，将图像画上画板上
        '''
        pt_length = len(pts)
        stroke_start_tag = False
        for i in range(1, pt_length):
            if pts[i][0] == -1 and pts[i][1] == 0:
                stroke_start_tag = True
                continue
            if stroke_start_tag:
                stroke_start_tag = False
                continue
            x_delta, y_delta = -xmin+x_shift, -ymin+y_shift
            cv.line(img, (pts[i-1][0]+x_delta, pts[i-1][1]+y_delta), (pts[i][0]+x_delta, pts[i][1]+y_delta), color=(0, 0, 0), thickness=5)
        return img



    def next_chat_from_buffer(self, file_buffer : BufferedReader, need_image = True):
        '''
        根据文件流逐个读取下一个字符
        '''
        img, tagcode = None, None

        if need_image:

            # 文件头，交代了该sample所占的字节数以及label以及笔画数
            header = np.fromfile(file_buffer, dtype='uint8', count=8)
            if not header.size: 
                return None, None
            sample_size = header[0] +(header[1]<<8)
            tagcode = header[2] + (header[3]<<8) + (header[4]<<16) + (header[5]<<24)
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

            canva = np.ones((ymax-ymin+2*y_shift, xmax-xmin+2*x_shift), dtype=np.uint8)*255
            pts = np.array(traj)
            img = self.draw_stroke(canva, pts, xmin, ymin, x_shift, y_shift)
            

            tagcode = struct.pack('>H', tagcode).decode('gbk')
            tagcode = tagcode.replace('\x00', '')
        else:
            # 文件头，交代了该sample所占的字节数以及label以及笔画数
            header = np.fromfile(file_buffer, dtype='uint8', count=8)
            if not header.size: 
                return None, None
            tagcode = header[2] + (header[3]<<8) + (header[4]<<16) + (header[5]<<24)
            stroke_num = header[6] + (header[7]<<8)
            img = None
            for i in range(stroke_num):
                while True:
                    x1, x2, y1, y2 = file_buffer.read(4)
                    if x1 == 255 and x2 == 255 and y1 == 0 and y2 == 0:
                        break;
            file_buffer.read(4)
            tagcode = struct.pack('>H', tagcode).decode('gbk')
            tagcode = tagcode.replace('\x00', '')
            

        return img, tagcode

    def get_data_iter(self, need_image = True):
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
                        img, tagcode = self.next_chat_from_buffer(buffer, need_image)
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




    def next(self, need_image : bool = True):
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

        img, tagcode = self.next_chat_from_buffer(file_buffer=self.__current_pot, need_image=need_image)
        
        # 文件头，交代了该sample所占的字节数以及label以及笔画数
        if img is None and tagcode is None:
            self.__pot_file_index += 1
            self.__current_pot.close()
            self.__current_pot = None
            return self.next()
        
        return img, tagcode
