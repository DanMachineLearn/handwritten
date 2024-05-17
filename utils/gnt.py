
from io import BufferedReader
import numpy as np
import struct
import cv2 as cv
from torch.utils.data import Dataset
import glob
import os

## 中科院字符文件的封装类
class Gnt(object):
    """
    A .gnt file may contains many images and charactors
    """

    def __init__(self, gnt_file, chinese_only = True):
        self.__gnt_file = gnt_file
        self.__buffer : BufferedReader = None
        self.__chinese_only = chinese_only

    def get_data_iter(self):
        '''
        获取iter对象
        '''
        
        with open(self.__gnt_file, 'rb') as buffer:
            while True:
                image, tagcode = self.get_image_from_buffer(buffer=buffer)
                if image is None and tagcode is None:
                    break;
                yield image, tagcode


    def caculate_gnt_file(self):
        ''' 
        统计该文件拥有的字符种类和数量
        '''
        header = bytes()
        labels, char_count = set(), 0
        with open(self.__gnt_file, mode='rb') as file_buffer:
            header = file_buffer.read()


        pre = 0
        while(True):
            sample_size = header[pre + 0] + (header[pre + 1] << 8) + (header[pre + 2] << 16) + (header[pre + 3] << 24)
            tagcode = header[pre + 5] + (header[pre + 4] << 8 )
            width = header[pre + 6] + (header[pre + 7] << 8)
            height = header[pre + 8] + (header[pre + 9] << 8)
            pre += 10
            if 10 + width * height != sample_size:
                break;
            pre += width * height

            try:
                tagcode = struct.pack('>H', tagcode).decode('gb2312')
                tagcode = tagcode.replace('\x00', '')
            except:
                continue
            
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


    def close(self):
        ''' 
        复位文件索引，使next()生效
        '''
        if self.__buffer is not None:
            self.__buffer.close()
            self.__buffer = None


    def next(self):
        ''' 
        获取下一个图像   
        '''
        if self.__buffer is None:
            self.__buffer = open(self.__gnt_file, 'rb')

        img, tagcode = self.get_image_from_buffer(buffer=self.__buffer)
        
        return img, tagcode

    def get_image_from_buffer(self, buffer : BufferedReader):
        ''' 
        '''
        header_size = 10
        header = np.fromfile(buffer, dtype='uint8', count=header_size)
        if not header.size:
            return None, None
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tagcode = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        if header_size + width * height != sample_size:
            return None, None
        image = np.fromfile(buffer, dtype='uint8', count=width * height).reshape((height, width))


        try:
            tagcode = struct.pack('>H', tagcode).decode('gb2312')
            tagcode = tagcode.replace('\x00', '')
        except:
            return self.get_image_from_buffer(buffer=buffer)

        if self.__chinese_only:
            if self.is_chinese(tagcode):
                return image, tagcode
            else:
                return self.get_image_from_buffer(buffer=buffer)
        else:
            return image, tagcode

        return image, tagcode

    

    def is_chinese(self, char):
        '''
        是否为中文字符
        '''
        if '\u4e00' <= char <= '\u9fff':
            return True
        else:
            return False

def main():

    import alive_progress
    gnt_file = "work\\Gnt1.0\\Gnt1.0Test.zip_out\\006-t.gnt"
    gnt = Gnt(gnt_file)
    labels, char_count = gnt.caculate_gnt_file()

    with alive_progress.alive_bar(char_count) as bar:
        for image, tagcode in gnt.get_data_iter():
            # cv.imshow(tagcode, image)
            # cv.waitKey(-1)
            bar()
    pass

if __name__ == '__main__':
    main()