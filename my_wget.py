# -*- encoding: utf-8 -*-
'''
@File		:	my_wget.py
@Time		:	2024/05/09 08:56:23
@Author		:	dan
@Description:	用原生python实现下载功能
'''

import os  # 导入os库
import urllib.request  # 导入urllib库
import pathlib

#######文件下载
def get(url, file_path):
    # 文件基准路径
    basedir = pathlib.Path(file_path).parent.absolute()
    # 如果没有这个path则直接创建
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    # 下载到服务器的地址
    if not os.path.exists(file_path):  # 判断是否存在文件
        # 文件url
        try:
            urllib.request.urlretrieve(url, filename=file_path)
        except IOError as ex:  # 设置抛出异常
            print(1, ex)
            raise ex
        except Exception as ex:  # 设置抛出异常
            print(2, ex)
            raise ex



def main():
    p = "work/PotSimpleTest"
    files = []
    for file_name in os.listdir(p):
            if file_name.endswith('.pot'):
                files.append(file_name)
    print(f"\"{"\",\"".join(files)}\"")
    pass

if __name__ == '__main__':
    main()