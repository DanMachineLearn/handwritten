# -*- encoding: utf-8 -*-
'''
@File		:	pot_downloader.py
@Time		:	2024/05/09 09:21:08
@Author		:	dan
@Description:	专门负责下载线上pot文件的类
'''


if __name__ == '__main__':
    import sys 
    sys.path.append(".")
import os
from alive_progress import alive_bar
import utils.my_wget as my_wget
import zipfile


class PotDownloader :
    '''
    专门下载pot文件的下载器
    '''

    

    FILE_LIST = {
        "Pot1.0": ["Pot1.0Test.zip","Pot1.0Train.zip"],
        "Pot1.1": ["Pot1.1Test.zip","Pot1.1Train.zip"],
        "Pot1.2": ["Pot1.2Test.zip","Pot1.2Train.zip"],
    }

    def __init__(self, download_url : str = 'https://gonsin-common.oss-cn-shenzhen.aliyuncs.com/handwritten', base_dir = "work") -> None:
        ''' 
        
        Parameters
        ----------
        download_url 基本的下载地址
        
        base_dir 基本的本地地址
        
        '''
        self.__base_dir = base_dir
        self.__download_url = download_url
        # 遍历 pot 文件数量
        file_count = 0
        file_list = []
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        file_list = self.get_filelist(base_dir, file_list, '.pot')
        self.__file_count = len(file_list)


    def get_filelist(self, dir, file_list : list, ext : str = None):
        '''
        递归遍历文件夹中所有文件
        dir 需要遍历的文件夹
        
        file_list : list  文件列表，用于递归
        
        ext : str  文件后缀
        '''
        new_dir = dir
        if os.path.isfile(dir):
            if ext is None:
                file_list.append(dir)
            else:
                if dir.endswith(ext):   
                    file_list.append(dir)
            # # 若只是要返回文件文，使用这个
            # Filelist.append(os.path.basename(dir))
        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                # 如果需要忽略某些文件夹，使用以下代码
                #if s == "xxx":
                    #continue
                new_dir = os.path.join(dir,s)
                self.get_filelist(new_dir, file_list)
        return file_list

    def start(self):
        ''' 
        下载放在阿里云上面的pot文件，方便随时训练
        
        Parameters
        ----------
        download_url : str 文件下载的基准网址
        '''

        ''' 
        下载放在阿里云上面的pot文件，方便随时训练
        
        Parameters
        ----------
        download_url : str 文件下载的基准网址
        '''

        # TODO 
        
        for key in PotDownloader.FILE_LIST:
            files = PotDownloader.FILE_LIST[key]
            print(f"开始下载{key}数据集")
            with alive_bar(len(files)) as bar:
                for f in files:
                    full_path = f"{self.__base_dir}/{key}/{f}"
                    if os.path.isfile(full_path):
                        bar()
                        continue;
                    full_url = f"{self.__download_url}/Pot/{f}"
                    my_wget.get(full_url, full_path)
                    bar()
            print(f"下载完毕")


        for key in PotDownloader.FILE_LIST:
            files = PotDownloader.FILE_LIST[key]
            print(f"开始解压{key}数据集")
            with alive_bar(len(files)) as bar:
                for f in files:
                    zip_path = f"{self.__base_dir}/{key}/{f}"
                    out_path = f"{self.__base_dir}/{key}/{f}_out"
                    if os.path.isdir(out_path):
                        bar()
                        continue;
                    zip_file = zipfile.ZipFile(zip_path)
                    zip_extract = zip_file.extractall(path=out_path)
                    zip_file.close()
                    bar()
            print(f"解压完毕")


def main():
    PotDownloader().start()
    pass

if __name__ == '__main__':
    main()