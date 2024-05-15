# -*- encoding: utf-8 -*-
'''
@File		:	progress.py
@Time		:	2024/05/08 09:30:58
@Author		:	dan
@Description:	一个简单例子，显示 alive_progress 的用法
'''


from alive_progress import alive_bar
import time

# retrieve your set of items

items = range(100)

# declare your expected total

with alive_bar(len(items)) as bar:

    # iterate as usual

    for item in items:

        # process each item

        bar()

        time.sleep(0.1)