# -*- encoding: utf-8 -*-
'''
@File		:	matplot_tools.py
@Time		:	2024/05/24 09:01:04
@Author		:	dan
@Description:	matplot工具类
'''

import matplotlib.pyplot as plt


def draw_plot(test_loss, test_correct, train_loss, train_correct, filename):
    ''' 
    '''
    x = list(range(len(test_loss)))
    plt.figure()
    plt.plot(x, test_loss, 'r')
    plt.plot(x, test_correct, 'b')
    plt.plot(x, train_loss)
    plt.plot(x, train_correct)
    plt.legend(['test_loss', 'test_correct', 'train_loss', 'train_correct'], loc='upper left')
    plt.savefig(f'pretrain/{filename}')


def main():
    test_loss_list = [1, 2, 3, 4]
    test_correct_list = [0.1, 0.2, 0.3, 0.4]
    train_loss_list = [6,7,8,9]
    train_correct_list = [0.2 , 0.3, 0.4, 0.5]
    draw_plot(test_loss=test_loss_list, test_correct=test_correct_list, train_loss=train_loss_list, train_correct=train_correct_list, filename="test.jpg") 
    pass

if __name__ == '__main__':
    main()