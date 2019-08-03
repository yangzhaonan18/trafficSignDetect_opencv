# -*- codint=utf-8 -*-


import numpy as np


def expand_yellow(list):
    list = np.array(list)
    print(list.shape)
    unite_list = []
    for i in range(len(list)):
        if list[i][2] > 10 and list[i][3] > 10:
            if list[i][2] < list[i][3]:  # 黄色区域的宽度 小于 高度的时候，将不变看成是 黄色施工标志进行尺寸的调整
                list[i][0] = int(list[i][0] - list[i][2] * 0.2)
                # list[i][1] = int(list[i][1] - list[i][3] * 0.2)
                list[i][2] = int(list[i][2] * 1.8)
                # list[i][3] = int(list[i][3] * 1.3)
            unite_list.append(list[i])
    return unite_list

def expand_blue(list):
    list = np.array(list)
    print(list.shape)
    unite_list = []
    for i in range(len(list)):
        if list[i][2] > 5 and list[i][3] > 5:
            if list[i][2] < list[i][3]:  # 黄色区域的宽度 小于 高度的时候，将不变看成是 黄色施工标志进行尺寸的调整
                list[i][0] = int(list[i][0] - list[i][2] * 0.2)
                list[i][1] = int(list[i][1] - list[i][3] * 0.2)
                list[i][2] = int(list[i][2] * 1.5)
                list[i][3] = int(list[i][3] * 1.5)
            unite_list.append(list[i])
    return unite_list
