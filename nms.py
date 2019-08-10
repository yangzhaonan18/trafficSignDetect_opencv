# -*- coding=utf-8 -*-
# py37
#  非极大值抑制，用于最后的的矩形框赛选、排除

from uniteRect import cal_IOU, AinB


def area(rect):
    return rect[2] * rect[3]


def nms(list, iou, type="removeIn"):
    newList = []
    index_remove = []

    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            # 去除掉IOU 较高的候选区域
            if type == "removeIou":
                # 满足IOU，且两者 不存在包含的关系
                if cal_IOU(list[i], list[j]) > iou and not AinB(list[i], list[j]) and not AinB(list[j], list[i]):
                    # if area(list[i]) > area(list[j]):
                    # 保留两个中比较方正的那个，删除矩形的那个
                    if max(list[i][2] / list[i][3], list[i][3] / list[i][2]) < max(list[j][2] / list[j][3],
                                                                                   list[j][3] / list[j][2]):
                        index_remove.append(j)
                    else:
                        index_remove.append(i)
                    pass
                else:  # 去除掉 候选区域中间的区域
                    if cal_IOU(list[i], list[j]) > 0.8:
                        # 两者存在包含关系的时候 删除 小的那个
                        if AinB(list[i], list[j]):
                            index_remove.append(i)
                        if AinB(list[j], list[i]):
                            index_remove.append(j)

    for i in range(len(list)):
        if list[i][2] < 10 or list[i][3] < 10:
            index_remove.append(i)

    for i in range(len(list)):
        if i not in index_remove:
            newList.append(list[i])

    print("after len(list) = ", len(newList))
    return newList
