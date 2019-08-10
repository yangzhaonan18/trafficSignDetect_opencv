# -*- coding=utf-8 -*-
# py37

#  找到满足条件的两个矩形，合并矩形
import numpy as np

# from nms import nms


# 计算交并比
def cal_IOU(rectA, rectB):
    maxX = max(rectA[0], rectB[0])
    minX = min(rectA[0] + rectA[2], rectB[0] + rectB[2])
    maxY = max(rectA[1], rectB[1])
    minY = min(rectA[1] + rectA[3], rectB[1] + rectB[3])

    area_I = max((minX - maxX), 0) * max((minY - maxY), 0)
    area_U = max(rectA[2] * rectA[3] + rectB[2] * rectB[3] - area_I, 1)  # 保证分母不能是0，以防意外
    IOU = area_I / area_U
    return IOU

#  合并两个矩形框
def unite_2rect(rectA, rectB):
    rectAB = [0, 0, 0, 0]
    rectAB[0] = min(rectA[0], rectB[0])
    rectAB[1] = min(rectA[1], rectB[1])
    rectAB[2] = max(rectA[0] + rectA[2], rectB[0] + rectB[2]) - min(rectA[0], rectB[0])
    rectAB[3] = max(rectA[1] + rectA[3], rectB[1] + rectB[3]) - min(rectA[1], rectB[1])

    # 将两个白色的合并成一个大的之后，需要扩充一下边界
    lineWidth = int(0.15 * ((rectAB[2] + rectAB[3]) / 2))
    rectAB[0] = max(rectAB[0] - lineWidth, 0)  # 最小不能小于零
    rectAB[1] = max(rectAB[1] - lineWidth, 0)
    inf = float("inf")
    rectAB[2] = min(rectAB[2] + 2 * lineWidth, inf)  # 最大不能大过 图像的尺寸 (这里的程序还没有写完，需要补充)
    rectAB[3] = min(rectAB[3] + 2 * lineWidth, inf)

    return rectAB

# 找到 四个矩形框中的第四个矩形框的信息
def findFourthPoint(rectA, rectB, rectC):
    width = (rectA[2] + rectB[2] + rectC[2]) / 3
    high = (rectA[3] + rectB[3] + rectC[3]) / 3
    # 计算三个区域的中心点的坐标
    rectAP = [rectA[0] + rectA[2] / 2, rectA[1] + rectA[3] / 2]
    rectBP = [rectB[0] + rectB[2] / 2, rectB[1] + rectB[3] / 2]
    rectCP = [rectC[0] + rectC[2] / 2, rectC[1] + rectC[3] / 2]

    # 第四个点的中心点的坐标
    rectDP = [0, 0]
    rectD = [0, 0, 0, 0]
    # 计算三个点，任意两点之间的距离平方
    distance_AB = sum([pow(rectAP[i] - rectBP[i], 2) for i in range(len(rectAP))])
    distance_AC = sum([pow(rectAP[i] - rectCP[i], 2) for i in range(len(rectAP))])
    distance_BC = sum([pow(rectBP[i] - rectCP[i], 2) for i in range(len(rectAP))])

    # 找到距离最长的那个边
    if distance_AB > distance_AC and distance_AB > distance_BC:
        rectDP[0] = rectAP[0] + rectBP[0] - rectCP[0]
        rectDP[1] = rectAP[1] + rectBP[1] - rectCP[1]
    elif distance_BC > distance_AB and distance_BC > distance_AC:
        rectDP[0] = - rectAP[0] + rectBP[0] + rectCP[0]
        rectDP[1] = - rectAP[1] + rectBP[1] + rectCP[1]
    else:  # if distance_AC > distance_BC and distance_AB > distance_AB:
        rectDP[0] = rectAP[0] - rectBP[0] + rectCP[0]
        rectDP[1] = rectAP[1] - rectBP[1] + rectCP[1]

    # 计算第四个点的 矩形坐标
    rectD[0] = int(rectDP[0] - width / 2)
    rectD[1] = int(rectDP[1] - width / 2)
    rectD[2] = int(width)
    rectD[3] = int(high)

    return rectD


# 合并四个矩形框，即求四个矩形框的外界矩形
def unite_rect(rectA, rectB, rectC, color):
    rectABC = [0, 0, 0, 0]
    if color == "yellow":  # 直接合并这三个候选区域
        rectABC[0] = min(rectA[0], rectB[0], rectC[0])
        rectABC[1] = min(rectA[1], rectB[1], rectC[1])
        rectABC[2] = max(rectA[0] + rectA[2], rectB[0] + rectB[2], rectC[0] + rectC[2]) - min(rectA[0], rectB[0],
                                                                                              rectC[0])
        rectABC[3] = max(rectA[1] + rectA[3], rectB[1] + rectB[3], rectC[1] + rectC[3]) - min(rectA[1], rectB[1],
                                                                                              rectC[1])
    elif color == "blue":  # 找到第四个区域中中心并合并，四个区域
        rectD = findFourthPoint(rectA, rectB, rectC)
        rectABC[0] = min(rectA[0], rectB[0], rectC[0], rectD[0]) - 1
        rectABC[1] = min(rectA[1], rectB[1], rectC[1], rectD[1]) - 1
        rectABC[2] = max(rectA[0] + rectA[2], rectB[0] + rectB[2], rectC[0] + rectC[2], rectD[0] + rectD[2]) - min(
            rectA[0], rectB[0], rectC[0], rectD[0]) + 1
        rectABC[3] = max(rectA[1] + rectA[3], rectB[1] + rectB[3], rectC[1] + rectC[3], rectD[1] + rectD[3]) - min(
            rectA[1], rectB[1], rectC[1], rectD[1]) + 1


        # rectABC[0] = int(rectABC[0] - rectABC[2] * 0.1)
        # rectABC[1] = int(rectABC[1] - rectABC[3] * 0.1)
        # rectABC[2] = int(rectABC[2] * 1.2)
        # rectABC[3] = int(rectABC[3] * 1.2)

    return rectABC


# 判断A矩形框是否在B矩形框中，
def AinB(A, B):  # A在B中，A比B小的情况
    flag = False
    # 判断的时候一定要有等号
    if A[0] >= B[0] and A[1] >= B[1] and (A[0] + A[2]) <= (B[0] + B[2]) and (A[1] + A[3]) <= (B[1] + B[3]):
        flag = True
    return flag


# 根据矩形框的长宽，来判断是否可能是交通标志
def haveSignSize(rectA, rectB):  # 各个部分 形状相似
    flag_1 = False  # 判断形状的标志
    flag_2 = False  # 判断是否包含的标志
    if max(round(rectA[2] / rectA[3]), round(rectA[3] / rectA[2])) in [1, 2]:  # 第一个矩形比较方正
        if max(round(rectB[2] / rectB[3]), round(rectB[3] / rectB[2])) in [1, 2]:  # 第二个矩形比较方正
            if max(round((rectA[2] * rectA[3]) / (rectB[2] * rectB[3])),  # 第一个和第二个矩形之间的面积比较相似
                   round((rectB[2] * rectB[3]) / (rectA[2] * rectA[3]))) in [1, 2, 3,
                                                                             4]:  # 设置成4是 为了将已经合并的三个blue 同最后一个 blue 合并
                flag_1 = True
    if not AinB(rectA, rectB) and not AinB(rectB, rectA):  # 不存在 AB之间是相互包含的额关系
        flag_2 = True
    return flag_1 and flag_2


# 判断AB是否可能合并
def can_Unite(rectA, rectB, iou):  # 两个区域之间有交集，面积基本相似就认为是满足合并的条件
    flag = False
    if cal_IOU(rectA, rectB) > iou and haveSignSize(rectA, rectB):
        flag = True
    return flag


# 合并两个白色矩形框
def unite2white(list, iou):
    # iou = 0.1 white
    list = np.array(list)
    unite_list = []  # 合并后的目标存放在这里面
    index_list = []  # 将需要合并的下标，存放在这里面
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            if can_Unite(list[i], list[j], iou):
                index_list.append(i)
                index_list.append(j)
                unite_list.append(unite_2rect(list[i], list[j]))  # 满足条件的两个矩形合并

    for i in range(len(list)):
        if i not in index_list:  # 将单独的、不需要合并的区域加入，候选区域中。
            unite_list.append(list[i])

    return unite_list


# 遍历在所有的矩形框中，找到满足合并条件的四个矩形框
def unite4(list, iou, color):
    list = np.array(list)
    print(list.shape)
    unite_list = []
    index_list = []
    for i in range(len(list)):  # i j k m
        for j in range(i + 1, len(list)):
            if can_Unite(list[i], list[j], iou):  # ij 相邻
                for k in range(len(list)):  # 再从所有的目标里面找一个k, 这个k 必须满足和i,j 中的一个是相邻的。
                    if k != i and k != j and (can_Unite(list[k], list[i], iou) or can_Unite(list[k], list[j], iou)):
                        print("i, j, k", i, j, k)

                        unite_list.append(unite_rect(list[i], list[j], list[k], color))  # 此时，ij 相邻，ik相邻，合并

                        # 将已经合并使用过的目标index 存储起来，后面不在考虑这些目标区域了。
                        index_list.append(i)
                        index_list.append(j)
                        index_list.append(k)
                        # 将已经合并使用的目标，重置为0，避免后续遍历时，被重复使用。
                        list[i] = [0, 0, 0, 0]
                        list[j] = [0, 0, 0, 0]
                        list[k] = [0, 0, 0, 0]
    # for i in range(len(list)):
    #     if i not in index_list:  # 将单独的、不需要合并的区域加入，候选区域中。
    #         unite_list.append(list[i])

    return unite_list  # 将能合并的目标区域都进行了合并


# 合并所有矩形框中的四个蓝色的矩形框 禁止停止标志
def unite4blue(list, iou):
    unite_list = unite4(list, iou, color="blue")
    unite_list = unite2white(unite_list, iou)  # 将最后的的两个大蓝色框合并成 一个完整的蓝色框。
    return unite_list


#  合并所有矩形框中的四个黄色的矩形框 向左 向右形式标志
def unite4yellow(list, iou):
    unite_list = unite4(list, iou, color="yellow")
    return unite_list


if __name__ == "__main__":
    list = [[70, 487, 11, 10], [303, 483, 9, 15], [1050, 480, 13, 18], [677, 426, 26, 11], [640, 331, 18, 9],
            [633, 336, 11, 7], [848, 283, 16, 11], [178, 281, 75, 34], [148, 283, 32, 24], [839, 271, 12, 17],
            [859, 270, 12, 16], [846, 262, 16, 12], [20, 106, 12, 10], [245, 104, 19, 13], [230, 92, 11, 10],
            [21, 1, 1038, 334], [1042, 1, 237, 205], [60, 164, 16, 23], [98, 68, 11, 15], [88, 53, 15, 12],
            [922, 13, 38, 17]]

    unite_list = unite4blue(list, iou=0.001)

    print(unite_list)
    print(np.array(unite_list).shape)
