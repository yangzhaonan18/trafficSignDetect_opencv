# -*- coding = utf-8 -*-
# 时间：2019年7月29日20:28:06
# 程序名称：交通标志检测
# 程序作用：使用 OpenCV 来提取交通标志,之后再使用SVM或者深度学习的方法来进行分类。


import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from water import find_water_objects

from uniteRect import unite2white, unite4blue, unite4yellow
from expand_rect import expand_yellow, expand_blue
from nms import nms

def drawSign(img, rectangleList):
    for i in range(len(rectangleList)):
        x = rectangleList[i][0]
        y = rectangleList[i][1]
        w = rectangleList[i][2]
        h = rectangleList[i][3]
        crop_img = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 对裁剪出来的区域进行分类（判断是否是所需的检测对象）
    cv2.imshow("img  with rectangle", img)
    cv2.waitKey(0)


def setColor(img):
    """
    对输入图像进行处理，现在只添加了色域转换，后面还要添加调节图片的饱和度、对比度等信息，方便进行mask 的提取。
    :param img: 原始图片
    :return:  处理后的图片
    """
    kernel_size = 5
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_hsv


def findMaskByColor(img_hsv, color):
    """
    :param img_hsv:  颜色通道转换成 HSV 之后的图片
    :param color: 需要提取的 sign 的颜色
    :return: 满足 颜色要求的 mask
    """

    if color == "red":
        # 设置红色标志的阈值
        redLower01 = np.array([0, 43, 46], dtype=np.uint8)  # 部分红
        redUpper01 = np.array([10, 255, 255], dtype=np.uint8)
        redLower02 = np.array([125, 43, 46], dtype=np.uint8)  # 部分红  包含了紫色
        redUpper02 = np.array([180, 255, 255], dtype=np.uint8)
        red_mask01 = cv2.inRange(img_hsv, redLower01, redUpper01)
        red_mask02 = cv2.inRange(img_hsv, redLower02, redUpper02)
        mask = red_mask01 + red_mask02

    elif color == "blue":
        # 设置蓝色标志的阈值
        Lower = np.array([100, 100, 46], dtype=np.uint8)  # blue  S 的值设置高一点，提取出的颜色重一点
        Upper = np.array([124, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, Lower, Upper)

    elif color == "white":
        Lower = np.array([0, 0, 150], dtype=np.uint8)
        Upper = np.array([180, 43, 255], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, Lower, Upper)
    elif color == "orange":
        Lower = np.array([11, 43, 221], dtype=np.uint8)
        Upper = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, Lower, Upper)
    elif color == "yellow":  # 黄色的图片会 偏橙色
        Lower = np.array([11, 100, 46], dtype=np.uint8)
        Upper = np.array([34, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, Lower, Upper)
    elif color == "black":
        Lower = np.array([0, 0, 0], dtype=np.uint8)
        Upper = np.array([180, 255, 60], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, Lower, Upper)




    else:
        mask = None
    return mask


def findContours(img, color, mask):
    """
    :param img: 原始的BGR 图片
    :param mask: 符合当前颜色要求的sign 的 mask
    :return: contours 轮廓
    """
    # 对感兴趣的图形区域求取 凸图形（分水岭的对象是）
    BinColor = cv2.bitwise_and(img, img, mask=mask)  # 原来的彩色图和 mask 与运算得到的彩图（大部分区域时黑色的）。

    # 形态学操作 先开后闭。在这里开运算（腐蚀掉白色区域）比闭运算更重要。
    kernel = np.ones((3, 3), np.uint8)
    BinColor = cv2.morphologyEx(BinColor, cv2.MORPH_CLOSE, kernel)  # 闭运算
    if color != "white":
        kernel = np.ones((3, 3), np.uint8)
        BinColor = cv2.morphologyEx(BinColor, cv2.MORPH_OPEN, kernel)  # 开运算
    if color == "white":
        kernel = np.eye(3, dtype=np.uint8)
        BinColor = cv2.morphologyEx(BinColor, cv2.MORPH_CLOSE, kernel)  # 腐蚀ERODE
    if color == "blue":
        kernel = np.ones((3, 3), np.uint8)
        BinColor = cv2.morphologyEx(BinColor, cv2.MORPH_CLOSE, kernel, iterations=1)  # 闭运算
    if color == "yellow":
        kernel = np.ones((3, 3), np.uint8)
        BinColor = cv2.morphologyEx(BinColor, cv2.MORPH_DILATE, kernel, iterations=2)  # 膨胀

    gray = cv2.cvtColor(BinColor, cv2.COLOR_BGR2GRAY)  # 转成灰色图像
    ret, BinThings = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # 灰色图像二值化（变黑白图像）
    if color == "yellow":
        cv2.imshow("asdf", BinThings)
        cv2.waitKey(0)

    _, contours, hierarchy = cv2.findContours(BinThings, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours, hierarchy = cv2.findContours(BinThings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 只要外面那一层的轮廓 contours

    return contours


# 根据颜色提取出可能是标志的目标
def getSign(img, color):
    """
    :param img:  原始的输入图片
    :param color:  需要提取的sign 的颜色
    :return: 可能是sign 的区域的坐标列表，[x, y, w, h]
    """
    img_hsv = setColor(img)
    mask = findMaskByColor(img_hsv, color=color)  # find the mask of a color
    contours = findContours(img, color=color, mask=mask)
    img_copy = img.copy()

    rectangleList = []
    img_black = np.zeros_like(img)
    for i in range(len(contours)):  # 绘制每一个轮廓
        cv2.drawContours(img_black, [contours[i]], -1, (255, 255, 255), -1)  # 使用黑色作为背景，白色最为感兴趣的区域。img_black 只用作分水岭算法使用。
        cv2.drawContours(img_copy, [contours[i]], -1, (2, 0, 155), 2)  # 使用黑色作为背景，白色最为感兴趣的区域。img_black 只用作分水岭算法使用。


        x, y, w, h = cv2.boundingRect(contours[i])
        # 只添加满足一定条件的矩形 round是四舍五入 注意这里没有考虑 遮挡和处于图像边缘的目标（
        # 换句话说，遮挡和边缘遮挡目标还无法检测到。）。
        if max(round(w / h), round(h / w)) in [1, 2, 3] and w >= 5 and h >= 5:
            if color in ["red", "white"] and w >= 10 and h >= 10 and max(round(w / h), round(h / w)) == 1:  # 1:1 方正的区域直接作为候选区域
                rectangleList.append([x, y, w, h])
                # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 在原图上画出 目标框来  显示使用
            elif color == "yellow" and w >= 10 and h >= 10:
                rectangleList.append([x, y, w, h])
            elif color == "blue" and w >= 3 and h >= 3 and max(round(w / h), round(h / w)) in [1, 2]:
                rectangleList.append([x, y, w, h])

            elif color == "red" and w >= 10 and h >= 10 and max(round(w / h), round(h / w)) in [2, 3, 4, 5]:
                # 长条形的区域，使用分水岭算法后，分割成多部分，作为候选区域。
                # 白色的就只存在方正的情况

                # 使用分水岭算法将物体区分开来
                crop_img = img_black[y:y + h, x:x + w]  # 截取部分区域之后，进行分水岭算法
                rectangleList_water = find_water_objects(crop_img)  # 得到的坐标是相对于crop_img的坐标，最终定位需要的是相对于原图的绝对坐标
                for j in range(len(rectangleList_water)):  # 将相对坐标转化成相对于原图的绝对坐标
                    rectangleList_water[j][0] += x
                    rectangleList_water[j][1] += y
                rectangleList.extend(rectangleList_water)
    # cv2.imshow("img_copy", img_copy)
    return rectangleList


# 读取图片
# img_path = "image//coin2.png"
img_path = "image//100.png"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# 找到红色区域的坐标
colors = ["red", "blue", "orange", "yellow", "black", "white"]
# 提取红色的情况
rectangleList_red01 = getSign(img, color="red")  # 红色在提取目标时，会产生四个 蓝色的框，需要将这四个框合并
rectangleList_red02 = unite2white(rectangleList_red01, iou=0.01)  # 提取红色时，会产生两个红色的框， 也需要处理
rectangleList_red04 = unite4blue(rectangleList_red01, 0.0001)  # 只保留的合并的区域，单独的小区域不保留了

# rectangleList_red01.extend(rectangleList_red02)  # 添加 二合一的结果
# rectangleList_red01.extend(rectangleList_red04)  # 添加 四合一的结果
rectangleList_red = rectangleList_red01
# rectangleList_red = nms(rectangleList_red01, iou=0.5, type="removeIn")


# 提取白色的情况
rectangleList = getSign(img, color="white")
# rectangleList_white = rectangleList
rectangleList_white = unite2white(rectangleList, iou=0.01)  # 合并被分成两种白色的情况

# 提取蓝色的情况
rectangleList_blue01 = getSign(img, color="blue")  # 单个blue
rectangleList_blue = rectangleList_blue01
rectangleList_blue01_ = expand_blue(rectangleList_blue01)
rectangleList_blue04 = unite4blue(rectangleList_blue01_, 0.0001)  # 四合一，这里只保留其中最红合并的那个，其他的都不保留。
rectangleList_blue01.extend(rectangleList_blue04)  # 将 合并后的大的 添加到 原来的有很多小的区域里去，
rectangleList_blue = nms(rectangleList_blue01, iou=0.5, type="removeIn")


# # 提取黄色的情况
rectangleList_yellow01 = getSign(img, color="yellow")  # 提取的黄色标志分为：向左向右行驶，ETC收费站，黄色施工标志
rectangleList_yellow = rectangleList_yellow01
rectangleList_yellow01_ = expand_yellow(rectangleList_yellow01)  # 将 宽度小于高度的 黄色标志 放大一点
rectangleList_yellow04 = unite4yellow(rectangleList_yellow01_, 0.001)
rectangleList_yellow01.extend(rectangleList_yellow04)
rectangleList_yellow = nms(rectangleList_yellow01, iou=0.1, type="removeIn")


rectangleList_add = []
# rectangleList_add.extend(rectangleList_red)
# rectangleList_add.extend(rectangleList_blue)
#
rectangleList_add.extend(rectangleList_yellow)
# rectangleList_add.extend(rectangleList_white)

print("The number of rectangle is ", len(rectangleList))

# 对候选区域进行非极大抑制处理，过滤掉重复的候选区域 NMS (其实就是去除重复的候选区域，去除候选区域中间的候选区域)
rectangleList_nms = rectangleList_add
rectangleList_nms = nms(rectangleList_add, iou=0.5, type="removeIou")


# 裁剪出区域

# 对裁剪出的区域进行分类

drawSign(img, rectangleList_nms)

print("THE END")
